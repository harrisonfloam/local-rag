import json
import logging
from typing import List

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.schemas import (
    ChatCompletionWithSources,
    ChatRequest,
    CollectionInfoResponse,
    DeleteRequest,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.core.llm_client import AsyncOllamaLLMClient, MockAsyncLLMClient
from app.core.prompts import RAG_USER_PROMPT
from app.core.vectorstore import VectorStore
from app.settings import settings
from app.utils.utils import truncate_long_strings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the RAG system."""
    # TODO: remake this with streaming
    if request.mock_llm:
        llm = MockAsyncLLMClient(log_level="INFO" if not settings.debug else "DEBUG")
    else:
        llm = AsyncOllamaLLMClient(log_level="INFO" if not settings.debug else "DEBUG")

    # Retrieve context
    retrieve_response = None
    if request.use_rag:
        query = request.messages[-1]["content"]
        # TODO: add summary of conversation history to query? core request? param to toggle this?
        # a new system prompt that takes the summarized conversation history
        # and rephrases the RAG query to be better
        retrieve_request = RetrieveRequest(query=query, top_k=request.top_k)
        retrieve_response = await retrieve(retrieve_request)
        retrieved_docs = "\n\n".join([str(doc) for doc in retrieve_response.results])

        # Build LLM payload
        query_with_context = RAG_USER_PROMPT.format(context=retrieved_docs, query=query)
        user_message = {"role": "user", "content": query_with_context}
    else:
        # Build message without context
        user_message = request.messages[-1]

    # Process conversation history
    # TODO: conversation memory scheme
    previous_messages = request.messages[:-1]

    messages = (
        [
            {"role": "system", "content": request.system_prompt}
        ]  # TODO: system prompt at the top? or before latest message?
        + previous_messages
        + [user_message]
    )

    response = await llm.chat(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
    )

    return ChatCompletionWithSources(
        sources=retrieve_response.results if retrieve_response else [],
        **response.model_dump(),
    )


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant documents based on a query."""
    vectorstore = VectorStore(
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model_name,
    )

    results = vectorstore.search(query=request.query, k=request.top_k)

    response = RetrieveResponse(query=request.query, results=results)

    logger.debug(
        f"Retrieve response for '{request.query}': {len(results)} results, top score: {results[0].score if results else 'N/A'}"
    )

    return response


@router.post("/documents/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    request: IngestRequest = Depends(),
) -> IngestResponse:
    """Ingest uploaded documents into the vectorstore."""
    vectorstore = VectorStore(
        log_level="INFO" if not settings.debug else "DEBUG",
        collection_name=request.collection_name,
        embedding_model=request.embedding_model,
    )

    results = await vectorstore.add_files(
        files=files,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    response = IngestResponse(
        chunk_ids=results.chunk_ids,
        errors=results.errors,
    )
    logger.debug(
        f"Ingest response:\n{json.dumps(response.model_dump(), indent=2, default=str)}"
    )
    return response


@router.get("/documents", response_model=CollectionInfoResponse)
async def list_documents():
    """List all documents in the vectorstore."""
    vectorstore = VectorStore(
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model_name,
    )

    collection_info = vectorstore.get_collection_info()

    response = CollectionInfoResponse(
        name=collection_info.name,
        total_chunks=collection_info.total_chunks,
        total_documents=collection_info.total_documents,
        embedding_model=collection_info.embedding_model,
        documents=collection_info.documents,
    )

    logger.debug(
        f"Documents list response:\n{json.dumps(response.model_dump(), indent=2, default=str)}"
    )

    return response


@router.delete("/documents", status_code=status.HTTP_204_NO_CONTENT)
async def delete_documents(request: DeleteRequest):
    """Delete documents or collections."""
    vectorstore = VectorStore(
        collection_name=request.collection_name or settings.collection_name
    )

    try:
        # If delete_collection is True, always delete collection (overrides document_ids)
        if request.delete_collection:
            vectorstore.delete_collection(request.collection_name)
        elif request.document_ids:
            vectorstore.delete_documents(request.document_ids, request.collection_name)

    except Exception as e:
        logger.error(f"Delete operation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Delete operation failed: {str(e)}"
        )


@router.get("/models")
async def list_models():
    """List available models."""
    llm = AsyncOllamaLLMClient(log_level="INFO" if not settings.debug else "DEBUG")
    models = await llm.client.models.list()
    model_names = [model.id for model in models.data]
    models_info = []

    # Check model capabilities
    async with httpx.AsyncClient(timeout=settings.httpx_timeout) as client:
        for model_name in model_names:
            try:
                response = await client.post(
                    f"{settings.ollama_base_url}/api/show", json={"name": model_name}
                )
                response.raise_for_status()
                model_info = response.json()
                # Extract model info
                capabilities = model_info.get("capabilities", [])
                models_info.append(
                    {
                        "name": model_name,
                        "capabilities": capabilities,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get details for model {model_name}: {e}")
                models_info.append(
                    {
                        "name": model_name,
                        "capabilities": ["unknown"],
                    }
                )

    available_models = {
        "models": {model["name"]: model for model in models_info},
        "completion_models": [
            m["name"] for m in models_info if "completion" in m["capabilities"]
        ],
        "embedding_models": [
            m["name"] for m in models_info if "embedding" in m["capabilities"]
        ],
        "total": len(models_info),
    }
    logger.debug(f"Completion models: {available_models['completion_models']}")
    logger.debug(f"Embedding models: {available_models['embedding_models']}")
    return available_models
