import json
import logging
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion

from app.api.schemas import (
    ChatCompletionWithSources,
    ChatRequest,
    CollectionInfoResponse,
    DeleteRequest,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    SyncDirectoriesRequest,
    SyncDirectoriesResponse,
)
from app.core.llm_helpers import (
    async_stream_completion,
    create_ollama_client,
    mock_chat_completion,
    mock_stream_completion,
)
from app.core.llm_helpers import (
    list_models as llm_list_models,
)
from app.core.prompts import RAG_USER_PROMPT
from app.core.vectorstore import VectorStore
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the RAG system."""
    retrieve_response = None
    if request.dev.use_rag:
        query = request.messages[-1]["content"]
        # TODO: add summary of conversation history to query? core request? param to toggle this?
        # a new system prompt that takes the summarized conversation history
        # and rephrases the RAG query to be better
        retrieve_request = RetrieveRequest(
            query=query,
            collection_name=request.collection_name,
            embedding_model=request.embedding_model,
            top_k=request.top_k,
            dev=request.dev,
        )
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
        [{"role": "system", "content": request.system_prompt}]
        + previous_messages
        + [user_message]
    )

    if request.dev.stream:
        return StreamingResponse(
            chat_stream_generator(messages, request, retrieve_response),
            media_type="text/plain",
            headers={"X-Stream-Response": "true"},
        )

    # Regular response
    if request.dev.mock_llm:
        response = mock_chat_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
        )
    else:
        client = create_ollama_client(
            base_url=settings.ollama_base_url, async_client=True
        )
        response = await client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            model=request.model,
            temperature=request.temperature,
            stream=False,
        )

    return ChatCompletionWithSources(
        sources=retrieve_response.results if retrieve_response else [],
        **response.model_dump(),
    )


async def chat_stream_generator(messages, request, retrieve_response):
    """Generate streaming chat responses with SSE-style format."""
    final_response = None

    def _sse_data(payload: str) -> str:
        # SSE requires each line of a multi-line data payload to be prefixed with `data:`.
        # Otherwise newline characters inside payload will be mis-parsed by clients.
        normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
        return f"data: {normalized.replace('\n', '\ndata: ')}\n\n"

    if request.dev.mock_llm:
        stream_iter = mock_stream_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
        )
    else:
        client = create_ollama_client(
            base_url=settings.ollama_base_url, async_client=True
        )
        stream_iter = async_stream_completion(
            client,
            request.model,
            messages,
            temperature=request.temperature,
        )

    async for chunk in stream_iter:
        # Final response object
        if isinstance(chunk, ChatCompletion):
            final_response = chunk.model_dump()
            # Add sources
            if retrieve_response:
                final_response["sources"] = [
                    doc.model_dump() for doc in retrieve_response.results
                ]
        else:
            # Stream content with SSE-style format
            yield _sse_data(chunk)
    # TODO: log the response
    if final_response:
        yield f"event: final\ndata: {json.dumps(final_response)}\n\n"


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant documents based on a query."""
    if request.dev.mock_rag_response:
        results = VectorStore.mock_retrieve(request.query, request.top_k)
        response = RetrieveResponse(query=request.query, results=results)
        logger.debug(
            f"Mock retrieve response for '{request.query}': {len(results)} results."
        )
        return response
    else:
        vectorstore = VectorStore(
            collection_name=request.collection_name,
            embedding_model=request.embedding_model or settings.embedding_model_name,
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


@router.post("/documents/sync", response_model=SyncDirectoriesResponse)
async def sync_directories(request: SyncDirectoriesRequest) -> SyncDirectoriesResponse:
    """Sync one or more local directories into a collection.

    For safety, only directories under settings.documents_path are allowed.
    """

    base_dir = Path(settings.documents_path).resolve()

    if not request.paths:
        raise HTTPException(status_code=400, detail="paths is required")

    directories: list[Path] = []
    for raw_path in request.paths:
        raw_path = (raw_path or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (base_dir / candidate).resolve()
        )

        if not target.exists() or not target.is_dir():
            raise HTTPException(
                status_code=400, detail=f"Directory not found: {raw_path}"
            )

        try:
            if not target.is_relative_to(base_dir):
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory must be under {str(base_dir)}: {raw_path}",
                )
        except AttributeError:
            # Python < 3.9 fallback (shouldn't be hit in this repo)
            if base_dir not in target.parents and target != base_dir:
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory must be under {str(base_dir)}: {raw_path}",
                )

        directories.append(target)

    allowed_exts = {".txt", ".md", ".pdf", ".docx"}
    files: list[Path] = []
    for directory in directories:
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed_exts:
                files.append(p)

    if not files:
        return SyncDirectoriesResponse(total_files=0, chunk_ids={}, errors={})

    vectorstore = VectorStore(
        log_level="INFO" if not settings.debug else "DEBUG",
        collection_name=request.collection_name,
        embedding_model=request.embedding_model,
    )

    results = await vectorstore.upsert_files(
        files=files,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    return SyncDirectoriesResponse(
        total_files=len(files),
        chunk_ids=results.chunk_ids,
        errors=results.errors,
    )


@router.get("/documents", response_model=CollectionInfoResponse)
async def list_documents(
    collection_name: str = settings.collection_name,
    embedding_model: Optional[str] = None,
):
    """List all documents in the vectorstore."""

    vectorstore = VectorStore(
        collection_name=collection_name,
        embedding_model=embedding_model or settings.embedding_model_name,
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
        f"Documents list response for collection '{collection_name}':\n{json.dumps(response.model_dump(), indent=2, default=str)}"
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
    client = create_ollama_client(base_url=settings.ollama_base_url, async_client=True)
    return await llm_list_models(
        client,
        base_url=settings.ollama_base_url,
        include_capabilities=True,
    )
