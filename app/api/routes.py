import json
import logging

import httpx
from fastapi import APIRouter, status

from app.api.schemas import (
    ChatCompletionWithSources,
    ChatRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from app.core.llm_client import AsyncOllamaLLMClient, MockAsyncLLMClient
from app.core.prompts import RAG_USER_PROMPT
from app.settings import settings

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
        llm = MockAsyncLLMClient(log_level="INFO")
    else:
        llm = AsyncLLMClient(log_level="INFO")

    # Retrieve context
    retrieve_response = None
    if request.use_rag:
        query = request.messages[-1]["content"]
        retrieve_request = RetrieveRequest(query=query, top_k=request.top_k)
        retrieve_response = await retrieve(retrieve_request)
        retrieved_docs = "\n\n".join([str(doc) for doc in retrieve_response.results])

        # Build LLM payload
        query_with_context = RAG_USER_PROMPT.format(
            context=retrieved_docs, question=query
        )
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

    logger.debug(
        f"Chat response:\n{json.dumps(response.model_dump(), indent=2, default=str)}"
    )

    return ChatCompletionWithSources(
        sources=retrieve_response.results if retrieve_response else [],
        **response.model_dump(),
    )


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant documents based on a query."""
    # TODO:
    # - embed query
    # - search vectorstore using search strategy, top_k
    # - rerank?
    return RetrieveResponse(query="", results=[])


@router.post("/documents/ingest")
def ingest_documents():
    """Ingest documents into the vectorstore."""
    # TODO:
    # - save files, extract text - from upload or directory
    # - embed and store in vectorstore
    # NOTE:
    # - this endpoint should get hit when /chat does - but if the docs already exist, it doesnt reingest
    # - or just auto ingest and make an endpoint that gives us the status?
    # - OR: strictly only ingest docs via this endpoint, track which docs are uploaded on frontend, /chat only uses whats in the vectorstore
    pass


@router.get("/documents")
async def list_documents():
    """List all documents in the vectorstore."""
    pass


@router.get("/models")
async def list_models():
    """List available models."""
    llm = AsyncLLMClient()
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
