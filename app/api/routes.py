import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.schemas import ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse
from app.core.llm_client import AsyncLLMClient, MockAsyncLLMClient
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
        llm = MockAsyncLLMClient()
    else:
        llm = AsyncLLMClient()

    # Retrieve context
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

    try:
        response = await llm.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
        )
        return ChatResponse(
            response=response.choices[0].message.content,
            sources=retrieve_response.results if request.use_rag else [],
            model=request.model,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat")


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
