import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.llm_client import AsyncLLMClient
from app.core.prompts import RAG_USER_PROMPT
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Request models
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of chat messages")
    system_prompt: str = Field(..., description="System prompt for the chat")
    model: str = Field(..., description="LLM model to use")
    temperature: float = Field(
        ..., ge=0.0, le=2.0, description="Temperature for response generation"
    )
    top_k: int = Field(..., ge=1, description="Number of documents to retrieve")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")


class DocumentChunk(BaseModel):
    content: str = Field(..., description="The text content of the document chunk")
    score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    document_id: str = Field(..., description="ID of the source document")
    document_title: str = Field(
        ..., description="Title/filename of the source document"
    )
    chunk_index: int = Field(..., description="Index of this chunk within the document")

    def __str__(self) -> str:
        return f"""Source: {self.document_title}\nScore: {self.score:.2f})\n{self.content}"""


class ChatResponse(BaseModel):
    response: str = Field(..., description="The LLM's response")
    sources: List[DocumentChunk] = Field(
        default=[], description="Retrieved document sources"
    )
    model_used: str = Field(..., description="The LLM model that was used")
    temperature_used: float = Field(..., description="The temperature setting used")


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="The original search query")
    results: List[DocumentChunk] = Field(..., description="Retrieved document chunks")


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the RAG system."""
    # TODO: remake this with streaming
    llm = AsyncLLMClient()

    # Retrieve context
    query = request.messages[-1]["content"]
    retrieve_request = RetrieveRequest(query=query, top_k=request.top_k)
    retrieve_response = await retrieve(retrieve_request)
    retrieved_docs = "\n\n".join([str(doc) for doc in retrieve_response.results])

    # Process conversation history
    # TODO: conversation memory scheme
    previous_messages = request.messages[:-1]

    # Build LLM payload
    query_with_context = RAG_USER_PROMPT.format(context=retrieved_docs, question=query)
    user_message = {"role": "user", "content": query_with_context}
    messages = (
        [{"role": "system", "content": request.system_prompt}]
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
            response=response["choices"][0]["message"]["content"],
            sources=retrieve_response.results,
            model_used=request.model,
            temperature_used=request.temperature,
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
    pass


@router.get("/documents")
async def list_documents():
    """List all documents in the vectorstore."""
    pass
