# Request models
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.ingestor import RetrievedDocumentChunk
from app.core.prompts import RAG_SYSTEM_PROMPT
from app.settings import settings


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of chat messages")
    system_prompt: str = Field(
        default=RAG_SYSTEM_PROMPT, description="System prompt for the chat"
    )
    model: str = Field(..., description="LLM model to use")
    temperature: float = Field(
        default=settings.temperature,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    top_k: int = Field(
        default=settings.top_k, ge=1, description="Number of documents to retrieve"
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG context",
    )
    mock_llm: bool = Field(
        default=settings.mock_llm,
        description="Use a mock LLM for testing purposes",
    )


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(
        default=settings.top_k, ge=1, le=50, description="Number of results to return"
    )


class ChatResponse(BaseModel):
    response: Optional[str] = Field(..., description="The LLM's response")
    sources: List[RetrievedDocumentChunk] = Field(
        default=[], description="Retrieved document sources"
    )
    model: str = Field(..., description="The LLM model that was used")
    finish_reason: Optional[str] = Field(None, description="Response finish reason")
    usage: Optional[Dict[str, Optional[int]]] = Field(
        None, description="Usage statistics if available"
    )


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="The original search query")
    results: List[RetrievedDocumentChunk] = Field(
        ..., description="Retrieved document chunks"
    )
