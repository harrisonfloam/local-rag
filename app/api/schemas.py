# Request models
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.ingestor import DocumentChunk


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


class ChatResponse(BaseModel):
    response: Optional[str] = Field(..., description="The LLM's response")
    sources: List[DocumentChunk] = Field(
        default=[], description="Retrieved document sources"
    )
    model: str = Field(..., description="The LLM model that was used")
    temperature: float = Field(..., description="The temperature setting used")


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="The original search query")
    results: List[DocumentChunk] = Field(..., description="Retrieved document chunks")
