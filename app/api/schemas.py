# Request models
from typing import Dict, List

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from app.core.ingestor import Document, RetrievedDocumentChunk
from app.core.prompts import RAG_SYSTEM_PROMPT
from app.settings import settings


# Chat endpoint
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
    embedding_model: str = Field(
        default=settings.embedding_model_name,
        description="Embedding model to use for RAG",
    )
    top_k: int = Field(
        default=settings.top_k, ge=1, description="Number of documents to retrieve"
    )
    # TODO: put dev settings in a subclass
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG context",
    )
    mock_llm: bool = Field(
        default=settings.mock_llm,
        description="Use a mock LLM for testing purposes",
    )


class ChatCompletionWithSources(ChatCompletion):
    """Chat response model that includes document sources."""

    sources: List[RetrievedDocumentChunk] = Field(
        default=[], description="Retrieved document sources"
    )


# Retrieve endpoint
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(
        default=settings.top_k, ge=1, le=50, description="Number of results to return"
    )


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="The original search query")
    results: List[RetrievedDocumentChunk] = Field(
        ..., description="Retrieved document chunks"
    )


# Document ingestion endpoint
class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    collection_name: str = Field(
        default=settings.collection_name,
        description="Name of the vectorstore collection",
    )
    embedding_model: str = Field(
        default=settings.embedding_model_name,
        description="Embedding model",
    )
    chunk_size: int = Field(
        default=settings.chunk_size, ge=100, le=8000, description="Size of text chunks"
    )
    chunk_overlap: int = Field(
        default=settings.chunk_overlap,
        ge=0,
        le=1000,
        description="Overlap between chunks",
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    chunk_ids: Dict[str, List[str]] = Field(
        default_factory=dict, description="Chunk IDs by filename"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict, description="Error messages by filename"
    )

    @property
    def total_files(self) -> int:
        """Total number of files processed."""
        return len(self.chunk_ids) + len(self.errors)

    @property
    def total_chunks(self) -> int:
        """Total number of chunks created."""
        return sum(len(chunk_ids) for chunk_ids in self.chunk_ids.values())

    @property
    def total_errors(self) -> int:
        """Total number of errors encountered."""
        return len(self.errors)

    @property
    def successful_files(self) -> int:
        """Number of successfully processed files."""
        return len(self.chunk_ids)

    @property
    def status(self) -> str:
        """Overall status of the ingestion."""
        if self.total_errors > 0:
            return "partial"
        elif self.total_files == 0:
            return "failed"
        else:
            return "completed"


# Document listing endpoint
class CollectionInfoResponse(BaseModel):
    name: str
    total_chunks: int
    total_documents: int
    embedding_model: str
    documents: List[Document]
