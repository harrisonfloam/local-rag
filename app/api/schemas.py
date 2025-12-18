# Request models
from typing import Dict, List, Optional

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, model_validator

from app.core.ingestor import Document, RetrievedDocumentChunk
from app.core.prompts import RAG_SYSTEM_PROMPT
from app.settings import settings


# Chat endpoint
class DevSettings(BaseModel):
    use_rag: bool = True
    mock_llm: bool = settings.mock_llm
    stream: bool = settings.stream
    mock_rag_response: bool = settings.mock_rag_response


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    system_prompt: str = RAG_SYSTEM_PROMPT
    model: str
    temperature: float = Field(default=settings.temperature, ge=0.0, le=2.0)
    top_k: int = Field(default=settings.top_k, ge=1)
    dev: DevSettings = Field(default_factory=DevSettings)


class ChatCompletionWithSources(ChatCompletion):
    """Chat response with document sources."""

    sources: List[RetrievedDocumentChunk] = []


# Retrieve endpoint
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    dev: DevSettings = Field(default_factory=DevSettings)


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrievedDocumentChunk]


# Document ingestion endpoint
class IngestRequest(BaseModel):
    collection_name: str = settings.collection_name
    embedding_model: str = settings.embedding_model_name
    chunk_size: int = Field(default=settings.chunk_size, ge=100, le=8000)
    chunk_overlap: int = Field(default=settings.chunk_overlap, ge=0, le=1000)


class IngestResponse(BaseModel):
    chunk_ids: Dict[str, List[str]] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)

    @property
    def total_files(self) -> int:
        return len(self.chunk_ids) + len(self.errors)

    @property
    def total_chunks(self) -> int:
        return sum(len(chunk_ids) for chunk_ids in self.chunk_ids.values())

    @property
    def total_errors(self) -> int:
        return len(self.errors)

    @property
    def successful_files(self) -> int:
        """Number of successfully processed files."""
        return len(self.chunk_ids)

    @property
    def status(self) -> str:
        if self.total_errors > 0:
            return "partial"
        elif self.total_files == 0:
            return "failed"
        else:
            return "completed"


# Document deletion endpoint
class DeleteRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    collection_name: Optional[str] = None
    delete_collection: bool = False

    @model_validator(mode="after")
    def validate_deletion_request(self):
        if not self.document_ids and not self.delete_collection:
            raise ValueError("Must specify document_ids or delete_collection")
        return self


# Document listing endpoint
class CollectionInfoResponse(BaseModel):
    name: str
    total_chunks: int
    total_documents: int
    embedding_model: str
    documents: List[Document]
