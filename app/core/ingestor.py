import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

from markitdown import MarkItDown
from pydantic import BaseModel, Field
from semantic_text_splitter import TextSplitter
from starlette.datastructures import UploadFile

from app.settings import settings


class Document(BaseModel):
    """Represents a source document before chunking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Document title or filename")
    content: str = Field(..., description="Full document content")
    source: str = Field(..., description="Source path or identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @classmethod
    def from_text(
        cls, content: str, title: str = "Untitled", source: str = "text", **metadata
    ) -> "Document":
        """Create document from text content."""
        return cls(title=title, content=content, source=source, metadata=metadata)

    @classmethod
    def from_file(cls, file: Union[str, Path], **metadata) -> "Document":
        """Create document from file path."""
        path = Path(file)

        suffix = path.suffix.lower()

        if suffix in [".txt", ".md"]:
            content = path.read_text(encoding="utf-8")
        elif suffix in [".pdf", ".docx"]:
            md = MarkItDown()
            result = md.convert(str(path))
            content = result.text_content
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return cls(
            title=path.name,
            content=content,
            source=str(path),
            metadata={
                "file_size": path.stat().st_size,
                "file_extension": path.suffix,
                **metadata,
            },
        )

    @classmethod
    def from_chunk_metadata(
        cls, document_id: str, metadata: Dict[str, Any]
    ) -> "Document":
        """Reconstruct Document from chunk metadata."""
        # Filter out chunk-specific and chunking-process metadata
        document_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "chunk_id",
                "document_id",
                "document_title",
                "chunk_index",
                "source",
            }
        }

        return cls(
            id=document_id,
            title=metadata.get("document_title", "Unknown"),
            content="",  # Content not available from metadata
            source=metadata.get("source", "unknown"),
            metadata=document_metadata,
        )

    @classmethod
    async def from_upload(cls, file: UploadFile, **metadata) -> "Document":
        """Create document from FastAPI UploadFile"""
        content_bytes = await file.read()

        filename = file.filename or ""
        suffix = Path(filename).suffix.lower() if filename else ""

        # Handle different file types based on filename or content type
        if file.content_type == "text/plain" or suffix in {".txt", ".md"}:
            content = content_bytes.decode("utf-8")
        elif suffix in {".pdf", ".docx"}:
            md = MarkItDown()
            result = md.convert_stream(BytesIO(content_bytes), file_extension=suffix)
            content = result.text_content
        else:
            raise ValueError(f"Unsupported file type: {file.content_type or 'unknown'}")

        return cls(
            title=file.filename or "Untitled",
            content=content,
            source=f"upload/{file.filename}",
            metadata={
                "file_size": len(content_bytes),
                "file_extension": Path(file.filename).suffix if file.filename else "",
                "content_type": file.content_type,
                **metadata,
            },
        )

    def to_chunks(
        self,
        chunk_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
    ) -> List["DocumentChunk"]:
        """Convert document to chunks using semantic-text-splitter."""
        # Record the actual chunking params used in the document metadata
        self.metadata["chunk_size"] = chunk_size
        self.metadata["chunk_overlap"] = overlap

        splitter = TextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", capacity=chunk_size, overlap=overlap
        )

        chunks_text = splitter.chunks(self.content)

        chunk_metadata = {
            **self.metadata,  # Original document metadata
            "source": self.source,
            "total_chunks": len(chunks_text),
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
        }

        return [
            DocumentChunk(
                content=chunk_text,
                document_id=self.id,
                document_title=self.title,
                chunk_index=i,
                metadata=chunk_metadata,
            )
            for i, chunk_text in enumerate(chunks_text)
        ]


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The text content of the document chunk")
    document_id: str = Field(..., description="ID of the source document")
    document_title: str = Field(
        ..., description="Title/filename of the source document"
    )
    chunk_index: int = Field(..., description="Index of this chunk within the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    @property
    def chroma_metadata(self) -> Dict[str, Any]:
        """Get metadata for ChromaDB storage."""
        return {
            **self.metadata,  # Original document metadata + chunking params
            "chunk_id": self.id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunk_index": self.chunk_index,
        }

    def __str__(self) -> str:
        return f"Chunk {self.chunk_index} from {self.document_title}:\n{self.content[:100]}..."


class RetrievedDocumentChunk(DocumentChunk):
    """DocumentChunk with retrieval score."""

    score: float = Field(..., description="Relevance score (0.0 to 1.0)")

    def __str__(self) -> str:
        return (
            f"Source: {self.document_title} (Score: {self.score:.2f})\n{self.content}"
        )
