import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions

from app.core.ingestor import Document, DocumentChunk, RetrievedDocumentChunk
from app.settings import settings
from app.utils.callbacks import CallbackMeta, with_callbacks

logger = logging.getLogger(__name__)


class VectorStore(metaclass=CallbackMeta):
    """ChromaDB vector store client."""

    def __init__(
        self,
        embedding_model: str = settings.embedding_model_name,
        log_level: str = settings.log_level,
    ):
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized.")

        # Connect to ChromaDB service
        self.client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )

        # Create embedding function for Ollama
        self.embedding_model = embedding_model
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="ollama",
            api_base=settings.ollama_url,
            model_name=embedding_model,
        )

        # Get or create collection
        # TODO: error handling?
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,  # type: ignore
        )

    @with_callbacks
    def add_document(
        self,
        document: Document,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> List[str]:
        """Add a document."""
        chunks = document.to_chunks(chunk_size=chunk_size, overlap=chunk_overlap)

        texts, metadatas, ids = [], [], []
        for chunk in chunks:
            texts.append(chunk.content)
            metadatas.append(chunk.chroma_metadata)
            ids.append(chunk.id)

        # Add to vector store
        self.collection.add(
            documents=texts,
            metadatas=metadatas,  # type: ignore
            ids=ids,
        )

        # TODO: capture error types here?
        return ids

    @with_callbacks
    def add_text(
        self,
        text: str,
        title: str,
        source: str,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> List[str]:
        """Add raw text."""
        document = Document.from_text(text, title=title, source=source)
        return self.add_document(
            document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    @with_callbacks
    def add_file(
        self,
        file_path: Union[str, Path],
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> List[str]:
        """Add a file."""
        document = Document.from_file(file_path)
        return self.add_document(
            document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def add_files(
        self,
        file_paths: List[Union[str, Path]],
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> Dict[str, List[str]]:
        """Add multiple files."""
        results = {}
        for file_path in file_paths:
            try:
                ids = self.add_file(
                    file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                results[str(file_path)] = ids
            except Exception as e:
                logger.error(f"Failed to add {file_path}: {e}")
                results[str(file_path)] = []
        return results

    @with_callbacks
    def search(self, query: str, k: int = 5) -> List[RetrievedDocumentChunk]:
        """Search for similar document chunks."""
        try:
            results = self.collection.query(query_texts=[query], n_results=k)
        except Exception as e:
            if "too long" in str(e).lower():
                raise ValueError(
                    f"Query too long for embedding model: {query[:100]}..."
                ) from e
            raise

        # Format results as RetrievedDocumentChunk objects
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                metadata = results["metadatas"][0][i] or {}

                chunk = RetrievedDocumentChunk(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    document_id=metadata.get("document_id", "unknown"),
                    document_title=metadata.get("document_title", "Unknown"),
                    chunk_index=metadata.get("chunk_index", 0),
                    metadata=metadata,
                    score=1
                    - results["distances"][0][i],  # Convert distance to similarity
                )
                chunks.append(chunk)

        return chunks

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        # TODO: get all the ids?
        return {
            "name": "documents",
            "count": count,
            "embedding_function": self.embedding_model,
        }

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        self.collection.delete(ids=ids)
        return True

    # Callback methods
    def _pre_add_document(self, document: Document, *args, **kwargs):
        logger.debug(f"Adding document: {document.title}")

    def _post_add_document(
        self, result: List[str], duration: float, document: Document, *args, **kwargs
    ):
        logger.debug(
            f"Added document '{document.title}' as {len(result)} chunks in {duration:.4f}s"
        )
        return result

    def _pre_add_text(self, text: str, title: str, *args, **kwargs):
        logger.debug(f"Adding text: {title}")

    def _post_add_text(
        self, result: List[str], duration: float, text: str, title: str, *args, **kwargs
    ):
        logger.debug(f"Added text '{title}' as {len(result)} chunks in {duration:.4f}s")
        return result

    def _pre_add_file(self, file_path: Union[str, Path], *args, **kwargs):
        logger.debug(f"Adding 1 file: {file_path}")

    def _post_add_file(
        self,
        result: List[str],
        duration: float,
        file_path: Union[str, Path],
        *args,
        **kwargs,
    ):
        logger.debug(
            f"Added file '{file_path}' as {len(result)} chunks in {duration:.4f}s"
        )
        return result

    def _pre_add_files(self, file_paths: List[Union[str, Path]], *args, **kwargs):
        logger.debug(f"Adding {len(file_paths)} files: {file_paths[:3]}...")

    def _post_add_files(
        self,
        result: List[str],
        duration: float,
        file_paths: List[Union[str, Path]],
        *args,
        **kwargs,
    ):
        logger.debug(
            f"Added {len(file_paths)} files as {len(result)} chunks in {duration:.4f}s"
        )
        return result

    def _pre_search(self, query: str, k: int, *args, **kwargs):
        logger.debug(f"Searching for: '{query[:50]}...' (k={k})")

    def _post_search(
        self,
        result: List[RetrievedDocumentChunk],
        duration: float,
        query: str,
        k: int,
        *args,
        **kwargs,
    ):
        logger.info(f"Found {len(result)} results in {duration:.4f}s")
        return result
