import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import chromadb
from attr import dataclass
from chromadb.utils import embedding_functions
from starlette.datastructures import UploadFile

from app.core.ingestor import Document, RetrievedDocumentChunk
from app.settings import settings
from app.utils.callbacks import CallbackMeta, with_callbacks

logger = logging.getLogger(__name__)


class CollectionInfo:
    """Domain model for collection information."""

    def __init__(
        self,
        name: str,
        embedding_model: str,
        documents: List[Document],
        total_chunks: int,
    ):
        self.name = name
        self.embedding_model = embedding_model
        self.documents = documents
        self.total_chunks = total_chunks

    @property
    def total_documents(self) -> int:
        """Computed property: total number of documents."""
        return len(self.documents)

    @property
    def total_file_size(self) -> int:
        """Computed property: total file size across all documents."""
        return sum(doc.metadata.get("file_size", 0) or 0 for doc in self.documents)

    @property
    def file_types(self) -> Dict[str, int]:
        """Computed property: count of files by extension."""
        file_types = {}
        for doc in self.documents:
            ext = doc.metadata.get("file_extension", "unknown") or "unknown"
            file_types[ext] = file_types.get(ext, 0) + 1
        return file_types

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"Collection '{self.name}': {self.total_documents} documents, "
            f"{self.total_chunks} chunks, embedding model: {self.embedding_model}"
        )

    @classmethod
    def from_documents(
        cls,
        name: str,
        embedding_model: str,
        documents: List[Document],
        total_chunks: int,
    ) -> "CollectionInfo":
        """Create CollectionInfo from a list of documents and total chunk count."""
        return cls(
            name=name,
            embedding_model=embedding_model,
            documents=documents,
            total_chunks=total_chunks,
        )


@dataclass
class AddFilesResult:
    """Result of adding files to the vector store."""

    chunk_ids: Dict[str, List[str]]  # filename -> chunk_ids
    errors: Dict[str, str]  # filename -> error_message

    @property
    def total_chunks(self) -> int:
        """Total number of chunks added."""
        return sum(len(ids) for ids in self.chunk_ids.values())


class VectorStore(metaclass=CallbackMeta):
    """ChromaDB vector store client."""

    def __init__(
        self,
        collection_name: str = settings.collection_name,
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
        # TODO: is there an async embedding function?
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="ollama",
            api_base=settings.ollama_url,
            model_name=embedding_model,
        )

        # Get or create collection
        # TODO: error handling?
        # TODO: need to self check if docs are duplicated
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
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

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.chroma_metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,  # type: ignore
                ids=ids,
            )
        except Exception as e:
            logger.warning(
                f"Failed to add document '{document.title}' to collection '{self.collection_name}': {e}"
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
    async def add_file(
        self,
        file: Union[str, Path, UploadFile],
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> List[str]:
        """Add a file (path or upload)."""
        if isinstance(file, UploadFile):
            document = await Document.from_upload(file)
        else:
            document = Document.from_file(file)

        return self.add_document(
            document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    async def add_files(
        self,
        files: Sequence[Union[str, Path, UploadFile]],
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> AddFilesResult:
        """Add multiple files."""
        chunk_ids = {}
        errors = {}
        for file in files:
            # Get filename
            if isinstance(file, UploadFile):
                filename = file.filename or "unknown"
            else:
                filename = str(file)
            try:
                ids = await self.add_file(
                    file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                chunk_ids[filename] = ids
            except Exception as e:
                errors[filename] = str(e)
        return AddFilesResult(
            chunk_ids=chunk_ids,
            errors=errors,
        )

    @with_callbacks
    def search(self, query: str, k: int = 5) -> List[RetrievedDocumentChunk]:
        """Search for similar document chunks."""
        try:
            results = self.collection.query(query_texts=[query], n_results=k)
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}") from e

        if not results["documents"]:
            return []

        # NOTE: Type checker isn't satisfied but our schemas ensure safety
        assert results["metadatas"] is not None and results["distances"] is not None
        assert (
            results["metadatas"][0] is not None and results["distances"][0] is not None
        )

        return [
            RetrievedDocumentChunk(
                id=str(chunk_id),
                content=str(doc),
                document_id=str(meta["document_id"]),
                document_title=str(meta["document_title"]),
                chunk_index=int(meta["chunk_index"] or 0),
                metadata=dict(meta),
                score=1.0 - float(distance),
            )
            for doc, meta, chunk_id, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0],
                results["distances"][0],
            )
        ]

    @with_callbacks
    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> CollectionInfo:
        """Get collection info."""
        if collection_name and collection_name != self.collection_name:
            # Create temporary instance for different collection
            temp_store = VectorStore(
                collection_name=collection_name,
                embedding_model=self.embedding_model,
            )
            return temp_store.get_collection_info()

        count = self.collection.count()

        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        # Reconstruct Document objects
        documents_by_id = {}
        for meta in metadatas:
            if meta and "document_id" in meta:
                doc_id = str(meta["document_id"])
                if doc_id not in documents_by_id:
                    # Create Document from first chunk's metadata
                    documents_by_id[doc_id] = Document.from_chunk_metadata(
                        doc_id, dict(meta)
                    )

        document_objects = list(documents_by_id.values())

        return CollectionInfo.from_documents(
            name=self.collection_name,
            embedding_model=self.embedding_model,
            documents=document_objects,
            total_chunks=count,
        )

    @with_callbacks
    def delete_documents(
        self, document_ids: List[str], collection_name: Optional[str] = None
    ) -> int:
        """Delete documents by document IDs, returns number of chunks deleted."""
        if collection_name and collection_name != self.collection_name:
            # Create temporary instance for different collection
            temp_store = VectorStore(
                collection_name=collection_name,
                embedding_model=self.embedding_model,
            )
            return temp_store.delete_documents(document_ids)

        # Find all chunks for these documents
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
        ids = results.get("ids", []) or []

        chunk_ids_to_delete = []
        for chunk_id, meta in zip(ids, metadatas):
            if meta and meta.get("document_id") in document_ids:
                chunk_ids_to_delete.append(chunk_id)

        if chunk_ids_to_delete:
            self.collection.delete(ids=chunk_ids_to_delete)

        return len(chunk_ids_to_delete)

    @with_callbacks
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete the entire collection."""
        self.client.delete_collection(name=collection_name or self.collection_name)
        return True

    # Callback methods
    def _post_add_document(
        self, result: List[str], duration: float, document: Document, *args, **kwargs
    ):
        logger.debug(
            f"Added document '{document.title}' as {len(result)} chunks to collection '{self.collection_name}' in {duration:.4f}s"
        )
        return result

    def _pre_search(self, query: str, k: int, *args, **kwargs):
        logger.debug(
            f"Searching collection '{self.collection_name}' for: '{query[:50]}...' (k={k})"
        )

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
        logger.debug(
            f"Search results: {[f'Chunk {chunk.chunk_index} from {chunk.document_title} (score {chunk.score:.2f})' for chunk in result[:5]]} ..."
        )
        return result

    def _post_delete_collection(
        self,
        result: bool,
        duration: float,
        collection_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        target_collection = collection_name or self.collection_name
        logger.info(f"Deleted collection '{target_collection}' in {duration:.4f}s")
        return result

    def _post_get_collection_info(
        self,
        result: CollectionInfo,
        duration: float,
        collection_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        logger.debug(f"Retrieved collection info:\n{result}")
        return result

    def _post_delete_documents(
        self,
        result: int,
        duration: float,
        document_ids: List[str],
        collection_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        target_collection = collection_name or self.collection_name
        logger.info(
            f"Deleted {len(document_ids)} documents ({result} chunks) from collection '{target_collection}' in {duration:.4f}s"
        )
        return result
