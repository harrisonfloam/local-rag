import logging

from fastapi import APIRouter, status

from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@router.post("/chat")
async def chat():
    """Chat with the RAG system."""
    # TODO:
    # - hit /retrieve to get documents
    # - send context, message to LLM
    # - return response with sources
    pass


@router.post("/retrieve")
async def retrieve():
    """Retrieve relevant documents based on a query."""
    # TODO:
    # - embed query
    # - search vectorstore using search strategy, top_k
    # - rerank?
    pass


@router.post("/documents/upload")
def upload_documents():
    """Upload and ingest documents into the vectorstore."""
    # TODO:
    # - save files, extract text - from upload or directory
    # - embed and store in vectorestore
    pass


@router.get("/documents")
async def list_documents():
    """List all documents in the vectorstore."""
    pass
