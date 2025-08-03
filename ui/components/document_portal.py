from contextlib import contextmanager
from typing import Dict, List, Optional

import httpx
import streamlit as st

from app.api.schemas import DeleteRequest, IngestRequest
from app.settings import settings

# Document portal functions converted for use in dedicated page
# Used by ui/pages/1_Documents.py


@contextmanager
def http_client():
    """Context manager for HTTP client with standard timeout."""
    with httpx.Client(timeout=settings.httpx_timeout) as client:
        yield client


def clear_collection_cache():
    """Clear collection cache to force refresh."""
    if "current_collection_info" in st.session_state:
        del st.session_state["current_collection_info"]


def handle_api_error(action: str, error: Exception):
    """Standard error handling for API calls."""
    st.error(f"❌ Error {action}: {error}")


def render_document_browser():
    """Terminal-style document browser showing collection contents - for page use."""
    st.subheader("Document Browser & Manager")

    # Collection selector
    col1, col2 = st.columns([3, 1])
    with col1:
        collection_name = st.text_input(
            "Collection Name",
            value=settings.collection_name,
            key="browser_collection_name",
            help="Enter collection name to browse",
        )
    with col2:
        if st.button("📊 Load Collection", key="fetch_collection_info"):
            with st.spinner(f"Loading collection '{collection_name}'..."):
                collection_info = _fetch_collection_info(collection_name)
                st.session_state.current_collection_info = collection_info
                st.session_state.current_collection_name = collection_name

    # Get current collection info
    collection_info = st.session_state.get("current_collection_info")
    current_collection_name = st.session_state.get(
        "current_collection_name", collection_name
    )

    if collection_info is None:
        st.info("👆 Click 'Load Collection' to browse documents")
        return

    if "error" in collection_info:
        st.error(f"Error loading collection: {collection_info['error']}")
        return

    # Terminal-style header
    _render_terminal_header(collection_info)

    # Document listing with integrated checkboxes
    _render_document_list_with_checkboxes(collection_info, current_collection_name)


def _render_terminal_header(collection_info: Dict):
    """Render terminal-style collection summary header."""
    total_docs = collection_info.get("total_documents", 0)
    total_chunks = collection_info.get("total_chunks", 0)
    embedding_model = collection_info.get("embedding_model", "unknown")

    # Terminal-style info block
    st.code(
        f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Collection: {collection_info.get("name", "unknown"):<30} Documents: {total_docs:<6}   ┃
┃ Embedding:  {embedding_model:<30} Chunks:    {total_chunks:<6}    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """.strip(),
        language="text",
    )


def _render_document_list_with_checkboxes(collection_info: Dict, collection_name: str):
    """Render document list with integrated checkboxes in a single code block."""
    documents = collection_info.get("documents", [])

    if not documents:
        st.info("📄 No documents in this collection")
        return

    # Selection controls
    col1, col2 = st.columns([1, 3])
    with col1:
        select_all = st.checkbox("Select All", key="select_all_docs")
    with col2:
        delete_selected = st.button(
            "🗑️ Delete Selected/Collection", key="delete_docs", type="primary"
        )

    # Build the document listing as a single code block with checkboxes
    selected_docs = []

    # Header
    listing_text = "☐  NAME                           SIZE     CHUNKS   STATUS     ID\n"
    listing_text += "─" * 70 + "\n"

    # Document rows
    for i, doc in enumerate(documents):
        doc_id = doc.get("id", f"doc_{i}")
        doc_name = doc.get("title", "unknown")[:25].ljust(25)
        doc_size = f"{doc.get('metadata', {}).get('file_size', 0):,}B".rjust(8)
        doc_chunks = str(doc.get("metadata", {}).get("total_chunks", 0)).rjust(6)
        doc_status = "✅ embedded".ljust(10)

        # Hidden checkbox for each document
        is_selected = st.checkbox(
            f"doc_{doc_id}",
            key=f"select_doc_{doc_id}",
            value=select_all,
            label_visibility="collapsed",
        )

        if is_selected:
            selected_docs.append(doc_id)

        # Add row to listing with checkbox indicator
        checkbox_indicator = "☑" if is_selected else "☐"
        listing_text += f"{checkbox_indicator}  {doc_name} {doc_size} {doc_chunks} {doc_status} {doc_id[:8]}...\n"

    # Display the complete listing
    st.code(listing_text, language="text")

    # Handle deletion
    if delete_selected:
        if select_all:
            # Delete entire collection
            _handle_collection_deletion(collection_name)
        elif selected_docs:
            # Delete selected documents
            _handle_document_deletion(collection_name, selected_docs)
        else:
            st.warning("No documents selected for deletion")

    # Store selected documents in session state
    st.session_state.selected_documents = selected_docs


def render_upload_interface():
    """Enhanced upload interface - for page use."""
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Select files to upload",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="portal_file_uploader",
    )

    col1, col2 = st.columns(2)
    with col1:
        collection_name = st.text_input(
            "Collection Name",
            key="upload_collection_name",
            value=settings.collection_name,
        )
    with col2:
        embedding_model = st.selectbox(
            "Embedding Model",
            options=st.session_state.get(
                "embedding_models", [settings.embedding_model_name]
            ),
            key="upload_embedding_model",
            index=0,
        )

    col3, col4 = st.columns(2)
    with col3:
        chunk_size = st.number_input(
            "Chunk Size",
            key="upload_chunk_size",
            min_value=50,
            max_value=2048,
            value=settings.chunk_size,
            step=50,
        )
    with col4:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            key="upload_chunk_overlap",
            min_value=0,
            max_value=512,
            value=settings.chunk_overlap,
            step=10,
        )

    if st.button(
        "🚀 Embed Documents", disabled=not uploaded_files, key="embed_documents"
    ):
        _handle_document_upload(
            uploaded_files, collection_name, embedding_model, chunk_size, chunk_overlap
        )


# Collection manager functionality is now integrated into the browse tab


def _fetch_collection_info(collection_name: Optional[str] = None) -> Optional[Dict]:
    """Fetch collection information from the API."""
    try:
        with httpx.Client(timeout=settings.httpx_timeout) as client:
            # Use query parameter for collection name
            params = {"collection_name": collection_name} if collection_name else {}
            response = client.get(f"{settings.api_url}/documents", params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}


def _handle_document_upload(
    files,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Handle document upload and embedding."""
    with st.spinner(f"Embedding {len(files)} files..."):
        try:
            ingest_request = IngestRequest(
                collection_name=collection_name,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            with httpx.Client(timeout=settings.httpx_timeout) as client:
                files_data = [
                    ("files", (file.name, file.getvalue(), file.type)) for file in files
                ]

                response = client.post(
                    f"{settings.api_url}/documents/ingest",
                    files=files_data,
                    data=ingest_request.model_dump(),
                )
                response.raise_for_status()
                response_data = response.json()

            # Show results
            if response_data["status"] == "success":
                st.success(
                    f"✅ Embedded {response_data['total_files']} files as {response_data['total_chunks']} chunks."
                )
            elif response_data["status"] == "partial":
                st.warning(
                    f"⚠️ Embedded {response_data['successful_files']} files with {response_data['total_chunks']} chunks. {response_data['total_errors']} errors."
                )

            # Clear the collection cache to force refresh
            if "current_collection_info" in st.session_state:
                del st.session_state["current_collection_info"]

        except Exception as e:
            st.error(f"❌ Error during ingestion: {e}")


def _handle_document_deletion(
    collection_name: str, selected_docs: Optional[List[str]] = None
):
    """Handle deletion of selected documents."""
    if not selected_docs:
        selected_docs = st.session_state.get("selected_documents", [])

    if not selected_docs:
        st.warning("No documents selected for deletion")
        return

    try:
        delete_request = DeleteRequest(
            document_ids=selected_docs,
            collection_name=collection_name,
            delete_collection=False,
        )

        with httpx.Client(timeout=settings.httpx_timeout) as client:
            response = client.request(
                "DELETE",
                f"{settings.api_url}/documents",
                json=delete_request.model_dump(),
            )
            response.raise_for_status()

        st.success(f"✅ Deleted {len(selected_docs)} documents")

        # Clear cache to force refresh
        if "current_collection_info" in st.session_state:
            del st.session_state["current_collection_info"]

    except Exception as e:
        st.error(f"❌ Error deleting documents: {e}")


def _handle_collection_deletion(collection_name: str):
    """Handle deletion of entire collection."""
    # Confirmation
    if st.button(
        f"⚠️ Confirm: Delete '{collection_name}' collection",
        key=f"confirm_delete_{collection_name}",
    ):
        try:
            delete_request = DeleteRequest(
                collection_name=collection_name, delete_collection=True
            )

            with httpx.Client(timeout=settings.httpx_timeout) as client:
                response = client.request(
                    "DELETE",
                    f"{settings.api_url}/documents",
                    json=delete_request.model_dump(),
                )
                response.raise_for_status()

            st.success(f"✅ Deleted collection '{collection_name}'")

            # Clear cache to force refresh
            if "current_collection_info" in st.session_state:
                del st.session_state["current_collection_info"]

        except Exception as e:
            st.error(f"❌ Error deleting collection: {e}")
