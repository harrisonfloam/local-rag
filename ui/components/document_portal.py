from typing import List, Optional

import httpx
import pandas as pd
import streamlit as st

from app.api.schemas import DeleteRequest, IngestRequest
from app.settings import settings
from ui.utils.exceptions import (
    StreamlitErrorMessage,
    StreamlitRerunNeeded,
    StreamlitToastMessage,
)


def render_document_portal():
    # Initialize uploader key nonce
    if "uploader_key_nonce" not in st.session_state:
        st.session_state.uploader_key_nonce = 0

    uploaded_files = st.file_uploader(
        "Select files to upload",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key_nonce}",
    )

    col1, col2 = st.columns(2)
    with col1:
        # TODO: need to sanitize this
        collection_name = st.text_input(
            "Collection Name",
            key="collection_name",
            value=settings.collection_name,
        )
    with col2:
        embedding_model = st.selectbox(
            "Embedding Model",
            options=st.session_state.get(
                "embedding_models", [settings.embedding_model_name]
            ),
            key="embedding_model",
            index=0,
        )

    col3, col4 = st.columns(2)
    with col3:
        chunk_size = st.number_input(
            "Chunk Size",
            key="chunk_size",
            min_value=50,
            max_value=2048,
            value=settings.chunk_size,
            step=50,
        )
    with col4:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            key="chunk_overlap",
            min_value=0,
            max_value=512,
            value=settings.chunk_overlap,
            step=10,
        )

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button(
            "Embed Documents",
            key="embed_documents",
            disabled=not uploaded_files,
            type="primary",
        ):
            handle_document_upload(
                uploaded_files,
                collection_name,
                embedding_model,
                chunk_size,
                chunk_overlap,
            )
    with button_col2:
        if st.button("Load Collection", key="load_collection"):
            with st.spinner(f"Loading collection '{collection_name}'..."):
                refresh_collection_info(collection_name)

    render_document_list()


# @st.fragment
def render_document_list():
    """Render document list using st.dataframe with row selection."""
    # TODO: is collection name always the same one that's in the field? what about when we haven't loaded it?
    collection_info = st.session_state.collection_info
    collection_name = st.session_state.collection_name
    if not collection_info:
        st.info("👆 Click `Load Collection` to browse documents")
        return

    documents = collection_info.get("documents", [])
    if not documents:
        st.info("📄 No documents in this collection")
        return

    # Create dataframe from documents
    df_data = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        df_data.append(
            {
                "Name": doc.get("title", "unknown")[:40],
                "Size": f"{metadata.get('file_size', 0):,}B",
                "Chunks": metadata.get("total_chunks", 0),
                "Chunk Size": metadata.get("chunk_size", "N/A"),
                "Chunk Overlap": metadata.get("chunk_overlap", "N/A"),
                "Status": "✅ embedded",
                "ID": doc.get("id", "unknown"),
            }
        )

    df = pd.DataFrame(df_data)
    select_all = st.checkbox(
        "Select All Documents", value=False, key="select_all_documents"
    )
    document_list_df = st.dataframe(
        df,
        key="document_list_df",
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "💡 Hold Ctrl+A or Shift+Click to select multiple rows in the table above"
    )

    selected_rows = (
        document_list_df.get("selection", {}).get("rows", [])
        if document_list_df
        else []
    )

    delete_selected = st.button(
        "🗑️ Delete Selected",
        key="delete_selected_docs",
        disabled=not (selected_rows or select_all),
    )

    if delete_selected:
        if select_all:
            # If select all is checked, delete collection
            selected_doc_ids = [doc["id"] for doc in documents]
            delete_collection(collection_name)
        elif selected_rows:
            # Delete selected rows
            selected_doc_ids = [documents[i]["id"] for i in selected_rows]
            delete_documents(collection_name, selected_doc_ids)
        else:
            raise StreamlitToastMessage("No documents selected", icon="⚠️")


def refresh_collection_info(collection_name: Optional[str] = None) -> bool:
    """Fetch collection information from the API. Returns True if successful."""
    try:
        with httpx.Client(timeout=settings.httpx_timeout) as client:
            params = {
                "collection_name": collection_name or st.session_state.collection_name
            }
            response = client.get(f"{settings.api_url}/documents", params=params)
            response.raise_for_status()
            collection_info = response.json()
            st.session_state.collection_info = collection_info
            raise StreamlitRerunNeeded("Collection info refreshed")
    except StreamlitRerunNeeded:
        raise
    except Exception as e:
        raise StreamlitErrorMessage(
            "Error fetching collection info", details=str(e), icon="❌"
        )


def handle_document_upload(
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

            # Clear the file uploader by incrementing the key nonce
            st.session_state.uploader_key_nonce += 1

            if "chunk_ids" in response_data:
                errors = response_data.get("errors", {})
                if errors:
                    # Clear collection info to refresh on next rerun
                    st.session_state.collection_info = None
                    raise StreamlitRerunNeeded(
                        f"Embedded {len(files) - len(errors)} files, {len(errors)} errors",
                        show_after_rerun=True,
                        pending_message_kwargs={"icon": "⚠️"},
                    )
                else:
                    refresh_collection_info(collection_name)

        except (StreamlitRerunNeeded, StreamlitErrorMessage, StreamlitToastMessage):
            raise
        except Exception as e:
            raise StreamlitErrorMessage("Error during ingestion", details=str(e))


def delete_documents(collection_name: str, selected_docs: Optional[List[str]] = None):
    """Handle deletion of selected documents."""
    if not selected_docs:
        selected_docs = st.session_state.get("selected_documents", [])

    if not selected_docs:
        raise StreamlitToastMessage("No documents selected for deletion", icon="⚠️")

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

            refresh_collection_info(collection_name)
            # TODO: i shouldnt need these reruns here...

    except (StreamlitRerunNeeded, StreamlitErrorMessage, StreamlitToastMessage):
        raise
    except Exception as e:
        raise StreamlitErrorMessage("Error deleting documents", details=str(e))


def delete_collection(collection_name: str):
    """Handle deletion of entire collection."""
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

            refresh_collection_info(collection_name)

    except (StreamlitRerunNeeded, StreamlitErrorMessage, StreamlitToastMessage):
        raise
    except Exception as e:
        raise StreamlitErrorMessage("Error deleting collection", details=str(e))
