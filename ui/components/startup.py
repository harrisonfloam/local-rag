import httpx
import streamlit as st

from app.settings import settings
from ui.components.document_portal import refresh_collection_info
from ui.utils.exceptions import StreamlitErrorMessage
from ui.utils.session_state import init_session_state


def run_startup_actions():
    """App startup procedures"""
    init_session_state()

    fetch_models()

    fetch_default_collection_info()

    display_pending_messages()


def fetch_models():
    """Fetch available models"""
    if st.session_state.get("model_info") is None:
        try:
            with httpx.Client(timeout=settings.httpx_timeout) as client:
                response = client.get(f"{settings.api_url}/models")
                response.raise_for_status()
                model_info = response.json()
        except Exception as e:
            # Set fallback info first, then raise exception
            model_info = {
                "models": [settings.model_name, settings.embedding_model_name],
                "completion_models": [settings.model_name],
                "embedding_models": [settings.embedding_model_name],
            }
            st.session_state.model_info = model_info
            st.session_state.completion_models = model_info.get("completion_models", [])
            st.session_state.embedding_models = model_info.get("embedding_models", [])
            raise StreamlitErrorMessage(
                "Error fetching models", details=str(e), icon="⚠️", style="toast"
            )
        st.session_state.model_info = model_info
        st.session_state.completion_models = model_info.get("completion_models", [])
        st.session_state.embedding_models = model_info.get("embedding_models", [])


def fetch_default_collection_info():
    """Fetch default collection info"""
    if st.session_state.get("collection_info") is None:
        refresh_collection_info(settings.collection_name)


def display_pending_messages():
    """Display any pending messages from previous operations"""
    if "pending_message" in st.session_state:
        msg = st.session_state.pending_message
        if msg["type"] == "toast":
            st.toast(msg["message"], icon=msg.get("icon", "ℹ️"))
        elif msg["type"] == "error":
            st.toast(msg["message"], icon="⚠️")
        del st.session_state.pending_message
