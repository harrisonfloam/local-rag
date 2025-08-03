import httpx
import streamlit as st

from app.settings import settings
from ui.components.document_portal import _fetch_collection_info


def run_startup_actions():
    """App startup procedures"""
    fetch_models()

    fetch_default_collection_info()


def fetch_models():
    """Fetch available models"""
    if st.session_state.get("model_info") is None:
        try:
            with httpx.Client(timeout=settings.httpx_timeout) as client:
                response = client.get(f"{settings.api_url}/models")
                response.raise_for_status()
                model_info = response.json()
        except Exception:
            st.warning("Error fetching models.")
            model_info = {
                "models": [settings.model_name, settings.embedding_model_name],
                "completion_models": [settings.model_name],
                "embedding_models": [settings.embedding_model_name],
            }
        st.session_state.model_info = model_info
        st.session_state.completion_models = model_info.get("completion_models", [])
        st.session_state.embedding_models = model_info.get("embedding_models", [])


def fetch_default_collection_info():
    """Fetch default collection info"""
    if st.session_state.get("current_collection_info") is None:
        collection_info = _fetch_collection_info(settings.collection_name)
        if collection_info and "error" not in collection_info:
            st.session_state.current_collection_info = collection_info
            st.session_state.current_collection_name = settings.collection_name
