from typing import Any, Dict

import streamlit as st

from app.settings import settings


def init_session_state():
    """
    Initialize the session state with default values.
    This function sets up the session state for the application,
    ensuring that necessary keys are present with default values.
    """
    session_state_defaults = {
        "mock_llm": settings.mock_llm,
        "mock_rag_response": settings.mock_rag_response,
        "model_info": None,
        "completion_models": [],
        "embedding_models": [],
        "messages": [],
    }
    for key, value in session_state_defaults.items():
        st.session_state.setdefault(key, value)


def get_chat_request_params() -> Dict[str, Any]:
    """Extract chat request parameters from session state."""
    return {
        "model": st.session_state.completion_model,
        "system_prompt": st.session_state.system_prompt,
        "temperature": st.session_state.temperature,
        "embedding_model": st.session_state.embedding_model,
        "top_k": st.session_state.top_k,
        "use_rag": st.session_state.use_rag,
        "mock_llm": st.session_state.mock_llm,
    }


def get_ingest_request_params() -> Dict[str, Any]:
    pass
