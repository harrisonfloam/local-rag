import streamlit as st

from app.core.prompts import RAG_SYSTEM_PROMPT
from app.settings import settings


def render_sidebar():
    """
    Sidebar contains:
    - Model selection
    - Hyperparameter selection
    - Dev mode options
    """
    with st.sidebar:
        render_checkboxes()
        st.markdown("---")
        render_model_selection()
        render_param_selection()


def render_checkboxes():
    # Check boxes
    col1, col2 = st.columns(2)
    with col1:
        use_rag = st.checkbox(
            "Use RAG",
            key="use_rag",
            value=settings.use_rag,
            # help="Use RAG context for the chat",
        )
    with col2:
        dev_mode = st.checkbox(
            "Dev Mode",
            key="dev_mode",
            value=settings.dev_mode,
            help="Enable dev mode for additional options",
        )
    # Dev mode options
    if st.session_state.dev_mode:
        col1, col2 = st.columns(2)
        with col1:
            mock_llm = st.checkbox(
                "Mock LLM",
                key="mock_llm",
                value=settings.mock_llm,
                # help="Use a mock LLM for testing purposes",
            )
        with col2:
            mock_rag_response = st.checkbox(
                "Mock RAG Response",
                key="mock_rag_response",
                value=settings.mock_rag_response,
                # help="Use a mock RAG response for testing purposes",
            )

        col1, col2 = st.columns(2)
        with col1:
            stream = st.checkbox(
                "Stream",
                key="stream",
                value=settings.stream,
                # help="Stream chat responses",
            )
    else:
        st.session_state.mock_llm = settings.mock_llm
        st.session_state.mock_rag_response = settings.mock_rag_response
        st.session_state.stream = settings.stream


def render_model_selection():
    completion_model = st.selectbox(
        "Completion model",
        key="completion_model",
        options=st.session_state.completion_models
        if not st.session_state.mock_llm
        else ["mock-llm"],
        index=(
            0
            if st.session_state.mock_llm
            else (
                st.session_state.completion_models.index(settings.model_name)
                if settings.model_name in st.session_state.completion_models
                else 0
            )
        ),
    )
    embedding_model = st.selectbox(
        "Embedding model",
        key="embedding_model",
        options=st.session_state.embedding_models,
        index=0
        if settings.embedding_model_name not in st.session_state.embedding_models
        else st.session_state.embedding_models.index(settings.embedding_model_name),
    )


def render_param_selection():
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input(
            "Temperature",
            key="temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.temperature,
            step=0.05,
        )
    with col2:
        top_k = st.number_input(
            "Top K",
            key="top_k",
            min_value=1,
            max_value=50,
            value=settings.top_k,
            step=1,
        )
    system_prompt = st.text_area(
        "System Prompt",
        key="system_prompt",
        value=RAG_SYSTEM_PROMPT,
        height=90,
        help="System prompt to guide the LLM's behavior",
    )
