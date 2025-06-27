import httpx
import streamlit as st

from app.core.prompts import RAG_SYSTEM_PROMPT
from app.settings import settings


def main():
    st.set_page_config(page_title="local-rag", layout="wide")
    st.title("local-rag")

    with st.sidebar:
        # Fetch available models
        if "model_info" not in st.session_state:
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

        file_uploader = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
        )
        st.markdown("---")
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
        st.markdown("---")
        # Dropdowns and inputs
        completion_model = st.selectbox(
            "Completion model",
            key="completion_model",
            options=st.session_state.completion_models
            if not st.session_state.mock_llm
            else ["mock-llm"],
            index=0
            if (
                settings.model_name not in st.session_state.completion_models
                or settings.mock_llm
            )
            else st.session_state.completion_models.index(settings.model_name),
        )
        embedding_model = st.selectbox(
            "Embedding model",
            key="embedding_model",
            options=st.session_state.embedding_models,
            index=0
            if settings.embedding_model_name not in st.session_state.embedding_models
            else st.session_state.embedding_models.index(settings.embedding_model_name),
        )
        temperature = st.number_input(
            "Temperature",
            key="temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.temperature,
            step=0.05,
        )
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
        use_rag = st.checkbox(
            "Use RAG",
            key="use_rag",
            value=settings.use_rag,
            help="Use RAG context for the chat",
        )
        mock_llm = st.checkbox(
            "Mock LLM",
            key="mock_llm",
            value=settings.mock_llm,
            help="Use a mock LLM for testing purposes",
        )

    # Request keys
    CHAT_REQUEST_KEYS = ["model", "temperature", "top_k", "use_rag", "mock_llm"]
    # TODO: ingest request keys

    # Clear chat
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Start a new chat..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)  # Display user message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get LLM response
                        with httpx.Client(timeout=settings.httpx_timeout) as client:
                            chat_response = client.post(
                                f"{settings.api_url}/chat",
                                json={
                                    "messages": st.session_state.messages,
                                    **{
                                        key: st.session_state[key]
                                        for key in CHAT_REQUEST_KEYS
                                    },
                                },
                            )
                            chat_response.raise_for_status()

                        llm_message = chat_response.json()["response"]
                        st.markdown(llm_message)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": llm_message}
                        )

                    except Exception as e:
                        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
