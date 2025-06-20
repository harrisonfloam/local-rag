import httpx
import streamlit as st

from app.core.prompts import RAG_SYSTEM_PROMPT
from app.settings import settings


def main():
    st.set_page_config(page_title="local-rag", layout="wide")
    st.title("local-rag")

    with st.sidebar:
        # Fetch available models
        if "available_models" not in st.session_state:
            try:
                with httpx.Client(timeout=settings.httpx_timeout) as client:
                    response = client.get(f"{settings.api_url}/models")
                    response.raise_for_status()
                    st.session_state.available_models = response.json()["models"]
            except Exception as e:
                st.warning(f"Could not fetch models: {e}")
                st.session_state.available_models = [settings.model_name]

        file_uploader = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
        )
        st.markdown("---")
        model = st.selectbox(
            "Model",
            key="model",
            options=st.session_state.available_models,
            index=0
            if settings.model_name not in st.session_state.available_models
            else st.session_state.available_models.index(settings.model_name),
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
