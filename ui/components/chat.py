import time

import httpx
import streamlit as st

from app.api.schemas import ChatRequest
from app.settings import settings
from ui.models import ChatMessageWithMetadata
from ui.utils.session_state import get_chat_request_params


def render_chat():
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []

    chat_container = render_chat_history()
    handle_chat_input(chat_container)


def handle_chat_input(chat_container):
    if prompt := st.chat_input("Start a new chat..."):
        chat_request_params = get_chat_request_params()
        user_message = ChatMessageWithMetadata.from_user_input(
            content=prompt, request_params=chat_request_params
        )
        st.session_state.messages.append(user_message.model_dump())
        chat_request = ChatRequest(
            messages=[
                ChatMessageWithMetadata(**msg).to_openai_format()
                for msg in st.session_state.messages
            ],
            **chat_request_params,
        )

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)  # Display user message
            with st.chat_message("assistant"):
                with st.spinner("Thinking...", show_time=True):
                    try:
                        # Get LLM response
                        response_data, response_time = get_chat_response(chat_request)
                        # TODO: handle truncation check

                        llm_message = response_data["choices"][0]["message"]["content"]
                        # TODO: extract sources, etc

                        st.markdown(llm_message)

                        assistant_message = (
                            ChatMessageWithMetadata.from_assistant_response(
                                content=llm_message,
                                response_data=response_data,
                                response_time=response_time,
                            )
                        )
                        st.session_state.messages.append(assistant_message.model_dump())

                    except Exception as e:
                        st.error(f"Error: {e}")


def get_chat_response(request: ChatRequest):
    start_time = time.time()
    with httpx.Client(timeout=settings.httpx_timeout) as client:
        chat_response = client.post(
            f"{settings.api_url}/chat",
            json=request.model_dump(),
        )
        chat_response.raise_for_status()
    response_time = time.time() - start_time
    return chat_response.json(), response_time


def render_chat_metrics():
    if st.session_state.messages:
        # Get last assistant message
        assistant_messages = [
            ChatMessageWithMetadata(**msg)
            for msg in st.session_state.messages
            if msg["role"] == "assistant"
        ]

        if assistant_messages:
            last_msg = assistant_messages[-1]

            # Check if model changed
            model_changed = False
            if len(assistant_messages) > 1:
                previous_model = assistant_messages[-2].metadata.model
                model_changed = previous_model != last_msg.metadata.model

            # Display current model info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if model_changed:
                    st.metric(
                        "Current Model",
                        last_msg.metadata.model,
                        delta="Changed",
                        delta_color="normal",
                    )
                else:
                    st.caption(f"🤖 {last_msg.metadata.model}")

            with col2:
                if last_msg.metadata.response_time:
                    # Show response time with trend
                    current_time = last_msg.metadata.response_time
                    delta_time = None
                    if len(assistant_messages) > 1:
                        prev_time = assistant_messages[-2].metadata.response_time
                        if prev_time:
                            delta_time = current_time - prev_time

                    st.metric(
                        "Response Time",
                        f"{current_time:.2f}s",
                        delta=f"{delta_time:+.2f}s" if delta_time else None,
                        delta_color="inverse",  # Lower is better
                    )

            with col3:
                if last_msg.metadata.usage:
                    total_tokens = last_msg.metadata.usage.get("total_tokens", 0)
                    # Show token usage with trend
                    delta_tokens = None
                    if len(assistant_messages) > 1:
                        prev_usage = assistant_messages[-2].metadata.usage
                        if prev_usage:
                            prev_tokens = prev_usage.get("total_tokens", 0)
                            delta_tokens = total_tokens - prev_tokens

                    st.metric(
                        "Total Tokens",
                        total_tokens,
                        delta=delta_tokens,
                        delta_color="off",  # Neutral color for tokens
                    )

            with col4:
                if last_msg.metadata.sources:
                    st.metric("Sources", len(last_msg.metadata.sources))
                else:
                    st.caption("No RAG sources")
    else:
        st.caption(f"🤖 Ready: {st.session_state.completion_model}")


def render_chat_history():
    chat_container = st.container(height=500, key="chat_container")
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    return chat_container
