import json
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
                try:
                    if chat_request.dev.stream:
                        # Handle streaming response
                        with st.spinner("Thinking...", show_time=True):
                            stream_gen, response_time, final_response_ref = (
                                get_chat_stream(chat_request)
                            )
                        # Spinner closes here, now start streaming
                        llm_message = st.write_stream(stream_gen)
                        # Ensure llm_message is a string
                        if isinstance(llm_message, list):
                            llm_message = "".join(llm_message)

                        # Use final response if available, otherwise create from stream
                        final_response = final_response_ref["data"]
                        if final_response:
                            assistant_message = (
                                ChatMessageWithMetadata.from_assistant_response(
                                    content=str(llm_message),
                                    response_data=final_response,
                                    response_time=response_time,
                                )
                            )
                        else:
                            assistant_message = ChatMessageWithMetadata.from_stream(
                                content=str(llm_message),
                                model=chat_request.model,
                                response_time=response_time,
                            )
                    else:
                        # Handle regular response
                        with st.spinner("Thinking...", show_time=True):
                            response_data, response_time = get_chat_response(
                                chat_request
                            )
                            llm_message = response_data["choices"][0]["message"][
                                "content"
                            ]
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


def get_chat_stream(request: ChatRequest):
    """Get streaming chat response with real-time SSE parsing."""
    start_time = time.time()

    # Shared state for final response (mutable reference)
    final_response_ref = {"data": None}

    def streaming_generator():
        with httpx.Client(timeout=settings.httpx_timeout) as client:
            with client.stream(
                "POST", f"{settings.api_url}/chat", json=request.model_dump()
            ) as response:
                response.raise_for_status()

                expecting_final_data = False

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        if expecting_final_data:
                            # Parse final response and store in shared state
                            try:
                                final_response_ref["data"] = json.loads(line[6:])
                            except Exception:
                                pass
                            expecting_final_data = False
                        else:
                            # Yield content immediately for real-time streaming
                            content = line[6:]  # Remove "data: " prefix
                            yield content
                    elif line == "event: final":
                        expecting_final_data = True

    response_time = time.time() - start_time
    return streaming_generator(), response_time, final_response_ref


def get_chat_response(request: ChatRequest):
    """Get regular chat response."""
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
            prev_msg = assistant_messages[-2] if len(assistant_messages) > 1 else None

            col1, col2, col3, col4 = st.columns(4)

            # Model name
            with col1:
                st.caption(f"Model: {last_msg.metadata.model}")

            # Response time
            with col2:
                if last_msg.metadata.response_time:
                    current_time = last_msg.metadata.response_time
                    delta_text = ""
                    if prev_msg and prev_msg.metadata.response_time:
                        delta = current_time - prev_msg.metadata.response_time
                        emoji = "🔻" if delta < 0 else "🔺"
                        delta_text = f" {emoji}{abs(delta):.2f}s"

                    st.caption(f"Response Time: {current_time:.2f}s{delta_text}")

            # Token usage
            with col3:
                if last_msg.metadata.usage:
                    prompt_tokens = last_msg.metadata.usage.get("prompt_tokens", 0)
                    delta_text = ""
                    if prev_msg and prev_msg.metadata.usage:
                        prev_prompt = prev_msg.metadata.usage.get("prompt_tokens", 0)
                        delta_prompt = prompt_tokens - prev_prompt
                        if delta_prompt != 0:
                            emoji = "🔻" if delta_prompt < 0 else "🔺"
                            delta_text = f"{emoji}{abs(delta_prompt):,}"

                    st.caption(f"Prompt tokens: {prompt_tokens:,} {delta_text}")

            with col4:
                if last_msg.metadata.usage:
                    completion_tokens = last_msg.metadata.usage.get(
                        "completion_tokens", 0
                    )
                    delta_text = ""
                    if prev_msg and prev_msg.metadata.usage:
                        prev_completion = prev_msg.metadata.usage.get(
                            "completion_tokens", 0
                        )
                        delta_completion = completion_tokens - prev_completion
                        if delta_completion != 0:
                            emoji = "🔻" if delta_completion < 0 else "🔺"
                            delta_text = f"{emoji}{abs(delta_completion):,}"

                    st.caption(f"Output tokens: {completion_tokens:,} {delta_text}")

            if st.session_state.use_rag:
                with col4:
                    sources_count = (
                        len(last_msg.metadata.sources)
                        if last_msg.metadata.sources
                        else 0
                    )
                    st.caption(
                        f"RAG Sources: {sources_count if sources_count > 0 else 'None'}"
                    )
    else:
        st.caption(f"Model: {st.session_state.completion_model}")


def render_chat_history():
    chat_container = st.container(height=500, key="chat_container")
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    return chat_container
