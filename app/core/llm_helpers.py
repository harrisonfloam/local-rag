"""Simple helper functions for working with LLMs.

This module is intentionally lightweight: it provides small functions around the
OpenAI-compatible client (Ollama) and keeps logging simple (no callback system).
"""

import logging
import random
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional, TypeVar, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel

from app.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def create_ollama_client(
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> OpenAI:
    """Create an OpenAI client configured for Ollama.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        **kwargs: Additional OpenAI client arguments

    Returns:
        OpenAI client configured for Ollama

    Example:
        >>> client = create_ollama_client()
        >>> response = client.chat.completions.create(
        ...     model="llama3.2:1b",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    return OpenAI(
        api_key="ollama",  # Ollama accepts any value
        base_url=f"{base_url.rstrip('/')}/v1",
        **kwargs,
    )


def create_async_ollama_client(
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> AsyncOpenAI:
    """Create an async OpenAI client configured for Ollama.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        **kwargs: Additional AsyncOpenAI client arguments

    Returns:
        AsyncOpenAI client configured for Ollama

    Example:
        >>> client = create_async_ollama_client()
        >>> response = await client.chat.completions.create(
        ...     model="llama3.2:1b",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    return AsyncOpenAI(
        api_key="ollama",
        base_url=f"{base_url.rstrip('/')}/v1",
        **kwargs,
    )


def completion(
    client: OpenAI,
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Simple prompt -> completion (text or structured).

    Args:
        client: OpenAI client
        prompt: User prompt
        model: Model name
        system_prompt: Optional system message
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model

    Example:
        >>> text = completion(client, "What is 2+2?", model="llama3.2:1b")
        >>> # With structured output:
        >>> answer = completion(client, "What is 2+2?", model="llama3.2:1b", response_model=Answer)
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Cast: our dicts match the ChatCompletionMessageParam protocol
    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


async def async_completion(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Async version of completion.

    Args:
        client: AsyncOpenAI client
        prompt: User prompt
        model: Model name
        system_prompt: Optional system message
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


def chat_completion(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Full chat completion with message history.

    Args:
        client: OpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = chat_completion(client, messages, model="llama3.2:1b")
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


async def async_chat_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    model: str,
    response_model: Optional[type[T]] = None,
    **kwargs,
) -> str | T:
    """Async version of chat_completion.

    Args:
        client: AsyncOpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        response_model: Optional Pydantic model for structured output
        **kwargs: Additional arguments for chat.completions.create

    Returns:
        str if no response_model, otherwise instance of response_model
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)

    if response_model:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            response_format=_build_response_format(response_model),
            **kwargs,
        )
        return response_model.model_validate_json(response.choices[0].message.content)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=msg_params,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


def stream_completion(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    **kwargs,
) -> Iterator[str | dict[str, Any]]:
    """Stream text chunks from an LLM completion, then yield a final response dict.

    Args:
        client: OpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        **kwargs: Additional arguments for chat.completions.create

    Yields:
        Text chunks as they arrive, then a final OpenAI-like dict

    Example:
        >>> for chunk in stream_completion(client, [{"role": "user", "content": "Hi"}], model="llama3.2:1b"):
        ...     print(chunk, end="", flush=True)
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)
    stream = client.chat.completions.create(
        model=model,
        messages=msg_params,
        stream=True,
        **kwargs,
    )

    content_chunks: list[str] = []
    finish_reason = "stop"

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            content_chunks.append(piece)
            yield piece
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    full_content = "".join(content_chunks)
    yield {
        "choices": [
            {"message": {"content": full_content}, "finish_reason": finish_reason}
        ],
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": len(content_chunks)},
    }


async def async_stream_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    model: str,
    **kwargs,
) -> AsyncIterator[str | dict[str, Any]]:
    """Async version of stream_completion (yields chunks, then final dict).

    Args:
        client: AsyncOpenAI client
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        **kwargs: Additional arguments for chat.completions.create

    Yields:
        Text chunks as they arrive, then a final OpenAI-like dict

    Example:
        >>> async for chunk in async_stream_completion(client, [{"role": "user", "content": "Hi"}], model="llama3.2:1b"):
        ...     print(chunk, end="", flush=True)
    """
    msg_params = cast(list[ChatCompletionMessageParam], messages)
    stream = await client.chat.completions.create(
        model=model,
        messages=msg_params,
        stream=True,
        **kwargs,
    )

    content_chunks: list[str] = []
    finish_reason = "stop"

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            content_chunks.append(piece)
            yield piece
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    full_content = "".join(content_chunks)
    yield {
        "choices": [
            {"message": {"content": full_content}, "finish_reason": finish_reason}
        ],
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": len(content_chunks)},
    }


async def mock_chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str = "mock-llm",
    temperature: float = settings.temperature,
    response_type: Optional[int] = None,
    seed: Optional[int] = None,
) -> ChatCompletion:
    """Mock ChatCompletion response (non-streaming)."""
    if seed is not None:
        random.seed(seed)

    user_message = messages[-1].get("content") if messages else "Hello"
    mock_responses = [
        user_message,
        "This is a mock response.",
        f"This is a mock response from {model} with temperature {temperature}.",
        f"Let me think about '{user_message}'...",
    ]

    if response_type is None:
        content = random.choice(mock_responses)
    else:
        content = mock_responses[abs(response_type) % len(mock_responses)]

    return _create_mock_response(content=content, model=model)


async def mock_stream_chat_with_final(
    *,
    messages: list[dict[str, Any]],
    model: str = "mock-llm",
    temperature: float = settings.temperature,
) -> AsyncIterator[str | dict[str, Any]]:
    """Mock streaming generator compatible with the legacy stream format."""
    _ = temperature
    user_message = messages[-1].get("content") if messages else "Hello"
    mock_response = (
        f"This is a mock streaming response about '{user_message}' from {model}."
    )

    word_count = 0
    for word in mock_response.split():
        yield f"{word} "
        word_count += 1

    yield {
        "choices": [{"message": {"content": mock_response}, "finish_reason": "stop"}],
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": word_count},
    }


def _build_response_format(response_model: type[BaseModel]) -> ResponseFormatJSONSchema:
    """Internal helper to build OpenAI response_format."""
    return ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchema(
            name=response_model.__name__,
            schema=response_model.model_json_schema(),
            strict=True,
        ),
    )


def _log_chat_request(
    *, model: str, temperature: float, messages: list[dict[str, Any]]
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "LLM request: model=%s temperature=%s messages=%s",
        model,
        temperature,
        len(messages),
    )


def _log_chat_response(*, model: str, content_len: int, duration_s: float) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "LLM response: model=%s content_len=%s duration_s=%.4f",
        model,
        content_len,
        duration_s,
    )


def _create_mock_response(*, content: str, model: str) -> ChatCompletion:
    return ChatCompletion(
        id="mock_id",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )
