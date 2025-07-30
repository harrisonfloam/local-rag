import json
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Dict, List, Optional, Union

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice

from app.settings import settings
from app.utils.callbacks import CallbackMeta, with_callbacks
from app.utils.utils import truncate_long_strings

logger = logging.getLogger(__name__)

ChatMessage = Dict[str, Any]


class BaseLLMClient(ABC, metaclass=CallbackMeta):
    """Abstract base class for all LLM clients."""

    def __init__(self, log_level: str = settings.log_level, *args, **kwargs):
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized.")

    @with_callbacks
    @abstractmethod
    def chat(self, *args, **kwargs) -> Union[Any, Awaitable[Any]]:
        pass

    def _pre_chat(self, *args, **kwargs):
        payload = {**kwargs, **dict(enumerate(args))}
        truncated_payload = truncate_long_strings(payload)
        logger.debug(
            f"Chat request:\n{json.dumps(truncated_payload, indent=2, default=str)}"
        )

    def _post_chat(self, result, duration, *args, **kwargs):
        try:
            result_dump = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )
        except Exception:
            result_dump = str(result)
        result_dump = truncate_long_strings(result_dump)
        logger.debug(
            f"Chat response in {duration:.4f}s:\n{json.dumps(result_dump, indent=2, default=str)}"
        )
        return result


class OpenAILLMClient(BaseLLMClient):
    """OpenAI client for chat completions."""

    def __init__(self, log_level: str = settings.log_level, **openai_kwargs):
        super().__init__(log_level=log_level)
        self.client = openai.OpenAI(**openai_kwargs)

    def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> ChatCompletion:
        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        response = self.client.chat.completions.create(stream=False, **payload)
        return response


class AsyncOpenAILLMClient(BaseLLMClient):
    """Async OpenAI client for chat completions."""

    def __init__(self, log_level: str = settings.log_level, **openai_kwargs):
        super().__init__(log_level=log_level)
        self.client = openai.AsyncOpenAI(**openai_kwargs)

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> ChatCompletion:
        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        response = await self.client.chat.completions.create(stream=False, **payload)
        return response


class OllamaLLMClient(OpenAILLMClient):
    """Ollama client using OpenAI-compatible API."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )


class AsyncOllamaLLMClient(AsyncOpenAILLMClient):
    """Async Ollama client using OpenAI-compatible API."""

    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(
            log_level=log_level,
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )


class MockAsyncLLMClient(BaseLLMClient):
    """Base mock LLM client with configurable response types and seeding."""

    def __init__(
        self,
        response_type: Optional[int] = None,
        seed: Optional[int] = None,
        log_level: str = settings.log_level,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(log_level=log_level, **kwargs)
        self.response_type = response_type
        if seed is not None:
            random.seed(seed)

    def _create_mock_response(self, content: str) -> Union[ChatCompletion, Any]:
        """Create a mock ChatCompletion with the given content."""
        return ChatCompletion(
            id="mock_id",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model="mock-llm",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason="stop",
                )
            ],
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """Generate a mock ChatCompletion response."""
        user_message = messages[-1]["content"] if messages else "Hello"

        mock_responses = [
            user_message,  # Echo
            "This is a mock response.",
            f"This is a mock response from {model} with temperature {temperature}.",
            f"Let me think about '{user_message}'...",
        ]

        if self.response_type is None:
            content = random.choice(mock_responses)
        else:
            # Ensure response_type is within bounds
            content = mock_responses[abs(self.response_type) % len(mock_responses)]

        return self._create_mock_response(content)
