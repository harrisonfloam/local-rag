import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice

from app.settings import settings
from app.utils.callbacks import CallbackMeta, with_callbacks

logger = logging.getLogger(__name__)

ChatMessage = Dict[str, Any]


class BaseLLMClient(ABC, metaclass=CallbackMeta):
    def __init__(self, log_level: str = settings.log_level, *args, **kwargs):
        logger.setLevel(log_level.upper())
        logger.debug(f"{self.__class__.__name__} initialized.")

    @with_callbacks
    @abstractmethod
    async def chat(self, *args, **kwargs) -> ChatCompletion:
        pass

    def _pre_chat(self, *args, **kwargs):
        payload = {**kwargs, **dict(enumerate(args))}
        logger.debug(f"Chat request:\n{json.dumps(payload, indent=2, default=str)}")

    def _post_chat(self, result, duration, *args, **kwargs):
        logger.debug(
            f"Chat response:\n{json.dumps(result.model_dump(), indent=2, default=str)}"
        )
        logger.info(f"Chat request completed in {duration:.4f}s")
        return result


class AsyncLLMClient(BaseLLMClient):
    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(log_level=log_level)
        self.client = openai.AsyncOpenAI(
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> ChatCompletion:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        return response


class MockAsyncLLMClient(BaseLLMClient):
    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        super().__init__(log_level=log_level)

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> ChatCompletion:
        response = ChatCompletion(
            id="mock_id",
            object="chat.completion",
            created=1234567890,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="This is a mock response.",
                    ),
                    finish_reason="stop",
                )
            ],
        )
        return response


# TODO: structured clients with Instructor
