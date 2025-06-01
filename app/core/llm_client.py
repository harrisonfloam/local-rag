from typing import Any, Dict, List

import openai

from app.settings import settings


class LLMClient:
    def __init__(self, model: str = settings.LLM_NAME):
        self.model = model
        self.client = openai.OpenAI(
            base_url=settings.OLLAMA_URL,
            api_key="ollama",  # required, but unused
        )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            **kwargs,
        )  # type: ignore


class AsyncLLMClient:
    def __init__(self, model: str = settings.LLM_NAME):
        self.model = model
        self.client = openai.AsyncOpenAI(
            base_url=settings.OLLAMA_URL,
            api_key="ollama",  # required, but unused
        )

    async def chat(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            **kwargs,
        )  # type: ignore
