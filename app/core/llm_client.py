import json
import logging
from typing import Any, Dict, List

import openai

from app.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        self.client = openai.OpenAI(
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )
        logger.setLevel(log_level.upper())
        logger.debug("LLMClient initialized.")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        logger.debug(
            f"LLM request payload:\n{json.dumps(payload, indent=2, default=str)}"
        )
        response = self.client.chat.completions.create(**payload)
        logger.debug(f"LLM response:\n{json.dumps(response, indent=2, default=str)}")
        return response


class AsyncLLMClient:
    def __init__(
        self, ollama_url: str = settings.ollama_url, log_level: str = settings.log_level
    ):
        self.client = openai.AsyncOpenAI(
            base_url=ollama_url,
            api_key="ollama",  # required, but unused
        )
        logger.setLevel(log_level.upper())

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = settings.model_name,
        temperature: float = settings.temperature,
        **kwargs,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        logger.debug(
            f"LLM request payload:\n{json.dumps(payload, indent=2, default=str)}"
        )
        response = await self.client.chat.completions.create(**payload)
        logger.debug(f"LLM response:\n{json.dumps(response, indent=2, default=str)}")
        return response
