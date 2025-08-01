import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.ingestor import RetrievedDocumentChunk


class MessageMetadata(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

    # For user messages
    request_params: Optional[Dict[str, Any]] = None

    # For assistant messages
    response_data: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    sources: Optional[List[RetrievedDocumentChunk]] = []


class ChatMessageWithMetadata(BaseModel):
    """Enhanced chat message with OpenAI compatibility."""

    role: str
    content: str
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)

    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI API format (just role + content)."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_user_input(
        cls, content: str, request_params: Optional[Dict[str, Any]] = None
    ) -> "ChatMessageWithMetadata":
        """Create user message with request parameters."""
        return cls(
            role="user",
            content=content,
            metadata=MessageMetadata(request_params=request_params or {}),
        )

    @classmethod
    def from_assistant_response(
        cls,
        content: str,
        response_data: Dict[str, Any],
        response_time: Optional[float] = None,
    ) -> "ChatMessageWithMetadata":
        """Create assistant message from API response."""
        return cls(
            role="assistant",
            content=content,
            metadata=MessageMetadata(
                response_data=response_data,
                response_time=response_time,
                model=response_data.get("model"),
                usage=response_data.get("usage"),
                finish_reason=response_data.get("choices", [{}])[0].get(
                    "finish_reason"
                ),
                sources=response_data.get("sources", []),
            ),
        )

    @classmethod
    def from_stream(
        cls,
        content: str,
        model: str,
        response_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ChatMessageWithMetadata":
        """Create assistant message from streaming response."""
        # Use metadata if available, otherwise create mock data
        if metadata:
            usage = metadata.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
            finish_reason = metadata.get("finish_reason", "stop")
            sources = metadata.get("sources", [])
            model_name = metadata.get("model", model)
        else:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            finish_reason = "stop"
            sources = []
            model_name = model

        return cls(
            role="assistant",
            content=content,
            metadata=MessageMetadata(
                response_data={
                    "choices": [
                        {
                            "message": {"content": content},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "model": model_name,
                    "usage": usage,
                    "sources": sources,
                },
                response_time=response_time,
                model=model_name,
                usage=usage,
                finish_reason=finish_reason,
                sources=sources,
            ),
        )
