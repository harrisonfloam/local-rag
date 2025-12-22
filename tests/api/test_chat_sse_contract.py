import json
import re
from typing import AsyncIterator

from fastapi.testclient import TestClient

from app.api.main import app


def test_chat_streaming_sse_preserves_embedded_newlines(monkeypatch):
    """Contract test: streaming responses use SSE-style framing and do not lose newlines."""
    import app.api.routes as routes

    async def fake_stream(*args, **kwargs) -> AsyncIterator[str | dict]:
        _ = (args, kwargs)
        yield "hello\nworld"
        yield "\n"
        yield {"choices": [{"message": {"content": "done"}}], "model": "mock-llm"}

    monkeypatch.setattr(routes, "mock_stream_chat_with_final", fake_stream)

    client = TestClient(app)
    payload = {
        "model": "mock-llm",
        "messages": [{"role": "user", "content": "hi"}],
        "dev": {"use_rag": False, "mock_llm": True, "stream": True},
    }

    with client.stream("POST", "/api/chat", json=payload) as res:
        assert res.status_code == 200
        assert res.headers.get("x-stream-response") == "true"
        text = "".join(res.iter_text())

    # Multiline chunk becomes multiline `data:` per SSE rules
    assert "data: hello\ndata: world\n\n" in text

    # Lone newline chunk becomes two empty data lines
    assert "data: \ndata: \n\n" in text

    # Final event exists and contains JSON
    m = re.search(r"event: final\s*\ndata: (\{.*\})\s*\n\n", text)
    assert m, "expected final event with json payload"
    final_obj = json.loads(m.group(1))
    assert final_obj["choices"][0]["message"]["content"] == "done"
