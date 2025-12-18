from fastapi.testclient import TestClient

from app.api.main import app


def test_health_ok():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_retrieve_mock_ok():
    client = TestClient(app)
    payload = {
        "query": "hello",
        "top_k": 3,
        "dev": {"mock_rag_response": True},
    }
    res = client.post("/api/retrieve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "hello"
    assert len(data["results"]) == 3
    assert all("mock retrieved chunk" in r["content"] for r in data["results"])


def test_chat_mock_llm_and_mock_rag_ok():
    client = TestClient(app)
    payload = {
        "model": "mock-llm",
        "messages": [{"role": "user", "content": "what is up"}],
        "top_k": 2,
        "dev": {
            "use_rag": True,
            "mock_llm": True,
            "mock_rag_response": True,
            "stream": False,
        },
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"]
    assert "sources" in data
    assert len(data["sources"]) == 2
