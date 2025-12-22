from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fastapi.testclient import TestClient

from app.api.main import app
from app.settings import settings


@dataclass
class _UpsertResult:
    chunk_ids: Dict[str, List[str]]
    errors: Dict[str, str]


class _VectorStoreStub:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: Optional[str] = None,
        log_level: str = "INFO",
    ):
        _ = (collection_name, embedding_model, log_level)

    async def upsert_files(
        self,
        files: Sequence[Path],
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ) -> _UpsertResult:
        _ = (chunk_size, chunk_overlap)
        return _UpsertResult(
            chunk_ids={str(p): [f"{p.name}-chunk-1"] for p in files},
            errors={},
        )


def test_documents_sync_rejects_paths_outside_base(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "documents_path", str(tmp_path))

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    client = TestClient(app)
    payload = {
        "collection_name": "c1",
        "embedding_model": "embed-1",
        "paths": [str(outside)],
        "chunk_size": 200,
        "chunk_overlap": 10,
    }
    res = client.post("/api/documents/sync", json=payload)
    assert res.status_code == 400
    assert "must be under" in res.json()["detail"].lower()


def test_documents_sync_filters_to_allowed_extensions(monkeypatch, tmp_path):
    import app.api.routes as routes

    monkeypatch.setattr(routes, "VectorStore", _VectorStoreStub)
    monkeypatch.setattr(settings, "documents_path", str(tmp_path))

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello", encoding="utf-8")
    (docs_dir / "b.bin").write_bytes(b"nope")

    client = TestClient(app)
    payload = {
        "collection_name": "c1",
        "embedding_model": "embed-1",
        "paths": ["documents"],
        "chunk_size": 200,
        "chunk_overlap": 10,
    }
    res = client.post("/api/documents/sync", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_files"] == 1
    assert data["errors"] == {}


def test_delete_documents_rejects_empty_payload():
    """Contract test: DeleteRequest schema enforces ids-or-collection semantics."""
    client = TestClient(app)
    res = client.request(
        "DELETE",
        "/api/documents",
        json={"collection_name": "c1", "delete_collection": False, "document_ids": []},
    )
    assert res.status_code == 422
