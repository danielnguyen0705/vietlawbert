import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from api.providers import MockProvider, RagProvider
from api.server import create_app


@pytest.fixture
def client():
    with TestClient(create_app("mock")) as client:
        yield client


def test_contract_and_mock(client):
    health = client.get("/api/v1/health").json()
    assert health["ready"] and health["model"] == "not_required"
    assert set(health["databases"].values()) == {"not_required"}
    for topic in ("giao thông", "doanh nghiệp", "hành chính"):
        data = client.post("/api/v1/chat", json={"query": f"  Demo {topic}  ", "top_k": 2}).json()
        assert data["query"] == f"Demo {topic}"
        assert len(data["contexts"]) == 2
        assert data["mode"] == "mock" and data["has_context"]
        assert data["latency_ms"] >= 0
        assert all(c["doc_number"].startswith("DEMO-") for c in data["contexts"])
    result = client.post("/api/v1/chat", json={"query": "không có mẫu"})
    assert result.status_code == 200
    assert not result.json()["has_context"]
    assert result.json()["contexts"] == []


@pytest.mark.parametrize("body", [
    {"query": ""}, {"query": "   "}, {"query": "x" * 4001}, {"query": None},
    {"query": "x", "top_k": 0}, {"query": "x", "top_k": 11},
    {"query": "x", "top_k": 1.5}, {"query": "x", "top_k": True},
    {"query": "x", "stream": True},
])
def test_validation(client, body):
    assert client.post("/api/v1/chat", json=body).status_code == 422


def test_mock_does_not_import_ai():
    subprocess.run([sys.executable, "-c", "from api.server import create_app; from fastapi.testclient import TestClient; import sys\nwith TestClient(create_app('mock')) as c: assert c.post('/api/v1/chat', json={'query':'Demo giao thông'}).status_code == 200\nassert not any(m in sys.modules for m in ['torch','rag.generator','configs.config','sentence_transformers'])"], check=True)


def test_unavailable_and_unexpected_error():
    def broken(mode):
        raise RuntimeError("private startup detail")
    with TestClient(create_app("rag", broken)) as c:
        assert c.get("/api/v1/health").status_code == 503
        assert c.post("/api/v1/chat", json={"query": "x"}).status_code == 503

    class Broken(MockProvider):
        def ask(self, query, top_k):
            raise RuntimeError("private error")
    with TestClient(create_app("mock", lambda mode: Broken())) as c:
        response = c.post("/api/v1/chat", json={"query": "x"})
        assert response.status_code == 500
        assert "private" not in response.text


def test_rag_adapter_lifecycle_and_failed_generation():
    class Generator:
        closed = 0
        calls = []
        failed = False
        def ask(self, **kwargs):
            self.calls.append(kwargs)
            return {"answer": "answer", "contexts": [{"text": "source"}], "model_used": "Failed" if self.failed else "test"}
        def close(self):
            self.closed += 1
    generator = Generator()
    created = []
    def factory(mode):
        created.append(mode)
        return RagProvider(generator)
    with TestClient(create_app("rag", factory)) as c:
        data = c.post("/api/v1/chat", json={"query": "hello", "top_k": 2}).json()
        assert data["contexts"][0]["content"] == "source"
        assert data["contexts"][0]["graph_boost"] is None
        assert data["attribution_score"] is None
        assert generator.calls == [{"query": "hello", "top_k": 2}]
        generator.failed = True
        assert c.post("/api/v1/chat", json={"query": "x"}).status_code == 502
    assert created == ["rag"] and generator.closed == 1


def test_rag_busy_does_not_block_health():
    entered, release = threading.Event(), threading.Event()
    class Slow(MockProvider):
        def ask(self, query, top_k):
            entered.set()
            assert release.wait(5)
            return super().ask(query, top_k)
    with TestClient(create_app("rag", lambda mode: Slow())) as c, ThreadPoolExecutor() as pool:
        first = pool.submit(c.post, "/api/v1/chat", json={"query": "giao thông"})
        try:
            assert entered.wait(3)
            assert c.get("/api/v1/health").status_code == 200
            assert c.post("/api/v1/chat", json={"query": "other"}).status_code == 503
        finally:
            release.set()
        assert first.result().status_code == 200


def test_openapi(client):
    document = client.get("/openapi.json").json()
    assert "/api/v1/chat" in document["paths"]
    assert document["components"]["schemas"]["ChatRequest"]["examples"]
