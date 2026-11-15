"""Check HTTP contracts and resource ownership without loading real models."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Event, get_ident
from time import sleep
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.backend.config import Settings
from app.backend.main import create_app
from app.backend.runtime import Runtime
from app.backend.schemas import public_answer


def answer_result():
    metadata = {"chapter_number": "1", "section_number": "1.1", "section_title": "Introduction"}
    source = {"source_id": "S1", "point_id": 7, "pdf_pages": [34], "metadata": metadata}
    return {"query": "What is a database?", "language": "en", "status": "answered",
            "answer": "A collection of related data.", "citations": [source], "sources": [source],
            "retrieved_hits": [{"point_id": 7, "page_content": "A database is a collection of related data."}],
            "retrieval_seconds": 1.0, "generation_seconds": 2.0, "total_seconds": 3.0,
            "prompt": "private prompt", "error": "token-secret C:/private/path"}


class FakeRuntime:
    def __init__(self, status="ready"):
        self.status = status
        self.generators = {name: SimpleNamespace(model=f"test-{name}") for name in ("local", "api")}
        self.models = [{"backend": name, "model": item.model, "available": True, "reason": None}
                       for name, item in self.generators.items()]
        self.calls = []
        self.threads = []
        self.closed = False

    def load(self):
        self.threads.append(get_ident())

    def close(self):
        self.threads.append(get_ident())
        self.closed = True

    def answer(self, question, *, language, backend):
        self.threads.append(get_ident())
        self.calls.append((question, language, backend))
        return {**answer_result(), "query": question, "language": language}


@pytest.mark.parametrize("backend", ["local", "api"])
def test_ask_routes_to_backend_and_returns_only_public_fields(backend):
    runtime = FakeRuntime()
    with TestClient(create_app(Settings(), runtime)) as client:
        result = client.post("/api/ask", json={"question": "  What is a database?  ", "language": "en", "backend": backend})
        assert result.status_code == 200
        body = result.json()
        assert runtime.calls == [("What is a database?", "en", backend)]
        assert body["sources"][0]["pdf_pages"] == [35]
        assert body["sources"][0]["text"].startswith("A database")
        assert body["citations"] == ["S1"] and body["model"] == f"test-{backend}"
        assert body["error"] is None
        assert not any(secret in result.text for secret in ["private", "token-secret", "prompt", "metadata"])
        assert client.get("/api/health").json()["ready"]
    assert runtime.closed and len(set(runtime.threads)) == 1


@pytest.mark.parametrize("body", [
    {"question": "   "}, {"question": "x" * 1001}, {"question": "q", "language": "fr"},
    {"question": "q", "backend": "unknown"}, {"question": "q", "base_url": "https://evil.test"},
])
def test_invalid_input_never_calls_pipeline(body):
    runtime = FakeRuntime()
    with TestClient(create_app(Settings(), runtime)) as client:
        assert client.post("/api/ask", json=body).status_code == 422
    assert not runtime.calls


def test_unavailable_backend_and_examples_mode():
    runtime = FakeRuntime("unavailable")
    with TestClient(create_app(Settings(), runtime)) as client:
        assert client.post("/api/ask", json={"question": "q"}).status_code == 503
        assert not any(option["available"] for option in client.get("/api/models").json()["models"])
        assert client.get("/api/health").json()["status"] == "unavailable"
    examples = Runtime(Settings(mode="examples"))
    with TestClient(create_app(Settings(mode="examples"), examples)) as client:
        for _ in range(50):
            if client.get("/api/health").json()["status"] == "examples":
                break
            sleep(.01)
        assert client.get("/api/health").json()["status"] == "examples"
        assert client.post("/api/ask", json={"question": "q"}).status_code == 503
        assert examples.retriever is None and not examples.generators


def test_slow_inference_does_not_block_health_and_rejects_parallel_requests():
    runtime = FakeRuntime()
    started, release = Event(), Event()
    original = runtime.answer
    def slow(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    runtime.answer = slow
    with TestClient(create_app(Settings(), runtime)) as client, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.post, "/api/ask", json={"question": "first"})
        try:
            assert started.wait(5)
            assert client.get("/api/health").json()["busy"]
            rejected = client.post("/api/ask", json={"question": "second"})
            assert rejected.status_code == 429 and rejected.headers["retry-after"] == "10"
        finally:
            release.set()
        assert future.result().status_code == 200
        assert not client.get("/api/health").json()["busy"]


def test_rate_limit_and_cors():
    with TestClient(create_app(Settings(requests_per_hour=1), FakeRuntime())) as client:
        assert client.post("/api/ask", json={"question": "q"}).status_code == 200
        assert client.post("/api/ask", json={"question": "q"}).status_code == 429
        allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
        denied = client.get("/api/health", headers={"Origin": "https://unlisted.test"})
        assert "access-control-allow-origin" not in denied.headers


def test_error_output_does_not_expose_private_diagnostics():
    result = answer_result()
    result.update(status="error", answer="", citations=[])
    public = public_answer(result, backend="api", model="test", request_id="id")
    assert public.status == "error" and public.error
    assert "token-secret" not in public.model_dump_json() and "private" not in public.model_dump_json()


def test_runtime_shares_retriever_without_reloading_or_changing_generator():
    calls = []
    class Retriever:
        def retrieve(self, question, **kwargs):
            calls.append(question)
            return []
    def generate(question, hits, **kwargs):
        return {"status": "insufficient_evidence", "answer": "No evidence", "citations": [], "sources": []}
    runtime = Runtime(Settings())
    runtime.retriever = Retriever()
    local, api = SimpleNamespace(generate=generate), SimpleNamespace(generate=generate)
    runtime.generators = {"local": local, "api": api}
    runtime.answer("one", language="vi", backend="local")
    runtime.answer("two", language="en", backend="api")
    assert calls == ["one", "two"] and runtime.generators == {"local": local, "api": api}


def test_runtime_loads_one_index_and_releases_it(monkeypatch):
    import sys
    from app.backend import runtime as module
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    events = []
    retriever = object()
    @contextmanager
    def open_index(*args, **kwargs):
        events.append("open")
        try:
            yield SimpleNamespace(retriever=retriever)
        finally:
            events.append("close")
    monkeypatch.setattr(module, "open_rag_pipeline", open_index)
    runtime = Runtime(Settings())
    def generators():
        runtime.generators = {"local": object(), "api": object()}
    monkeypatch.setattr(runtime, "_load_generators", generators)
    runtime.load()
    assert runtime.status == "ready" and runtime.retriever is retriever and events == ["open"]
    runtime.close()
    assert events == ["open", "close"] and runtime.retriever is None


def test_api_generator_uses_configured_counter_and_source_mapping(monkeypatch):
    import sys
    from src.generation.api_generation import APIGenerator
    counter = lambda messages: len(json.dumps(messages))
    monkeypatch.setitem(sys.modules, "app_test_counter", SimpleNamespace(count=counter))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=None))
    monkeypatch.setenv("APP_TEST_KEY", "server-only-key")
    settings = Settings(api_model="test-api", api_url="https://provider.test/v1",
                        api_token_counter="app_test_counter:count", api_key_env="APP_TEST_KEY")
    generator = Runtime(settings)._generator("api")
    assert isinstance(generator, APIGenerator) and generator.count_tokens is counter
    raw = json.dumps({"status": "answered", "answer": "Related data.", "citations": ["S1"]})
    monkeypatch.setattr(generator, "_request", lambda payload: {
        "choices": [{"finish_reason": "stop", "message": {"content": raw}}]})
    hits = [{"point_id": 7, "page_content": "A database is a collection of related data.",
             "metadata": {"sources": [{"record_id": "book:p34", "pdf_page": 34}]}}]
    result = generator.generate("What is a database?", hits, language="en")
    assert result["status"] == "answered" and result["citations"][0]["point_id"] == 7
    assert "server-only-key" not in json.dumps(result)


def test_local_failure_does_not_disable_configured_api(monkeypatch):
    runtime = Runtime(Settings(api_model="test-api", api_url="https://provider.test/v1"))
    api = object()
    def generator(backend):
        if backend == "local":
            raise RuntimeError("private-path secret-key")
        return api
    monkeypatch.setattr(runtime, "_generator", generator)
    runtime._load_generators()
    assert runtime.generators == {"api": api}
    assert runtime.models[1]["available"] and not runtime.models[0]["available"]
    assert "secret-key" not in json.dumps(runtime.models)


def test_settings_and_recorded_examples(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "examples")
    monkeypatch.setenv("RAG_ALLOWED_ORIGINS", "https://portfolio.test, http://localhost:5173")
    monkeypatch.setenv("RAG_INDEX_MANIFEST", "output/index.json")
    settings = Settings.from_env()
    assert settings.allowed_origins == ["https://portfolio.test", "http://localhost:5173"]
    assert settings.index_manifest.is_absolute()
    from pathlib import Path
    examples = json.loads((Path(__file__).resolve().parents[1] / "frontend/public/examples.json").read_text(encoding="utf-8"))
    assert len(examples) == 3
    for example in examples:
        serialized = json.dumps(example)
        assert not any(word in serialized for word in ["prompt", "qdrant_path", "LLM_API_KEY"])
        result = example["result"]
        assert set(result["citations"]) <= {source["source_id"] for source in result["sources"]}
        assert all(page >= 1 for source in result["sources"] for page in source["pdf_pages"])
