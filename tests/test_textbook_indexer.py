"""Indexing với Qdrant thật trên đĩa/bộ nhớ và embedding giả, không tải model."""

import hashlib
import importlib.util
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("textbook_indexer", ROOT / "src/ingestion/textbook_indexer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_qdrant_index = module.build_qdrant_index


class FakeEmbedder:
    model_name = "intfloat/multilingual-e5-small"
    revision = "test-revision"

    def __init__(self):
        self.model = SimpleNamespace(
            get_sentence_embedding_dimension=lambda: 3, max_seq_length=20,
            tokenizer=lambda text, truncation: {"input_ids": text.split() + ["start", "end"]},
        )
        self.calls = []

    def embed_documents(self, chunks):
        self.calls.append([chunk.page_content for chunk in chunks])
        vectors = np.array([[len(chunk.page_content), 1, 2] for chunk in chunks], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def inputs(tmp_path):
    rows = [{"page_content": text, "metadata": {"chapter_number": "1", "sources": sources}}
            for text, sources in [
                ("database schema", [{"record_id": "book:p34:b1", "pdf_page": 34,
                                      "metadata": {"footnote_refs": [{"marker": "1"}]}}]),
                ("SECTION heading", []),
                ("transaction atomicity", [{"record_id": "book:p35:b1", "pdf_page": 35}]),
            ]]
    chunk_path = tmp_path / "chunks.jsonl"
    chunk_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return chunk_path, tmp_path / "index.json", rows


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    yield client
    client.close()


def notebook_collection(embedder, path):
    key = embedder.model_name + embedder.revision + path.read_text(encoding="utf-8")
    return "e5_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def test_round_trip_persistence_repeat_and_retriever_profile(tmp_path, inputs):
    chunk_path, manifest_path, rows = inputs
    original = chunk_path.read_bytes()
    parser_path = tmp_path / "parser.jsonl"
    parser_path.write_text("parser fixture", encoding="utf-8")
    embedder = FakeEmbedder()
    with closing(QdrantClient(path=str(tmp_path / "qdrant"))) as client:
        profile = build_qdrant_index(client, embedder, chunk_path, manifest_path,
                                     batch_size=1, parser_path=parser_path)
        assert profile["collection"] == notebook_collection(embedder, chunk_path)
        assert profile["status"] == "complete"
        assert profile["qdrant_path"] == str((tmp_path / "qdrant").resolve())
        assert profile["input_chunks"] == 3 and profile["chunks"] == 2
        assert profile["skipped_without_sources"] == 1 and profile["upserted"] == 2
        assert profile["parser_sha256"] == hashlib.sha256(parser_path.read_bytes()).hexdigest()
        assert embedder.calls == [[rows[0]["page_content"]], [rows[2]["page_content"]]]
        saved = client.retrieve(profile["collection"], ids=[0, 2], with_vectors=True)
        vectors = {point.id: point.vector for point in saved}
    with closing(QdrantClient(path=str(tmp_path / "qdrant"))) as client:
        repeated = build_qdrant_index(client, embedder, chunk_path, manifest_path, batch_size=2)
        assert repeated["collection"] == profile["collection"]
        assert client.count(profile["collection"]).count == 2
        saved = client.retrieve(profile["collection"], ids=[0, 2], with_vectors=True)
        assert {point.id: point.payload for point in saved} == {0: rows[0], 2: rows[2]}
        for point in saved:
            np.testing.assert_allclose(point.vector, vectors[point.id], atol=1e-6)
        # Manifest này dùng trực tiếp được với retriever đã có.
        spec = importlib.util.spec_from_file_location("index_retriever", ROOT / "src/retrieval/textbook_retrieval.py")
        retrieval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retrieval)
        retriever = retrieval.TextbookRetriever(client, profile["collection"], embedder, None, repeated)
        assert retriever.chunks == {0: rows[0], 2: rows[2]}
        assert client.query_points(profile["collection"], query=vectors[0], limit=1).points[0].id == 0
    assert chunk_path.read_bytes() == original
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == repeated


def test_new_input_or_revision_keeps_old_collection(client, inputs):
    chunk_path, manifest_path, rows = inputs
    embedder = FakeEmbedder()
    first = build_qdrant_index(client, embedder, chunk_path, manifest_path)
    rows[0]["page_content"] = "new content"
    chunk_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    second = build_qdrant_index(client, embedder, chunk_path, manifest_path)
    embedder.revision = "another-revision"
    third = build_qdrant_index(client, embedder, chunk_path, manifest_path)
    assert len({p["collection"] for p in (first, second, third)}) == 3
    assert len(client.get_collections().collections) == 3


@pytest.mark.parametrize("row", [
    {}, {"page_content": "", "metadata": {"sources": []}},
    {"page_content": "text", "metadata": {"sources": None}},
    {"page_content": "text", "metadata": {"sources": []}},
    {"page_content": "text", "metadata": {"sources": [{"record_id": "x", "pdf_page": True}]}},
])
def test_bad_input_fails_before_writing(client, inputs, row):
    chunk_path, manifest_path, _ = inputs
    chunk_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        build_qdrant_index(client, FakeEmbedder(), chunk_path, manifest_path)
    assert not manifest_path.exists()
    assert not client.get_collections().collections


def test_oversized_last_chunk_is_detected_before_first_write(client, inputs):
    chunk_path, manifest_path, rows = inputs
    rows[2]["page_content"] = "transaction atomicity explanation"
    chunk_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    embedder = FakeEmbedder()
    embedder.model.max_seq_length = 5
    with pytest.raises(ValueError, match="token"):
        build_qdrant_index(client, embedder, chunk_path, manifest_path, batch_size=1)
    assert not embedder.calls and not manifest_path.exists()
    assert not client.get_collections().collections


def test_crlf_input_keeps_notebook_collection_name(client, inputs):
    chunk_path, manifest_path, _ = inputs
    chunk_path.write_bytes(chunk_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    embedder = FakeEmbedder()
    profile = build_qdrant_index(client, embedder, chunk_path, manifest_path)
    assert profile["collection"] == notebook_collection(embedder, chunk_path)
    assert profile["chunk_sha256"] == hashlib.sha256(chunk_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("vectors", [
    [[1, 0]], [[0, 0, 0]], [[float("nan"), 0, 1]], [[2, 0, 0]],
])
def test_invalid_vectors_mark_failed_without_upsert(client, inputs, vectors):
    chunk_path, manifest_path, _ = inputs
    embedder = FakeEmbedder()
    embedder.embed_documents = lambda chunks: vectors
    with pytest.raises(ValueError, match="vector"):
        build_qdrant_index(client, embedder, chunk_path, manifest_path, batch_size=1)
    profile = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert profile["status"] == "failed" and profile["upserted"] == 0
    assert client.count(profile["collection"]).count == 0


def test_partial_failure_can_be_retried_without_duplicates(client, inputs, monkeypatch):
    chunk_path, manifest_path, _ = inputs
    embedder = FakeEmbedder()
    original = embedder.embed_documents

    def fail_second(chunks):
        if embedder.calls:
            raise RuntimeError("embedding interrupted")
        return original(chunks)

    monkeypatch.setattr(embedder, "embed_documents", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_qdrant_index(client, embedder, chunk_path, manifest_path, batch_size=1)
    failed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["upserted"] == 1
    assert client.count(failed["collection"]).count == 1
    monkeypatch.setattr(embedder, "embed_documents", original)
    complete = build_qdrant_index(client, embedder, chunk_path, manifest_path, batch_size=1)
    assert complete["status"] == "complete"
    assert client.count(complete["collection"]).count == 2


@pytest.mark.parametrize("problem", ["dimension", "distance", "extra_id"])
def test_existing_incompatible_collection_is_not_modified(client, inputs, problem):
    chunk_path, manifest_path, _ = inputs
    embedder = FakeEmbedder()
    collection = notebook_collection(embedder, chunk_path)
    client.create_collection(collection, vectors_config=models.VectorParams(
        size=2 if problem == "dimension" else 3,
        distance=models.Distance.DOT if problem == "distance" else models.Distance.COSINE,
    ))
    if problem == "extra_id":
        client.upsert(collection, points=[models.PointStruct(id=999, vector=[1, 0, 0], payload={"keep": True})])
    with pytest.raises(ValueError):
        build_qdrant_index(client, embedder, chunk_path, manifest_path)
    assert not embedder.calls and not manifest_path.exists()
    assert client.count(collection).count == (1 if problem == "extra_id" else 0)


def test_manifest_cannot_overwrite_chunk_or_parser(client, inputs):
    chunk_path, _, _ = inputs
    original = chunk_path.read_bytes()
    with pytest.raises(ValueError, match="ghi đè"):
        build_qdrant_index(client, FakeEmbedder(), chunk_path, chunk_path)
    assert chunk_path.read_bytes() == original


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size(client, inputs, batch_size):
    chunk_path, manifest_path, _ = inputs
    with pytest.raises(ValueError, match="batch_size"):
        build_qdrant_index(client, FakeEmbedder(), chunk_path, manifest_path, batch_size=batch_size)


def test_revision_must_be_fixed(client, inputs):
    chunk_path, manifest_path, _ = inputs
    embedder = FakeEmbedder()
    embedder.revision = None
    with pytest.raises(ValueError, match="revision"):
        build_qdrant_index(client, embedder, chunk_path, manifest_path)
