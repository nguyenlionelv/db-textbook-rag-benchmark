"""Unit test retrieval, không tải model hoặc mở index của notebook."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "textbook_retrieval_under_test", ROOT / "src/retrieval/textbook_retrieval.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeModel:
    def __init__(self, model_name, **kwargs):
        self.config = (model_name, kwargs)
        self.calls = []
        self.scores = [0.1, 0.9]

    def tokenizer(self, question, passage, truncation):
        assert truncation is False
        return {"input_ids": [0] * (len(question.split()) + len(passage.split()) + 4)}

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


@pytest.fixture
def reranker(monkeypatch):
    monkeypatch.setattr(module, "CrossEncoder", FakeModel)
    return module.CrossEncoderReranker(revision="test-revision")


@pytest.fixture
def hits():
    return [
        {"point_id": i, "page_content": text, "score": 0.03, "score_type": "rrf",
         "metadata": {"sources": [{"record_id": f"p{i}:b0", "pdf_page": i}]}}
        for i, text in [(1, "database schema"), (2, "database transaction")]
    ]


def test_reranker_scores_pairs_reorders_and_preserves_sources(reranker, hits):
    original = deepcopy(hits)
    result = reranker.rerank("database?", hits)

    assert [hit["point_id"] for hit in result] == [2, 1]
    assert [hit["score"] for hit in result] == [0.9, 0.1]
    assert all(hit["score_type"] == "reranker" for hit in result)
    assert result[0]["page_content"] == hits[1]["page_content"]
    assert result[0]["metadata"] == hits[1]["metadata"]
    assert hits == original
    pairs, options = reranker.model.calls[0]
    assert pairs == [["database?", hit["page_content"]] for hit in hits]
    assert options["batch_size"] == 2
    assert isinstance(options["activation_fn"], torch.nn.Identity)
    result[0]["metadata"]["sources"].clear()
    assert hits == original


def test_reranker_loads_once_and_empty_hits_skip_prediction(reranker, hits):
    model = reranker.model
    assert model.config == ("BAAI/bge-reranker-v2-m3", {
        "revision": "test-revision", "device": "cpu", "max_length": 512, "cache_folder": None,
    })
    assert reranker.rerank("database", []) == []
    assert model.calls == []
    reranker.rerank("database", hits)
    reranker.rerank("transaction", hits)
    assert reranker.model is model
    assert len(model.calls) == 2


@pytest.mark.parametrize("scores", [[0.1], [0.1, float("nan")], [0.1, float("inf")]])
def test_reranker_rejects_invalid_model_scores(reranker, hits, scores):
    reranker.model.scores = scores
    with pytest.raises(ValueError, match="mỗi chunk"):
        reranker.rerank("database", hits)


def test_reranker_handles_one_hit_and_breaks_ties_by_id(reranker, hits):
    reranker.model.scores = [0.5]
    assert len(reranker.rerank("database", hits[:1])) == 1
    reranker.model.scores = [0.5, 0.5]
    assert [h["point_id"] for h in reranker.rerank("database", hits[::-1])] == [1, 2]


@pytest.mark.parametrize("name", ["batch_size", "max_length"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_reranker_configuration_fails_before_model_load(monkeypatch, name, value):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Không được tải model khi cấu hình không hợp lệ")
    monkeypatch.setattr(module, "CrossEncoder", unexpected_load)
    with pytest.raises(ValueError, match=name):
        module.CrossEncoderReranker(**{name: value})


class FakeClient:
    def __init__(self):
        self.offsets = []

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=SimpleNamespace(size=2, distance="Cosine"),
        )))

    def scroll(self, **kwargs):
        self.offsets.append(kwargs["offset"])
        ids = [1, 2] if kwargs["offset"] is None else [3]
        texts = {1: "database schema", 2: "garden flower", 3: "biology animal"}
        points = [SimpleNamespace(id=i, payload={
            "page_content": texts[i], "metadata": {"sources": [{"record_id": f"p{i}:b0", "pdf_page": i}]},
        }) for i in ids]
        return points, (3 if kwargs["offset"] is None else None)

    def query_points(self, **kwargs):
        self.query = kwargs
        return SimpleNamespace(points=[SimpleNamespace(id=2, score=0.8)])


class FakeEmbedder:
    model_name = "fake-e5"
    revision = "test-revision"
    model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 2)

    def embed_query(self, query):
        self.query = query
        return np.array([1.0, 0.0])


@pytest.fixture
def retriever(reranker):
    return module.TextbookRetriever(FakeClient(), "textbook", FakeEmbedder(), reranker)


def test_reads_all_pages_once_and_bm25_matches_normalized_tokens(retriever):
    assert retriever.client.offsets == [None, 3]
    assert retriever._bm25_search("DATABASE!", 20)[0][0] == 1
    assert retriever._bm25_search("unseenword", 20) == []
    retriever._bm25_search("database", 20)
    assert retriever.client.offsets == [None, 3]


def test_dense_uses_query_embedding_and_selected_collection(retriever):
    assert retriever._dense_search("schema?", 7) == [(2, 0.8)]
    assert retriever.embedder.query == "schema?"
    assert retriever.client.query == {
        "collection_name": "textbook", "query": [1.0, 0.0],
        "limit": 7, "with_payload": False,
    }


def test_rrf_adds_reciprocal_ranks_and_merges_shared_ids(retriever):
    result = retriever._rrf([(1, 100), (2, 10)], [(2, 0.9), (3, 0.8)], 20)
    assert [i for i, _ in result] == [2, 1, 3]
    assert dict(result) == pytest.approx({2: 1 / 62 + 1 / 61, 1: 1 / 61, 3: 1 / 62})
    assert retriever._rrf([(1, 100)], [(2, 0.9)], 1) == [(1, 1 / 61)]
    assert retriever._rrf([], [], 20) == []


@pytest.mark.parametrize("query", [None, 123, "", " \n"])
def test_invalid_query(retriever, reranker, query):
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)
    with pytest.raises(ValueError, match="query"):
        reranker.rerank(query, [])


@pytest.mark.parametrize("name", ["top_k", "candidate_k"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits(retriever, name, value):
    with pytest.raises(ValueError, match=name):
        retriever.retrieve("database", **{name: value})


def test_candidate_count_must_cover_top_k(retriever):
    with pytest.raises(ValueError, match="candidate_k"):
        retriever.retrieve("database", top_k=5, candidate_k=2)


def test_pair_token_limit_counts_both_texts_and_special_tokens(reranker, hits):
    reranker.max_length = 7  # 1 query + 2 passage + 4 special tokens.
    reranker.rerank("database", hits)
    reranker.model.calls.clear()
    with pytest.raises(ValueError, match="8 token"):
        reranker.rerank("database schema", hits)
    assert reranker.model.calls == []


def test_ties_use_numeric_ids_before_uuid_strings(retriever, reranker, hits):
    assert [i for i, _ in retriever._rrf([(10, 1)], [(2, 1)], 2)] == [2, 10]
    hits[0]["point_id"], hits[1]["point_id"] = 10, 2
    reranker.model.scores = [0.5, 0.5]
    assert [h["point_id"] for h in reranker.rerank("database", hits)] == [2, 10]
    assert sorted(["uuid", 10, 2], key=module._id_key) == [2, 10, "uuid"]


@pytest.mark.parametrize("payload", [
    None, {}, {"page_content": " "}, {"page_content": "text", "metadata": {}},
    {"page_content": "text", "metadata": {"sources": []}},
    {"page_content": "text", "metadata": {"sources": [{"record_id": "p0", "pdf_page": -1}]}},
    {"page_content": "text", "metadata": {"sources": [{"pdf_page": 0}]}},
])
def test_bad_payload_fails_at_initialization(monkeypatch, reranker, payload):
    client = FakeClient()
    monkeypatch.setattr(client, "scroll", lambda **kw: ([SimpleNamespace(id=1, payload=payload)], None))
    with pytest.raises(ValueError, match="Point 1"):
        module.TextbookRetriever(client, "textbook", FakeEmbedder(), reranker)


def test_embedding_profile_is_checked_once(reranker):
    profile = {"collection": "textbook", "model": "fake-e5", "model_revision": "test-revision"}
    module.TextbookRetriever(FakeClient(), "textbook", FakeEmbedder(), reranker, profile)
    for key in profile:
        with pytest.raises(ValueError, match="embedding_profile"):
            module.TextbookRetriever(FakeClient(), "textbook", FakeEmbedder(), reranker,
                                     {**profile, key: "wrong"})


def test_embedding_dimension_mismatch_is_rejected(reranker):
    embedder = FakeEmbedder()
    embedder.model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 3)
    profile = {"collection": "textbook", "model": "fake-e5", "model_revision": "test-revision"}
    with pytest.raises(ValueError, match="số chiều"):
        module.TextbookRetriever(FakeClient(), "textbook", embedder, reranker, profile)


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf"), 1], [[1, 0]]])
def test_invalid_query_vectors_fail_before_search(retriever, monkeypatch, vector):
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda q: vector)
    with pytest.raises(ValueError, match="Vector query"):
        retriever._dense_search("database", 5)
    assert not hasattr(retriever.client, "query")
