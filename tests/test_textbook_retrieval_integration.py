"""Pipeline với Qdrant thật trong bộ nhớ, embedder và model rerank giả."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "textbook_retrieval_integration", ROOT / "src/retrieval/textbook_retrieval.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeEmbedder:
    def embed_query(self, query):
        return np.array([1.0, 0.0])


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.pairs = []

    def tokenizer(self, question, passage, truncation):
        assert truncation is False
        return {"input_ids": [0] * (len(question.split()) + len(passage.split()) + 4)}

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        scores = {"database schema": 0.2, "transaction atomicity": 0.9,
                  "garden flower": 0.99}
        return np.array([scores[text] for _, text in pairs])


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(module, "CrossEncoder", FakeModel)
    client = QdrantClient(":memory:")
    client.create_collection("textbook", vectors_config=models.VectorParams(
        size=2, distance=models.Distance.COSINE,
    ))
    examples = [(1, "database schema", [1.0, 0.0]),
                (2, "transaction atomicity", [0.8, 0.6]),
                (3, "garden flower", [0.0, 1.0])]
    client.upsert("textbook", points=[
        models.PointStruct(id=i, vector=vector, payload={
            "page_content": text,
            "metadata": {"chapter_number": "1", "sources": [
                {"record_id": f"p{i}:b0", "pdf_page": i, "source_start_index": 0},
            ]},
        }) for i, text, vector in examples
    ])
    reranker = module.CrossEncoderReranker()
    retriever = module.TextbookRetriever(client, "textbook", FakeEmbedder(), reranker)
    yield retriever
    client.close()


def test_full_pipeline_reranks_only_fused_candidates_and_preserves_payload(pipeline):
    result = pipeline.retrieve("database", top_k=1, candidate_k=2)

    assert [h["point_id"] for h in result] == [2]
    assert result[0]["score"] == pytest.approx(0.9)
    assert result[0]["score_type"] == "reranker"
    stored = pipeline.client.retrieve("textbook", ids=[2])[0].payload
    assert result[0]["page_content"] == stored["page_content"]
    assert result[0]["metadata"] == stored["metadata"]
    assert pipeline.reranker.model.pairs == [
        ["database", "database schema"], ["database", "transaction atomicity"],
    ]
    # Retriever không đóng client hoặc thay đổi collection sau khi tìm kiếm.
    assert pipeline.client.count("textbook").count == 3


def test_missing_lexical_match_still_uses_dense_and_returns_available_hits(pipeline):
    result = pipeline.retrieve("unseenword", top_k=5, candidate_k=20)
    assert [hit["point_id"] for hit in result] == [3, 2, 1]
    assert len({hit["point_id"] for hit in result}) == 3


def test_empty_collection_skips_models(pipeline):
    pipeline.client.create_collection("empty", vectors_config=models.VectorParams(
        size=2, distance=models.Distance.COSINE,
    ))
    retriever = module.TextbookRetriever(pipeline.client, "empty", None, None)
    assert retriever.retrieve("database") == []


def test_evidence_metric_requires_same_record_and_complete_span():
    spec = importlib.util.spec_from_file_location("retrieval_evaluation", ROOT / "scripts/evaluate_retrieval.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    case = {"record_id": "p1:b0", "start": 10, "end": 20}
    for record_id, start, end, expected in [
        ("p1:b0", 10, 20, True), ("p1:b0", 0, 30, True),
        ("p2:b0", 0, 30, False), ("p1:b0", 11, 20, False), ("p1:b0", 10, 19, False),
    ]:
        hit = {"metadata": {"sources": [{"record_id": record_id,
                "source_start_index": start, "source_end_index": end}]}}
        assert evaluation.contains_evidence(hit, case) is expected
