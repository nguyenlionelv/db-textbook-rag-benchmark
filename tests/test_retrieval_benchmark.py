"""Check evidence metrics, bilingual labels and saved evaluation results."""

from copy import deepcopy
from collections import defaultdict
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import build_foundations_benchmark as builder
from scripts import evaluate_retrieval as evaluation


def hit(point_id, record_id, start=0, end=20):
    return {"point_id": point_id, "page_content": "database data", "score": 1.0,
            "score_type": "rrf", "metadata": {"sources": [{"record_id": record_id,
              "pdf_page": 1, "source_start_index": start, "source_end_index": end}]}}


@pytest.fixture
def case():
    return {"question_id": "q1_vi", "pair_id": "q1", "topic": "database", "family": "basic",
            "kind": "comparison", "language": "vi", "split": "dev", "text": "Question",
            "evidence_groups": [[{"record_id": "a", "start": 2, "end": 5, "text": "234"}],
                                [{"record_id": "b", "start": 4, "end": 7, "text": "456"}]]}


def test_any_evidence_is_distinct_from_complete_evidence(case):
    scores = evaluation.score_hits([hit(0, "other"), hit(1, "a")], case)
    assert scores == {"hit_at_5": 1, "mrr_at_5": 0.5,
                      "evidence_coverage_at_5": 0.5, "all_evidence_at_5": 0}
    scores = evaluation.score_hits([hit(1, "a"), hit(2, "b")], case)
    assert scores["all_evidence_at_5"] == scores["evidence_coverage_at_5"] == 1


def test_alternative_evidence_satisfies_one_group(case):
    case["evidence_groups"][0].append({"record_id": "alternative", "start": 0, "end": 5})
    scores = evaluation.score_hits([hit(1, "alternative"), hit(2, "b")], case)
    assert scores["all_evidence_at_5"] == 1


def test_metric_ignores_sixth_hit_and_rejects_partial_spans(case):
    misses = [hit(i, "other") for i in range(5)]
    assert evaluation.score_hits(misses + [hit(6, "a")], case)["hit_at_5"] == 0
    assert not evaluation.contains_evidence(hit(1, "a", start=3), case)
    assert not evaluation.contains_evidence(hit(1, "a", end=4), case)


def test_legacy_evidence_does_not_use_the_question_as_the_source_text():
    legacy = {"topic": "data", "language": "en", "split": "dev", "text": "What are data?",
              "record_id": "a", "start": 2, "end": 5, "evidence": "234"}
    evaluation.validate_cases([legacy], {0: hit(0, "a")}, [{"id": "a", "text": "0123456789"}])
    assert evaluation.score_hits([hit(0, "a")], legacy)["mrr_at_5"] == 1


def paired(case):
    english = deepcopy(case)
    english.update(question_id="q1_en", language="en", text="English question")
    return [case, english]


def test_source_evidence_matches_indexed_corpus(case):
    evaluation.validate_cases(paired(case), {0: hit(0, "a"), 1: hit(1, "b")},
                              [{"id": key, "text": "0123456789"} for key in ("a", "b")])


@pytest.mark.parametrize("problem", ["missing_source", "text", "offsets", "empty_group"])
def test_invalid_evidence_is_rejected(case, problem):
    cases = paired(case)
    chunks = {0: hit(0, "a"), 1: hit(1, "b")}
    if problem == "missing_source":
        chunks.pop(1)
    elif problem == "text":
        cases[0]["evidence_groups"][0][0]["text"] = "wrong"
    elif problem == "offsets":
        cases[0]["evidence_groups"][0][0]["start"] = -1
    else:
        cases[0]["evidence_groups"] = [[]]
    with pytest.raises(AssertionError):
        evaluation.validate_cases(cases, chunks, [{"id": key, "text": "0123456789"} for key in ("a", "b")])


def test_empty_benchmark_is_rejected():
    with pytest.raises(AssertionError, match="empty"):
        evaluation.validate_cases([], {}, [])


def test_evaluation_runs_once_per_question_and_exports_both_methods(case, tmp_path):
    class Reranker:
        def rerank(self, query, hits):
            return [hits[1], hits[0]]

    class Retriever:
        chunks = {0: {key: value for key, value in hit(0, "other").items() if key != "point_id"},
                  1: {key: value for key, value in hit(1, "a").items() if key != "point_id"}}
        reranker = Reranker()

        def _bm25_search(self, query, k):
            return [(0, 1), (1, 0.5)]

        def _dense_search(self, query, k):
            return [(0, 1), (1, 0.5)]

        def _rrf(self, lexical, dense, k):
            return lexical

        def retrieve(self, query, **kwargs):
            return self.reranker.rerank(query, evaluation.retrieve_candidates(self, query))

    result = evaluation.evaluate(Retriever(), paired(case), tmp_path)
    assert result["questions"] == 2 and result["results"] == 4
    frame = pd.read_csv(tmp_path / "per_query.csv")
    assert len(frame) == 4
    assert frame.loc[frame["method"] == "hybrid", "mrr_at_5"].eq(0.5).all()
    assert frame.loc[frame["method"] == "hybrid_rerank", "mrr_at_5"].eq(1.0).all()
    assert frame["all_evidence_at_5"].eq(0).all()
    assert len((tmp_path / "topk.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads((tmp_path / "progress.json").read_text())["completed_queries"] == 2


def test_token_preflight_rejects_oversized_candidates_without_inference(tmp_path, monkeypatch):
    reranker = SimpleNamespace(model=SimpleNamespace(tokenizer=lambda *args, **kwargs: {"input_ids": [0] * 513}),
                              max_length=512)
    retriever = SimpleNamespace(reranker=reranker)
    monkeypatch.setattr(evaluation, "retrieve_candidates", lambda *args: [hit(0, "a")])
    with pytest.raises(AssertionError, match="exceeds"):
        evaluation.preflight(retriever, [{"text": "Question"}], tmp_path)


def test_authored_benchmark_is_balanced_and_source_checked():
    root = builder.BENCHMARK
    cases = [json.loads(line) for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert len(cases) == 100 and len({case["pair_id"] for case in cases}) == 50
    assert sum(case["language"] == "vi" for case in cases) == 50
    assert sum(case["language"] == "en" for case in cases) == 50
    assert sum(case["split"] == "dev" for case in cases) == 60
    assert builder.sha256(root / "queries.jsonl") == manifest["queries_sha256"]
    pairs, families = defaultdict(list), defaultdict(set)
    for case in cases:
        pairs[case["pair_id"]].append(case)
        families[case["family"]].add(case["split"])
    for pair in pairs.values():
        assert len(pair) == 2 and {case["language"] for case in pair} == {"vi", "en"}
        assert pair[0]["split"] == pair[1]["split"]
        assert pair[0]["evidence_groups"] == pair[1]["evidence_groups"]
    assert all(len(splits) == 1 for splits in families.values())
