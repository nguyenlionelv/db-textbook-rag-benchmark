"""Evaluate annotated retrieval questions against a verified textbook index.

Defaults preserve the pilot benchmark. Pass --index-manifest, --queries and
--output-dir for another corpus. Models are loaded from the local cache.
"""

import argparse
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_foundations_benchmark import contains_span, sha256

RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BASE = ROOT / "output/retrieval/structure_aware_recursive"
METRICS = ["hit_at_5", "mrr_at_5", "evidence_coverage_at_5", "all_evidence_at_5",
           "candidate_hit_at_20", "candidate_all_evidence_at_20", "seconds"]


def read_json(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    """Read non-empty JSONL records."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    """Write a complete JSON result."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evidence_groups(case):
    """Return AND groups containing OR alternatives; support pilot labels."""
    if "evidence_groups" in case:
        return case["evidence_groups"]
    span = {key: case[key] for key in ("record_id", "start", "end")}
    if "evidence" in case:
        span["text"] = case["evidence"]
    return [[span]]


def matched_groups(hits, case):
    """Check which annotated evidence groups appear in the retrieved hits."""
    return [any(contains_span(hit, span) for hit in hits for span in group)
            for group in evidence_groups(case)]


def contains_evidence(hit, case):
    """Return whether a hit contains any annotated evidence alternative."""
    return any(matched_groups([hit], case))


def score_hits(hits, case):
    """Score any evidence, its first rank, and all required evidence groups."""
    hits = hits[:5]
    ranks = [rank for rank, hit in enumerate(hits, 1) if contains_evidence(hit, case)]
    covered = matched_groups(hits, case)
    return {"hit_at_5": int(bool(ranks)), "mrr_at_5": 1 / ranks[0] if ranks else 0.0,
            "evidence_coverage_at_5": sum(covered) / len(covered),
            "all_evidence_at_5": int(all(covered))}


def validate_cases(cases, chunks, records):
    """Confirm that prepared labels still match the corpus being evaluated."""
    assert cases, "The benchmark is empty"
    by_id = {record["id"]: record for record in records}
    for case in cases:
        groups = evidence_groups(case)
        for group in groups:
            for span in group:
                text = by_id[span["record_id"]]["text"]
                start, end = span["start"], span["end"]
                assert 0 <= start < end <= len(text), "Invalid source offsets"
                if "text" in span:
                    assert text[start:end] == span["text"], "Source text mismatch"
        assert groups and all(matched_groups(chunks.values(), case)), "Evidence is outside the indexed corpus"


def retrieve_candidates(retriever, query):
    """Run the retriever's unchanged BM25, dense and RRF candidate steps."""
    lexical = retriever._bm25_search(query, 20)
    dense = retriever._dense_search(query, 20)
    candidates = retriever._rrf(lexical, dense, 20)
    return [{"point_id": point_id, **deepcopy(retriever.chunks[point_id]),
             "score": score, "score_type": "rrf"} for point_id, score in candidates]


def preflight(retriever, cases, output_dir):
    """Check actual query-candidate token counts without reranker inference."""
    rows, maximum = [], 0
    for case in cases:
        hits = retrieve_candidates(retriever, case["text"])
        counts = [len(retriever.reranker.model.tokenizer(case["text"], hit["page_content"],
                                                       truncation=False)["input_ids"]) for hit in hits]
        largest = max(counts, default=0)
        maximum = max(maximum, largest)
        rows.append({"question_id": case.get("question_id"), "candidate_ids": [hit["point_id"] for hit in hits],
                     "maximum_pair_tokens": largest})
    report = {"questions": len(cases), "maximum_pair_tokens": maximum,
              "limit": retriever.reranker.max_length, "cases": rows}
    write_json(output_dir / "token_preflight.json", report)
    assert maximum <= retriever.reranker.max_length, "Query-candidate length exceeds the reranker limit"
    return report


def run_case(retriever, case, index):
    """Retrieve once, rerank the candidates, and score both result lists."""
    started = perf_counter()
    before = retrieve_candidates(retriever, case["text"])
    hybrid_seconds = perf_counter() - started
    started = perf_counter()
    after = retriever.reranker.rerank(case["text"], before)[:5]
    rerank_seconds = perf_counter() - started

    covered = matched_groups(before, case)
    common = {"query_index": index, "question_id": case.get("question_id", f"q{index}"),
              "pair_id": case.get("pair_id", case["topic"]), "split": case["split"],
              "language": case["language"], "topic": case["topic"],
              "family": case.get("family", case["topic"]), "kind": case.get("kind", "single"),
              "evidence_groups": len(covered), "candidate_hit_at_20": int(any(covered)),
              "candidate_all_evidence_at_20": int(all(covered))}
    rows, rankings = [], []
    for method, hits, seconds, rerank_time in (
            ("hybrid", before[:5], hybrid_seconds, 0.0),
            ("hybrid_rerank", after, hybrid_seconds + rerank_seconds, rerank_seconds)):
        rows.append({**common, "method": method, **score_hits(hits, case), "seconds": seconds,
                     "hybrid_seconds": hybrid_seconds, "rerank_seconds": rerank_time})
        rankings.append({"question_id": common["question_id"], "query": case["text"],
                         "method": method, "hits": hits})
    return rows, rankings


def summarize(frame, columns):
    """Count questions and average metrics for the requested groups."""
    grouped = frame.groupby(columns)
    summary = grouped[METRICS].mean()
    summary.insert(0, "cases", grouped.size())
    return summary.reset_index()


def save_results(frame, question_count, chunk_count, output_dir):
    """Export summaries, evidence misses, and the evaluation report."""
    summary = summarize(frame, ["split", "language", "method"])
    summary.to_csv(output_dir / "summary.csv", index=False)
    overall = summarize(frame, ["language", "method"])
    overall.to_csv(output_dir / "overall.csv", index=False)
    misses = frame[(frame["method"] == "hybrid_rerank") & (frame["all_evidence_at_5"] == 0)]
    misses.to_csv(output_dir / "evidence_misses.csv", index=False)
    lines = ["# Retrieval: full corpus, annotated evidence", "",
             f"{question_count} questions; {chunk_count} searchable chunks; no chapter filter.",
             "Queries use annotated textbook evidence. Any-evidence Hit/MRR differ from complete evidence coverage.", "",
             "| Split | Language | Method | Cases | Hit@5 | MRR@5 | All evidence@5 | Coverage@5 | Candidate all@20 | Seconds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.split} | {row.language} | {row.method} | {row.cases} | {row.hit_at_5:.3f} | "
                     f"{row.mrr_at_5:.3f} | {row.all_evidence_at_5:.3f} | {row.evidence_coverage_at_5:.3f} | "
                     f"{row.candidate_all_evidence_at_20:.3f} | {row.seconds:.2f} |")
    lines += ["", "Evidence labels were fixed before retrieval and were not expanded after inspecting results.",
              "Annotations are not exhaustive relevance qrels; another valid passage can be scored as a miss.",
              "Paired translations share a split. Authoring and scoring use the coding agent, without independent human review.",
              "Topics overlap earlier project experiments; this is not a wholly unseen final benchmark.",
              "Timing includes live hybrid retrieval plus reranking, with candidates shared across methods.",
              "Model/index setup, token preflight and one warm-up are excluded. One pass; no p95 claim.",
              "This run evaluates retrieval only; no LLM answers or API calls.",
              "", "Files: queries.jsonl, frozen_config.json, per_query.csv, topk.jsonl, summary.csv, overall.csv, evidence_misses.csv."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(retriever, cases, output_dir):
    """Run live retrieval, keeping completed questions on disk.

    Total reranked time is live candidate retrieval plus reranker inference.
    Setup, token preflight and warm-up are excluded from per-query times.
    """
    ordered = [case for split in ("dev", "eval") for case in cases if case["split"] == split]
    started = perf_counter()
    retriever.retrieve(ordered[0]["text"], top_k=5, candidate_k=40)
    warm_seconds = perf_counter() - started

    rows = []
    for index, case in enumerate(ordered):
        case_rows, rankings = run_case(retriever, case, index)
        rows.extend(case_rows)
        with (output_dir / "topk.jsonl").open("a", encoding="utf-8") as stream:
            for ranking in rankings:
                stream.write(json.dumps(ranking, ensure_ascii=False) + "\n")
        pd.DataFrame(rows).to_csv(output_dir / "per_query.csv", index=False)
        write_json(output_dir / "progress.json", {"status": "running", "completed_queries": index + 1,
                   "total_queries": len(ordered), "last_question_id": case_rows[0]["question_id"]})
        hybrid, reranked = case_rows
        print(f"{index + 1}/{len(ordered)} {hybrid['question_id']} {case['split']} "
              f"hybrid={hybrid['hybrid_seconds']:.2f}s rerank={reranked['rerank_seconds']:.2f}s", flush=True)

    save_results(pd.DataFrame(rows), len(cases), len(retriever.chunks), output_dir)
    return {"results": len(rows), "questions": len(cases), "warmup_seconds": warm_seconds}


def main():
    """Select inputs, verify corpus/labels, freeze configuration, then evaluate."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-manifest", type=Path, default=BASE / "manifest.json")
    parser.add_argument("--queries", type=Path, default=BASE / "queries.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--e5-cache", type=Path, default=ROOT / "output/model_cache/e5_hub")
    parser.add_argument("--reranker-cache", type=Path, default=ROOT / "output/model_cache/reranker_hub")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from qdrant_client import QdrantClient
    from src.ingestion.textbook_embedder import E5Embedder
    from src.retrieval.textbook_retrieval import CrossEncoderReranker, TextbookRetriever

    torch.set_num_threads(4)
    started = perf_counter()
    profile = read_json(args.index_manifest)
    chunk_path = Path(profile.get("chunk_path", ROOT / "output/chunks/structure_aware_recursive.jsonl"))
    parser_path = Path(profile.get("parser_path", ROOT / "output/parsed_chapter1.jsonl"))
    db_path = Path(profile.get("qdrant_path", ROOT / "output/qdrant_db"))
    assert sha256(chunk_path) == profile["chunk_sha256"], "Chunk input hash mismatch"
    assert sha256(parser_path) == profile["parser_sha256"], "Parsed input hash mismatch"
    cases = read_jsonl(args.queries)
    chunks = read_jsonl(chunk_path)
    expected = {i: row for i, row in enumerate(chunks) if row["metadata"]["sources"]}
    records = read_jsonl(parser_path)
    validate_cases(cases, expected, records)
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or ROOT / "output/retrieval/runs" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "queries.jsonl").write_bytes(args.queries.read_bytes())
    benchmark_path = args.queries.with_name("manifest.json")
    benchmark = read_json(benchmark_path)
    query_hash = sha256(args.queries)
    assert query_hash == benchmark.get("queries_sha256", query_hash), "Benchmark hash mismatch"
    assert all(benchmark[key] == profile[key] for key in ("chunk_sha256", "parser_sha256")), "Benchmark corpus mismatch"
    configuration = {"index_manifest": str(args.index_manifest.resolve()), "index_manifest_sha256": sha256(args.index_manifest),
                     "collection": profile["collection"], "indexed_chunks": len(expected),
                     "chunk_sha256": profile["chunk_sha256"], "parser_sha256": profile["parser_sha256"],
                     "queries_sha256": query_hash, "benchmark_manifest_sha256": sha256(benchmark_path),
                     "model": profile["model"], "model_revision": profile["model_revision"],
                     "reranker": "BAAI/bge-reranker-v2-m3", "reranker_revision": RERANKER_REVISION,
                     "top_k": 5, "candidate_k": 20, "rrf_constant": 60,
                     "reranker_max_length": args.reranker_max_length, "reranker_batch_size": 2,
                     "torch_threads": 4, "device": "cpu", "search_filter": None,
                     "methods": ["hybrid", "hybrid_rerank"],
                     "code_sha256": {name: sha256(ROOT / name) for name in
                                    ("scripts/evaluate_retrieval.py", "scripts/build_foundations_benchmark.py",
                                     "src/retrieval/textbook_retrieval.py",
                                     "src/ingestion/textbook_embedder.py")},
                     "packages": {name: version(name) for name in ("torch", "sentence-transformers", "qdrant-client",
                                                                  "rank-bm25", "transformers", "pandas")}}
    write_json(output_dir / "frozen_config.json", configuration)
    write_json(output_dir / "benchmark_manifest.json", benchmark)
    manifest = {"status": "in_progress", "configuration": configuration}
    write_json(output_dir / "manifest.json", manifest)
    print(f"Validated {len(cases)} questions / {len(expected)} searchable chunks. Output: {output_dir}", flush=True)
    try:
        embedder = E5Embedder(model_name=profile["model"], revision=profile["model_revision"],
                              cache_folder=str(args.e5_cache), local_files_only=True)
        reranker = CrossEncoderReranker(revision=RERANKER_REVISION,
                                        cache_folder=str(args.reranker_cache), max_length=args.reranker_max_length)
        with closing(QdrantClient(path=str(db_path))) as client:
            retriever = TextbookRetriever(client, profile["collection"], embedder, reranker, profile)
            assert retriever.chunks == expected, "Indexed payloads do not match the prepared corpus"
            token_check = preflight(retriever, cases, output_dir)
            print(f"Token preflight passed: maximum pair length {token_check['maximum_pair_tokens']}.", flush=True)
            setup_seconds = perf_counter() - started
            result = evaluate(retriever, cases, output_dir)
        assert sha256(args.queries) == query_hash, "Benchmark changed during evaluation"
        assert all(sha256(ROOT / name) == digest for name, digest in configuration["code_sha256"].items()), "Code changed during evaluation"
        artifacts = ["queries.jsonl", "frozen_config.json", "benchmark_manifest.json", "token_preflight.json",
                     "per_query.csv", "topk.jsonl", "summary.csv", "overall.csv", "evidence_misses.csv", "report.md"]
        manifest.update(status="complete", **result, setup_and_preflight_seconds=setup_seconds,
                        wall_seconds=perf_counter() - started,
                        artifacts={name: sha256(output_dir / name) for name in artifacts})
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "progress.json", {"status": "complete", "completed_queries": len(cases),
                                                 "total_queries": len(cases)})
        print(f"Complete: {result['questions']} questions, {result['results']} method results.", flush=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
