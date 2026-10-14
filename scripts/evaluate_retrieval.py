"""So sánh Hybrid và Hybrid + reranker trên 32 câu hỏi đã có.

Chạy từ thư mục dự án: python scripts/evaluate_retrieval.py
Lần đầu cần tải model; kết quả lưu riêng, không ghi đè thực nghiệm notebook 05.
"""

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingestion.textbook_embedder import E5Embedder
from src.retrieval.textbook_retrieval import CrossEncoderReranker, TextbookRetriever

RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BASE = ROOT / "output/retrieval/structure_aware_recursive"
OUTPUT = ROOT / "output/retrieval/hybrid_reranking"


def contains_evidence(hit, case):
    return any(
        source["record_id"] == case["record_id"]
        and source["source_start_index"] <= case["start"]
        and source["source_end_index"] >= case["end"]
        for source in hit["metadata"]["sources"]
    )


def evaluate(retriever, profile):
    cases = [json.loads(line) for line in (BASE / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = retriever.chunks
    rows, rankings = [], []
    # Warm-up một lượt; thời gian dưới đây không gồm load model/index.
    retriever.retrieve(cases[0]["text"])
    for index, case in enumerate(cases):
        assert any(contains_evidence(chunk, case) for chunk in chunks.values()), "Gold ngoài corpus"
        start = perf_counter()
        lexical = retriever._bm25_search(case["text"], 20)
        dense = retriever._dense_search(case["text"], 20)
        candidates = retriever._rrf(lexical, dense, 20)
        before = [{"point_id": i, **chunks[i], "score": score, "score_type": "rrf"}
                  for i, score in candidates]
        hybrid_seconds = perf_counter() - start
        start = perf_counter()
        after = retriever.retrieve(case["text"], top_k=5, candidate_k=20)
        reranked_seconds = perf_counter() - start
        assert {h["point_id"] for h in after} <= {i for i, _ in candidates}
        for method, hits, seconds in (("hybrid", before[:5], hybrid_seconds),
                                      ("hybrid_rerank", after, reranked_seconds)):
            ranks = [rank for rank, hit in enumerate(hits, 1) if contains_evidence(hit, case)]
            rows.append({"query_index": index, "split": case["split"], "language": case["language"],
                         "method": method, "hit_at_5": int(bool(ranks)),
                         "mrr_at_5": 1 / ranks[0] if ranks else 0,
                         "candidate_hit_at_20": int(any(contains_evidence(h, case) for h in before)),
                         "seconds": seconds})
            rankings.append({"query_index": index, "query": case["text"], "method": method, "hits": hits})
        print(f"{index + 1}/{len(cases)} {case['language']} {case['topic']}: {reranked_seconds:.2f}s", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_query = pd.DataFrame(rows)
    summary = per_query.groupby(["split", "language", "method"], as_index=False)[
        ["hit_at_5", "mrr_at_5", "candidate_hit_at_20", "seconds"]
    ].mean()
    per_query.to_csv(OUTPUT / "per_query.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "topk.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rankings) + "\n", encoding="utf-8",
    )
    manifest = {**profile, "reranker": "BAAI/bge-reranker-v2-m3", "reranker_revision": RERANKER_REVISION,
                "methods": ["hybrid", "hybrid_rerank"], "reranker_max_length": retriever.reranker.max_length,
                "device": "cpu", "torch_num_threads": torch.get_num_threads(),
                "candidate_k": 20, "top_k": 5, "reranker_batch_size": retriever.reranker.batch_size,
                "timing": "1 warm-up, 1 pass/query/method; includes query encoding, excludes loading",
                "queries_sha256": hashlib.sha256((BASE / "queries.jsonl").read_bytes()).hexdigest()}
    # Không tái sử dụng thời gian load hoặc phiên bản thư viện từ thực nghiệm cũ.
    manifest.pop("load_seconds", None)
    from importlib.metadata import version
    manifest["packages"] = {name: version(name) for name in profile["packages"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Hybrid trước và sau reranking", "", "32 câu hỏi cố định: 16 dev, 16 eval; Anh và Việt.",
             "Hit@5 và MRR@5 dựa trên khoảng evidence đã gán, không phải Recall đầy đủ.", "",
             "| Split | Ngôn ngữ | Phương pháp | Hit@5 | MRR@5 | Candidate Hit@20 | Giây/query |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.split} | {row.language} | {row.method} | {row.hit_at_5:.3f} | "
                     f"{row.mrr_at_5:.3f} | {row.candidate_hit_at_20:.3f} | {row.seconds:.2f} |")
    comparison = per_query.pivot(index="query_index", columns="method", values="mrr_at_5")
    delta = comparison["hybrid_rerank"] - comparison["hybrid"]
    lines += ["", f"Trên 32 câu: {int((delta > 0).sum())} câu tăng MRR@5, "
              f"{int((delta < 0).sum())} câu giảm, {int((delta == 0).sum())} câu giữ nguyên.", ""]
    examples = list(delta[delta > 0].nlargest(2).index) + list(delta[delta < 0].nsmallest(2).index)
    for index in examples:
        old, new = comparison.loc[index, "hybrid"], comparison.loc[index, "hybrid_rerank"]
        lines.append(f"- {cases[index]['text']} — MRR@5: {old:.3f} → {new:.3f}.")
    lines += ["", "Thời gian là trung bình một lượt sau warm-up trên máy hiện tại, không phải benchmark p95.",
              "Reranker chỉ sắp xếp 20 ứng viên; không tìm lại bằng chứng ngoài tập ứng viên.",
              "Xem per_query.csv và topk.jsonl để đối chiếu từng câu và nguồn."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def main():
    torch.set_num_threads(4)
    profile = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    chunk_path = ROOT / "output/chunks/structure_aware_recursive.jsonl"
    assert hashlib.sha256(chunk_path.read_bytes()).hexdigest() == profile["chunk_sha256"]
    parser_path = ROOT / "output/parsed_chapter1.jsonl"
    assert hashlib.sha256(parser_path.read_bytes()).hexdigest() == profile["parser_sha256"]
    embedder = E5Embedder(model_name=profile["model"], revision=profile["model_revision"])
    reranker = CrossEncoderReranker(revision=RERANKER_REVISION,
                                   cache_folder=str(ROOT / "output/model_cache/reranker_hub"))
    client = QdrantClient(path=str(ROOT / "output/qdrant_db"))
    try:
        retriever = TextbookRetriever(client, profile["collection"], embedder, reranker, profile)
        rows = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {i: row for i, row in enumerate(rows) if row["metadata"]["sources"]}
        assert retriever.chunks == expected, "Payload trong index không khớp corpus"
        evaluate(retriever, profile)
    finally:
        client.close()


if __name__ == "__main__":
    main()
