"""Ingest textbook chapters and appendices into a separate E5/Qdrant index.

Run from the repository root with the .venv Python interpreter. The PDF and
pinned E5 model must be available locally. Use --stage prepare to inspect
coverage, then --stage index --output-dir <same directory> to build the index.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.textbook_parser import TextbookParser
from src.ingestion.textbook_chunker import StructureAwareRecursiveChunker

import torch
from qdrant_client import QdrantClient
from src.ingestion.textbook_embedder import E5Embedder
from src.ingestion.textbook_indexer import build_qdrant_index

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CODE_PATHS = [
    Path(__file__),
    ROOT / "src/ingestion/textbook_parser.py",
    ROOT / "src/ingestion/textbook_chunker.py",
    ROOT / "src/ingestion/textbook_embedder.py",
    ROOT / "src/ingestion/textbook_indexer.py",
]


def sha256(path):
    """Hash a file without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Write a UTF-8 JSON result."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def is_blank_page(text):
    """Recognize the textbook's intentional blank-page label."""
    return " ".join(text.lower().split()) == "this page intentionally left blank"


def content_ranges(outline, page_count):
    """Resolve chapter/appendix bounds using the next sibling or parent entry.

    Outline pages are one-based; returned start_page/end_page are zero-based
    and inclusive. Part dividers, front matter, bibliography and index are
    outside the selected ranges.
    """
    ranges = []
    for index, (level, title, page) in enumerate(outline):
        match = re.match(r"^(chapter|appendix)\s+(\d+|[A-Z])\s+(.+)$", title, re.I)
        if not match:
            continue

        end_page = page_count - 1
        for next_level, _, next_page in outline[index + 1:]:
            if next_level <= level:
                end_page = next_page - 2
                break
        if not 1 <= page <= end_page + 1 <= page_count:
            raise ValueError(f"Invalid outline range: {title}")
        ranges.append({
            "number": match[2].upper(), "title": match[3].strip(),
            "kind": match[1].lower(), "start_page": page - 1, "end_page": end_page,
        })

    numbers = [int(item["number"]) for item in ranges if item["kind"] == "chapter"]
    unique = len({item["number"] for item in ranges}) == len(ranges)
    ordered = all(left["end_page"] < right["start_page"] for left, right in zip(ranges, ranges[1:]))
    if not numbers or numbers != list(range(1, len(numbers) + 1)) or not unique or not ordered:
        raise ValueError("Expected distinct, ordered chapters from 1 and non-overlapping appendices")
    return ranges


def validate_sources(records, chunks, *, include_footnotes):
    """Check source offsets and coverage of every non-whitespace character."""
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Parsed record IDs are not unique")

    spans = defaultdict(list)
    source_count = 0
    for chunk in chunks:
        for source in chunk.metadata["sources"]:
            record = by_id[source["record_id"]]
            start, end = source["source_start_index"], source["source_end_index"]
            left, right = source["chunk_start_index"], source["chunk_end_index"]
            if (
                not 0 <= start < end <= len(record["text"])
                or not 0 <= left < right <= len(chunk.page_content)
                or record["text"][start:end] != chunk.page_content[left:right]
                or record["pdf_page"] != source["pdf_page"]
                or record["chapter_number"] != chunk.metadata["chapter_number"]
            ):
                raise ValueError(f"Source mismatch: {record['id']}")
            spans[record["id"]].append((start, end))
            source_count += 1

    expected = [
        record for record in records
        if record["text"].strip() and (include_footnotes or record["type"] != "footnote")
    ]
    missing = []
    for record in expected:
        cursor = 0
        gaps = []
        for start, end in sorted(spans[record["id"]]):
            if start > cursor:
                gaps.append(record["text"][cursor:start])
            cursor = max(cursor, end)
        gaps.append(record["text"][cursor:])
        if any(gap.strip() for gap in gaps):
            missing.append(record["id"])
    if missing:
        raise ValueError(f"Chunking lost source text in {len(missing)} records: {missing[:5]}")

    return {
        "checked_source_spans": source_count, "expected_records": len(expected),
        "covered_records": len(spans), "missing_non_whitespace_records": 0,
    }


def page_coverage(doc, records, ranges):
    """Count extracted records and inspect each PDF page."""
    chapters = {
        page: item["number"] for item in ranges
        for page in range(item["start_page"], item["end_page"] + 1)
    }
    counts = Counter(record["pdf_page"] for record in records)
    rows = []
    for page, pdf_page in enumerate(doc):
        text = pdf_page.get_text().strip()
        rows.append({
            "pdf_page": page + 1, "selected": page in chapters,
            "chapter_number": chapters.get(page, ""), "extracted_records": counts[page],
            "raw_text_characters": len(text), "intentionally_blank": is_blank_page(text),
            "images": len(pdf_page.get_images()),
        })
    return rows


def chapter_coverage(records, chunks, ranges):
    """Summarize records, chunks and sections by chapter or appendix."""
    counts = Counter(chunk.metadata["chapter_number"] for chunk in chunks)
    rows = []
    for item in ranges:
        group = [record for record in records if record["chapter_number"] == item["number"]]
        if not counts[item["number"]]:
            raise ValueError(f"No sourced chunks for {item['number']}")
        rows.append({
            **item, "first_pdf_page": item["start_page"] + 1,
            "last_pdf_page": item["end_page"] + 1, "records": len(group),
            "chunks": counts[item["number"]],
            "content_types": dict(Counter(record["type"] for record in group)),
            "sections": len({record["section_number"] for record in group if record["section_number"]}),
        })
    return rows


def prepare_ingestion(
    pdf_path, output_dir, *, chunk_size=700, chunk_overlap=100, include_footnotes=True,
):
    """Parse outline ranges, chunk their text, and save coverage before indexing."""
    started = perf_counter()
    output_dir = Path(output_dir)
    pdf_path = Path(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    blank_records = 0

    with closing(TextbookParser(pdf_path)) as parser:
        outline, page_count = parser.doc.get_toc(), len(parser.doc)
        ranges = content_ranges(outline, page_count)
        for item in ranges:
            chapter = {key: item[key] for key in ("number", "title")}
            group = parser.parse_pages(item["start_page"], item["end_page"], chapter=chapter)
            selected = [record for record in group if not is_blank_page(record["text"])]
            blank_records += len(group) - len(selected)
            records.extend(selected)
            print(f"Parsed {item['kind']} {item['number']}: {len(selected)} records", flush=True)

        parser.export_jsonl(records, output_dir / "parsed.jsonl")
        page_rows = page_coverage(parser.doc, records, ranges)

    chunker = StructureAwareRecursiveChunker(chunk_size, chunk_overlap, include_footnotes)
    all_chunks = chunker.chunk(records)
    chunks = [chunk for chunk in all_chunks if chunk.metadata["sources"]]
    source_checks = validate_sources(records, chunks, include_footnotes=include_footnotes)
    chapter_rows = chapter_coverage(records, chunks, ranges)

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            row = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    chunks_per_page = Counter(
        page for chunk in chunks
        for page in {source["pdf_page"] for source in chunk.metadata["sources"]}
    )
    for row in page_rows:
        row["chunks"] = chunks_per_page[row["pdf_page"] - 1]
    with (output_dir / "page_coverage.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(page_rows[0]))
        writer.writeheader()
        writer.writerows(page_rows)

    references = [
        {"record_id": record["id"], "pdf_page": record["pdf_page"] + 1, **ref}
        for record in records for ref in record["metadata"].get("footnote_refs", [])
    ]
    coverage = {
        "pdf_pages": page_count, "selected_pages": sum(row["selected"] for row in page_rows),
        "excluded_pdf_pages": [row["pdf_page"] for row in page_rows if not row["selected"]],
        "selected_pages_without_records": [
            row["pdf_page"] for row in page_rows if row["selected"] and not row["extracted_records"]
        ],
        "records": len(records), "chunks": len(chunks),
        "excluded_blank_page_records": blank_records,
        "skipped_chunks_without_sources": len(all_chunks) - len(chunks),
        "duplicate_chunk_texts": len(chunks) - len({chunk.page_content for chunk in chunks}),
        "footnote_reference_status": dict(Counter(ref["status"] for ref in references)),
        "unresolved_footnotes": [ref for ref in references if ref["status"] != "resolved"],
        "source_checks": source_checks, "chapters": chapter_rows,
        "scope": (
            "Extracted chapter and appendix text, including captions and table-like text. "
            "Front matter, part dividers, bibliography, index and intentional blank-page labels are excluded. "
            "No OCR or interpretation of diagrams/table structure."
        ),
    }
    write_json(output_dir / "coverage.json", coverage)
    manifest = {
        "status": "prepared", "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "pdf_path": str(pdf_path.resolve()), "pdf_sha256": sha256(pdf_path),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in CODE_PATHS},
        "chunking": {
            "strategy": chunker.strategy, "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap, "include_footnotes": include_footnotes,
        },
        "artifacts": {
            name: sha256(output_dir / name)
            for name in ("parsed.jsonl", "chunks.jsonl", "coverage.json", "page_coverage.csv")
        },
        "records": len(records), "chunks": len(chunks),
        "prepare_seconds": perf_counter() - started,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"Prepared {len(records)} records / {len(chunks)} chunks "
        f"in {manifest['prepare_seconds']:.1f}s", flush=True,
    )
    return manifest


def index_ingestion(output_dir, model_cache, *, batch_size=8):
    """Index a verified preparation with the pinned local E5 model."""

    started = perf_counter()
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    files = {output_dir / name: digest for name, digest in manifest["artifacts"].items()}
    files.update({ROOT / name: digest for name, digest in manifest["code_sha256"].items()})
    files[Path(manifest["pdf_path"])] = manifest["pdf_sha256"]
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f"Prepared artifact changed: {path}; prepare a new run")

    torch.set_num_threads(4)
    embedder = E5Embedder(
        MODEL, revision=REVISION, batch_size=batch_size,
        cache_folder=str(model_cache), local_files_only=True,
    )
    with (output_dir / "chunks.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]

    token_counts = [
        len(embedder.model.tokenizer("passage: " + row["page_content"], truncation=False)["input_ids"])
        for row in rows
    ]
    write_json(output_dir / "token_validation.json", {
        "model": MODEL, "revision": REVISION,
        "limit": embedder.model.max_seq_length, "maximum": max(token_counts),
        "oversized_chunk_ids": [
            i for i, count in enumerate(token_counts) if count > embedder.model.max_seq_length
        ],
    })

    # The indexer checks token limits and vectors before writing them.
    db_path = ROOT / "output/qdrant_full_book" / output_dir.name
    print(f"Indexing {len(rows)} chunks into {db_path}", flush=True)
    with closing(QdrantClient(path=str(db_path))) as client:
        profile = build_qdrant_index(
            client, embedder, output_dir / "chunks.jsonl",
            output_dir / "index_manifest.json", batch_size=64,
            parser_path=output_dir / "parsed.jsonl",
        )

    # Reopen the database to check persistence and one vector search per unit.
    sample_ids = {}
    for point_id, row in enumerate(rows):
        sample_ids.setdefault(row["metadata"]["chapter_number"], point_id)

    checks = []
    with closing(QdrantClient(path=str(db_path))) as client:
        persisted_count = client.count(profile["collection"], exact=True).count
        points = client.retrieve(
            profile["collection"], ids=list(sample_ids.values()), with_payload=True, with_vectors=True,
        )
        if persisted_count != len(rows) or {point.id for point in points} != set(sample_ids.values()):
            raise ValueError("Persisted index is missing expected points")

        for point in points:
            hit = client.query_points(profile["collection"], query=point.vector, limit=1).points[0]
            valid_sample = point.payload == rows[point.id] and hit.score >= 0.999
            if not valid_sample:
                raise ValueError(f"Persisted sample mismatch: {point.id}")
            checks.append({
                "chapter_number": point.payload["metadata"]["chapter_number"],
                "point_id": point.id, "top_hit_id": hit.id, "score": hit.score,
            })

    write_json(output_dir / "index_verification.json", {
        "persisted_points": persisted_count, "sample_checks": checks,
        "quality_evaluation": "Not performed; persistence and vector-search checks only",
    })
    manifest.update(status="complete", index=profile, index_seconds=perf_counter() - started)
    manifest["artifacts"].update({
        name: sha256(output_dir / name)
        for name in ("token_validation.json", "index_manifest.json", "index_verification.json")
    })
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"Complete: {len(rows)} persisted vectors; {len(checks)} sample searches; "
        f"{manifest['index_seconds']:.1f}s", flush=True,
    )
    return manifest


def main():
    """Run preparation and/or indexing with isolated output paths."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/raw/Fundamentals_of_Database_System.pdf")
    parser.add_argument("--stage", choices=("prepare", "index", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-cache", type=Path, default=ROOT / "output/model_cache/e5_hub")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--exclude-footnotes", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.stage == "index" and args.output_dir is None:
        parser.error("--stage index requires --output-dir from a prepared run")
    run_id = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or ROOT / "output/ingestion/full_book" / run_id
    print(f"Output: {output_dir}", flush=True)

    if args.stage in ("prepare", "all"):
        prepare_ingestion(
            args.pdf, output_dir, chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap, include_footnotes=not args.exclude_footnotes,
        )
    if args.stage in ("index", "all"):
        index_ingestion(output_dir, args.model_cache, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
