"""Check book scope, opening pages, source coverage and persisted indexing."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
import numpy as np
import pymupdf
import pytest

from scripts import run_full_book_ingestion as ingestion
from src.ingestion.textbook_parser import TextbookParser


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "book.pdf"
    texts = ["Front matter", "Opening chapter: related data and database users.",
             "Part divider", "Opening chapter: database schema and architecture.",
             "Appendix storage details.", "Bibliography outside the selected corpus."]
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            page.insert_text((60, 110), text)
        doc[4].insert_text((60, 170), "A.1 Disk parameters")
        doc[4].insert_text((60, 220), "A disk stores blocks of related data.")
        doc.set_toc([[1, "Preface", 1], [1, "part 1 Introduction", 2],
                     [2, "chapter 1 Database Users", 2], [1, "part 2 Architecture", 3],
                     [2, "chapter 2 Database Architecture", 4],
                     [1, "appendix A Storage", 5], [1, "Selected Bibliography", 6]])
        doc.save(path)
    return path


def test_outline_bounds_exclude_dividers_and_back_matter(book):
    with pymupdf.open(book) as doc:
        ranges = ingestion.content_ranges(doc.get_toc(), len(doc))
    assert [(item["number"], item["start_page"], item["end_page"]) for item in ranges] == [
        ("1", 1, 1), ("2", 3, 3), ("A", 4, 4)]


def test_last_chapter_can_end_on_last_pdf_page():
    ranges = ingestion.content_ranges([[1, "chapter 1 Title", 1]], 2)
    assert ranges[0]["end_page"] == 1


@pytest.mark.parametrize("outline", [
    [], [[1, "Preface", 1]], [[1, "chapter 2 Title", 1]],
    [[1, "chapter 1 Title", 1], [1, "chapter 3 Title", 2]],
    [[1, "chapter 1 Title", 1], [1, "chapter 1 Duplicate", 2]],
    [[1, "chapter 1 Title", 0]], [[1, "chapter 1 Title", 4]],
])
def test_invalid_outline_is_rejected(outline):
    with pytest.raises(ValueError):
        ingestion.content_ranges(outline, 3)


def test_fixed_chapter_keeps_opening_page_and_appendix_section(book):
    parser = TextbookParser(book)
    try:
        chapter = {"number": "2", "title": "Database Architecture"}
        records = parser.parse_pages(3, 3, chapter=chapter)
        assert records and all(record["chapter_number"] == "2" for record in records)
        assert any("Opening chapter" in record["text"] for record in records)
        assert chapter == {"number": "2", "title": "Database Architecture"}
        appendix = parser.parse_pages(4, 4, chapter={"number": "A", "title": "Storage"})
        paragraph = next(record for record in appendix if "disk stores" in record["text"])
        assert paragraph["chapter_number"] == "A" and paragraph["section_number"] == "A.1"
        fallback = parser.parse_pages(1, 1)
        assert all(record["chapter_number"] is None for record in fallback)
    finally:
        parser.close()


def test_appendix_contents_entry_is_not_a_section_without_a_chapter(book):
    parser = TextbookParser(book)
    try:
        assert parser.detect_section("C.1 Basic Retrievals in QBE") is None
        parser.current_chapter = {"number": "C", "title": "QBE"}
        assert parser.detect_section("C.1 Basic Retrievals in QBE")["number"] == "C.1"
        parser.current_chapter = {"number": "1", "title": "Introduction"}
        assert parser.detect_section("C.1 Basic Retrievals in QBE") is None
    finally:
        parser.close()


@pytest.mark.parametrize("chapter", [{}, {"number": "1"}, {"number": 1, "title": "Title"}, "1"])
def test_invalid_fixed_chapter_is_rejected(book, chapter):
    parser = TextbookParser(book)
    try:
        with pytest.raises(ValueError, match="chapter"):
            parser.parse_pages(1, 1, chapter=chapter)
    finally:
        parser.close()


def test_preparation_preserves_sources_and_does_not_overwrite(book, tmp_path):
    output = tmp_path / "prepared"
    manifest = ingestion.prepare_ingestion(book, output, chunk_size=160, chunk_overlap=20)
    coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["status"] == "prepared" and manifest["chunks"] == len(rows)
    assert coverage["selected_pages"] == 3 and coverage["excluded_pdf_pages"] == [1, 3, 6]
    assert coverage["source_checks"]["missing_non_whitespace_records"] == 0
    assert {row["metadata"]["chapter_number"] for row in rows} == {"1", "2", "A"}
    assert not any("Bibliography" in row["page_content"] for row in rows)
    for name, digest in manifest["artifacts"].items():
        assert ingestion.sha256(output / name) == digest
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        ingestion.prepare_ingestion(book, output)
    assert (output / "manifest.json").read_bytes() == original


def test_blank_page_label_is_reported_and_excluded(tmp_path):
    path = tmp_path / "blank.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((60, 110), "A database is related data.")
        doc.new_page().insert_text((60, 110), "This page intentionally left blank")
        doc.set_toc([[1, "chapter 1 Database Users", 1]])
        doc.save(path)
    output = tmp_path / "prepared"
    ingestion.prepare_ingestion(path, output)
    coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["excluded_blank_page_records"] == 1
    assert coverage["selected_pages_without_records"] == [2]
    assert "This page intentionally left blank" not in (output / "chunks.jsonl").read_text(encoding="utf-8")


@pytest.fixture
def source_data():
    record = {"id": "book:p1:b1", "text": "related data", "pdf_page": 1,
              "chapter_number": "1", "type": "normal_text"}
    source = {"record_id": record["id"], "pdf_page": 1, "source_start_index": 0,
              "source_end_index": 12, "chunk_start_index": 0, "chunk_end_index": 12}
    chunk = Document(page_content="related data", metadata={"chapter_number": "1", "sources": [source]})
    return [record], [chunk]


@pytest.mark.parametrize("problem", ["missing_text", "wrong_offset", "wrong_page", "wrong_chapter"])
def test_source_validation_rejects_loss_or_bad_mapping(source_data, problem):
    records, chunks = deepcopy(source_data)
    if problem == "missing_text":
        chunks = []
    elif problem == "wrong_offset":
        chunks[0].metadata["sources"][0]["source_start_index"] = 1
    elif problem == "wrong_page":
        chunks[0].metadata["sources"][0]["pdf_page"] = 2
    else:
        chunks[0].metadata["chapter_number"] = "2"
    with pytest.raises(ValueError):
        ingestion.validate_sources(records, chunks, include_footnotes=True)


def test_full_preparation_to_persistent_index(book, tmp_path, monkeypatch):
    import qdrant_client
    from src.ingestion import textbook_embedder

    class Embedder:
        def __init__(self, model, revision, **kwargs):
            self.model_name, self.revision = model, revision
            self.model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 3,
                                          max_seq_length=512,
                                          tokenizer=lambda text, **kw: {"input_ids": text.split()})

        def embed_documents(self, chunks):
            vectors = np.array([[len(chunk.page_content), 1, 2] for chunk in chunks], dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    client_class = qdrant_client.QdrantClient
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: client_class(path=str(tmp_path / "db")))
    monkeypatch.setattr(textbook_embedder, "E5Embedder", Embedder)
    output = tmp_path / "prepared"
    prepared = ingestion.prepare_ingestion(book, output)
    result = ingestion.index_ingestion(output, tmp_path / "model")
    assert result["status"] == "complete" and result["index"]["status"] == "complete"
    assert result["index"]["chunks"] == prepared["chunks"]
    checked = json.loads((output / "index_verification.json").read_text(encoding="utf-8"))
    assert {row["chapter_number"] for row in checked["sample_checks"]} == {"1", "2", "A"}
    assert checked["persisted_points"] == prepared["chunks"]
    (output / "chunks.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        ingestion.index_ingestion(output, tmp_path / "model")
