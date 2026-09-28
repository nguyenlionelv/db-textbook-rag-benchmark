"""Unit tests for the production chunking strategies."""

import importlib.util
import json
from copy import deepcopy
from itertools import count
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "ingestion" / "textbook_chunker.py"
spec = importlib.util.spec_from_file_location("textbook_chunker_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BaseChunker = module.BaseChunker
StrictFixedSizeChunker = module.StrictFixedSizeChunker
GlobalRecursiveChunker = module.GlobalRecursiveChunker
StructureAwareRecursiveChunker = module.StructureAwareRecursiveChunker
RECORD_NUMBERS = count()
CHUNKER_CLASSES = (StrictFixedSizeChunker, GlobalRecursiveChunker, StructureAwareRecursiveChunker)


def record(
    text, section_number="1.1", section_title="Introduction", kind="normal_text",
    **overrides,
):
    """Fixture đủ schema Parser, dùng ID riêng kể cả khi hai record cùng text."""
    metadata = {"footnote_refs": []}
    if kind == "footnote":
        metadata = {
            "marker": "1", "referenced_by": [], "reference_contexts": [],
            "status": "unreferenced", "recognition_basis": "layout_only",
            "source_block_numbers": [0],
        }
    result = {
        "id": f"test-document:p0:b{next(RECORD_NUMBERS)}",
        "pdf_page": 0,
        "chapter_number": "1",
        "chapter_title": "Databases and Database Users",
        "section_number": section_number,
        "section_title": section_title,
        "type": kind,
        "text": text,
        "raw_text": text,
        "bbox": [60.0, 100.0, 400.0, 120.0],
        "metadata": metadata,
    }
    result.update(overrides)
    return result


def test_base_chunker_is_abstract_and_configuration_is_validated():
    with pytest.raises(TypeError):
        BaseChunker()
    with pytest.raises(ValueError):
        StrictFixedSizeChunker(chunk_size=0)
    with pytest.raises(ValueError):
        StrictFixedSizeChunker(chunk_size=10, chunk_overlap=10)


@pytest.mark.parametrize("kwargs", [
    {"chunk_size": True}, {"chunk_size": 1000.0},
    {"chunk_overlap": -1}, {"chunk_overlap": 1.5},
    {"chunk_overlap": False}, {"include_footnotes": "false"},
])
def test_configuration_requires_correct_types(kwargs):
    with pytest.raises(ValueError):
        StrictFixedSizeChunker(**kwargs)


def test_fixed_size_uses_exact_character_overlap():
    chunks = StrictFixedSizeChunker(chunk_size=30, chunk_overlap=5).chunk([
        record("A" * 80),
    ])

    assert all(len(chunk.page_content) <= 30 for chunk in chunks)
    assert all(
        left.page_content[-5:] == right.page_content[:5]
        for left, right in zip(chunks, chunks[1:])
    )


def test_footnotes_are_excluded_by_default():
    chunks = StrictFixedSizeChunker(chunk_size=200, chunk_overlap=0).chunk([
        record("Main text"),
        record("Footnote should not appear", kind="footnote"),
    ])

    assert "Footnote should not appear" not in chunks[0].page_content


def test_structure_aware_chunks_keep_one_section_and_hierarchy():
    chunks = StructureAwareRecursiveChunker(chunk_size=45, chunk_overlap=5).chunk([
        record("First section sentence. " * 4),
        record("Nested section sentence. " * 4, "1.2.1", "Nested Section"),
    ])

    assert {chunk.metadata["section_number"] for chunk in chunks} == {"1.1", "1.2.1"}
    assert all(chunk.metadata["strategy"] == "structure_aware_recursive" for chunk in chunks)
    nested = next(chunk for chunk in chunks if chunk.metadata["section_number"] == "1.2.1")
    assert nested.metadata["parent_section_id"] == "sec_1.2"


def test_recursive_chunker_returns_global_offsets():
    chunks = GlobalRecursiveChunker(chunk_size=50, chunk_overlap=10).chunk([
        record("A sentence with enough words. " * 5),
    ])

    assert chunks
    assert all(
        chunk.metadata["end_index"]
        == chunk.metadata["start_index"] + len(chunk.page_content)
        for chunk in chunks
    )


@pytest.mark.parametrize("field", [
    "id", "pdf_page", "chapter_number", "chapter_title", "section_number",
    "section_title", "type", "text", "raw_text", "bbox", "metadata",
])
def test_preparation_rejects_missing_parser_fields_with_record_context(field):
    item = record("Content", id="broken-record")
    del item[field]
    with pytest.raises(ValueError, match=rf"record\[0\].*{field}"):
        StrictFixedSizeChunker()._prepare_chapters([item])


@pytest.mark.parametrize(("field", "value"), [
    ("text", None), ("raw_text", 123), ("chapter_number", 1),
    ("section_number", 1.1), ("pdf_page", True), ("metadata", []),
])
def test_preparation_rejects_wrong_types_instead_of_coercing_to_text(field, value):
    item = record("Content", id="broken-record")
    item[field] = value
    with pytest.raises(ValueError, match=rf"broken-record.*{field}"):
        StrictFixedSizeChunker()._prepare_chapters([item])


@pytest.mark.parametrize(("field", "value"), [
    ("pdf_page", -1), ("section_number", " "),
    ("bbox", [0, 1, 2]), ("bbox", [0, 1, float("nan"), 3]),
    ("bbox", [0, 1, True, 3]), ("metadata", {"not_json": {1, 2}}),
])
def test_preparation_rejects_invalid_source_values(field, value):
    with pytest.raises(ValueError, match=field):
        StrictFixedSizeChunker()._prepare_chapters([record("Content", **{field: value})])


def test_preparation_checks_unique_ids_and_accepts_additional_fields():
    item = record("Content", extra="unused", bbox=(0, 1, 2, 3))
    chapters = StrictFixedSizeChunker()._prepare_chapters([item])
    assert chapters[0]["source_spans"][0]["record"]["extra"] == "unused"
    with pytest.raises(ValueError, match="id bị trùng"):
        StrictFixedSizeChunker()._prepare_chapters([item, item])


@pytest.mark.parametrize("records", ["not records", {"text": "Content"}, [None]])
def test_preparation_rejects_invalid_input_container(records):
    with pytest.raises(ValueError):
        StrictFixedSizeChunker()._prepare_chapters(records)


@pytest.mark.parametrize("chapter_number", [None, "", " "])
def test_retained_record_requires_chapter_context(chapter_number):
    with pytest.raises(ValueError, match="missing-chapter.*chapter_number"):
        StrictFixedSizeChunker()._prepare_chapters([
            record("Content", id="missing-chapter", chapter_number=chapter_number),
        ])


def test_filtering_precedes_required_chapter_context_and_preserves_content_types():
    records = [
        record(" ", chapter_number=None),
        record("Note", kind="footnote", chapter_number=None),
        record("Body"), record("Table", kind="table_like"),
        record("Figure", kind="figure_caption"), record("Caption", kind="table_caption"),
    ]
    chapter = StrictFixedSizeChunker()._prepare_chapters(records)[0]
    assert [span["record"]["type"] for span in chapter["source_spans"]] == [
        "normal_text", "table_like", "figure_caption", "table_caption",
    ]
    assert StrictFixedSizeChunker()._prepare_chapters(records[:2]) == []
    assert StrictFixedSizeChunker()._prepare_chapters([]) == []


def test_preparation_separates_chapters_and_resets_section_unit_index():
    chapters = StrictFixedSizeChunker()._prepare_chapters([
        record("First", section_number="1.1"),
        record("Second", section_number="2.1", chapter_number="2", chapter_title="Chapter Two"),
    ])
    assert [chapter["metadata"]["chapter_number"] for chapter in chapters] == ["1", "2"]
    assert all(chapter["sections"][0]["metadata"]["section_unit_index"] == 0 for chapter in chapters)
    assert "Second" not in chapters[0]["text"]
    assert "First" not in chapters[1]["text"]


def test_preparation_rejects_a_chapter_reappearing_after_another_chapter():
    with pytest.raises(ValueError, match="chapter_number.*xuất hiện lại"):
        StrictFixedSizeChunker()._prepare_chapters([
            record("First"),
            record("Second", chapter_number="2", chapter_title="Chapter Two"),
            record("Back to chapter one"),
        ])


def test_reappearing_section_and_included_footnote_keep_reading_order():
    records = [
        record("First section", section_number="1.1"),
        record("Second section", section_number="1.2", section_title="Second"),
        record("Note for first section", section_number="1.1", kind="footnote"),
    ]
    chapter = StrictFixedSizeChunker(include_footnotes=True)._prepare_chapters(records)[0]
    assert [section["metadata"]["section_number"] for section in chapter["sections"]] == [
        "1.1", "1.2", "1.1",
    ]
    assert [section["metadata"]["section_unit_index"] for section in chapter["sections"]] == [0, 1, 2]
    assert [span["record"]["id"] for span in chapter["source_spans"]] == [r["id"] for r in records]


def test_preparation_normalizes_intro_and_hierarchy_without_changing_sources():
    chapter = StrictFixedSizeChunker()._prepare_chapters([
        record("Opening", section_number=None, section_title=None),
        record("Nested", section_number="1.3.1", section_title=None),
    ])[0]
    intro, nested = chapter["sections"]
    assert intro["metadata"]["section_number"] == "1.intro"
    assert intro["metadata"]["section_id"] == "sec_1.intro"
    assert intro["metadata"]["parent_section_id"] == "chapter_1"
    assert intro["text"] == "[[SECTION 1.intro: Chapter Introduction]]\nOpening"
    assert intro["source_spans"][0]["record"]["section_number"] is None
    assert nested["metadata"]["section_title"] == "Untitled Section"
    assert nested["metadata"]["parent_section_id"] == "sec_1.3"


def test_preparation_resolves_titles_from_later_records_in_the_same_unit():
    chapter = StrictFixedSizeChunker()._prepare_chapters([
        record("First", chapter_title=None, section_title=None),
        record("Second"),
    ])[0]
    assert chapter["metadata"]["chapter_title"] == "Databases and Database Users"
    assert chapter["sections"][0]["metadata"]["section_title"] == "Introduction"
    assert chapter["sections"][0]["text"].startswith("[[SECTION 1.1: Introduction]]\n")
    untitled = StrictFixedSizeChunker()._prepare_chapters([record("Content", chapter_title=None)])
    assert untitled[0]["metadata"]["chapter_title"] is None


@pytest.mark.parametrize("field", ["chapter_title", "section_title"])
def test_preparation_rejects_conflicting_titles(field):
    with pytest.raises(ValueError, match=rf"{field}.*mâu thuẫn"):
        StrictFixedSizeChunker()._prepare_chapters([
            record("First"), record("Second", **{field: "Different title"}),
        ])


def test_prepared_text_and_source_spans_preserve_whitespace_and_input():
    reference = {"marker": "1", "footnote_id": "note-1", "raw_offset": 5}
    records = [
        record("  repeated text  ", metadata={"footnote_refs": [reference]}),
        record("  repeated text  "),
        record("Nested", section_number="1.2.1", section_title="Nested"),
    ]
    before = deepcopy(records)
    chapter = StrictFixedSizeChunker()._prepare_chapters(records)[0]
    assert chapter["text"] == (
        "[[SECTION 1.1: Introduction]]\n  repeated text  \n  repeated text  \n\n"
        "[[SECTION 1.2.1: Nested]]\nNested"
    )
    for span in chapter["source_spans"]:
        assert chapter["text"][span["start"]:span["end"]] == span["record"]["text"]
    for section in chapter["sections"]:
        assert chapter["text"][section["start"]:section["end"]] == section["text"]
        for span in section["source_spans"]:
            assert section["text"][span["start"]:span["end"]] == span["record"]["text"]
    assert records == before
    chapter["source_spans"][0]["record"]["metadata"]["footnote_refs"][0]["marker"] = "changed"
    assert records == before


@pytest.mark.parametrize("chunker_class", CHUNKER_CLASSES)
def test_strategies_split_chapters_independently_and_reset_chunk_index(chunker_class):
    records = [
        record("First chapter sentence. " * 8),
        record("Second chapter sentence. " * 8, section_number="2.1",
               chapter_number="2", chapter_title="Chapter Two"),
    ]
    chunks = chunker_class(chunk_size=80, chunk_overlap=20).chunk(records)
    for number in ("1", "2"):
        chapter_chunks = [chunk for chunk in chunks if chunk.metadata["chapter_number"] == number]
        assert len(chapter_chunks) > 1
        assert [chunk.metadata["chunk_index"] for chunk in chapter_chunks] == list(range(len(chapter_chunks)))
        assert chapter_chunks[0].metadata["start_index"] == 0
        assert all(
            source["record_id"] == records[int(number) - 1]["id"]
            for chunk in chapter_chunks for source in chunk.metadata["sources"]
        )


@pytest.mark.parametrize("chunker_class", CHUNKER_CLASSES)
def test_chunk_offsets_and_source_ranges_match_actual_text(chunker_class):
    records = [
        record("  A sentence with whitespace. " * 4, bbox=(0, 1, 2, 3)),
        record("Another sentence. " * 6),
        record("Nested section. " * 6, section_number="1.2.1", section_title="Nested"),
    ]
    before = deepcopy(records)
    chunker = chunker_class(chunk_size=80, chunk_overlap=20)
    chunks = chunker.chunk(records)
    prepared = chunker._prepare_chapters(records)[0]
    records_by_id = {item["id"]: item for item in records}
    for chunk in chunks:
        metadata = chunk.metadata
        text = prepared["text"]
        if metadata["offset_scope"] == "section":
            text = prepared["sections"][metadata["section_unit_index"]]["text"]
            assert len(metadata["section_ids"]) == 1
        assert 0 < len(chunk.page_content) <= 80
        assert text[metadata["start_index"]:metadata["end_index"]] == chunk.page_content
        assert metadata["end_index"] - metadata["start_index"] == len(chunk.page_content)
        for source in metadata["sources"]:
            original = records_by_id[source["record_id"]]
            assert source["pdf_page"] == original["pdf_page"]
            assert source["bbox"] == list(original["bbox"])
            assert source["section_number"] == original["section_number"]
            assert source["source_start_index"] < source["source_end_index"]
            assert original["text"][source["source_start_index"]:source["source_end_index"]] == (
                chunk.page_content[source["chunk_start_index"]:source["chunk_end_index"]]
            )
        json.dumps({"page_content": chunk.page_content, "metadata": metadata}, allow_nan=False)
    assert records == before


@pytest.mark.parametrize("chunker_class", CHUNKER_CLASSES)
def test_sources_only_include_records_intersecting_the_chunk(chunker_class):
    records = [record("A" * 100), record("B" * 100)]
    chunks = chunker_class(chunk_size=60, chunk_overlap=10).chunk(records)
    first_with_source = next(chunk for chunk in chunks if chunk.metadata["sources"])
    assert [source["record_id"] for source in first_with_source.metadata["sources"]] == [records[0]["id"]]
    assert first_with_source.metadata["sources"][0]["source_end_index"] < len(records[0]["text"])


@pytest.mark.parametrize("chunker_class", [StrictFixedSizeChunker, GlobalRecursiveChunker])
def test_global_strategies_describe_chunks_crossing_multiple_sections(chunker_class):
    chunks = chunker_class(chunk_size=300, chunk_overlap=0).chunk([
        record("First section"),
        record("Second section", section_number="1.2", section_title="Second"),
    ])
    assert len(chunks) == 1
    assert chunks[0].metadata["section_ids"] == ["sec_1.1", "sec_1.2"]
    assert len(chunks[0].metadata["sources"]) == 2
    assert all(chunks[0].metadata[field] is None for field in (
        "section_id", "section_number", "section_title", "parent_section_id", "section_unit_index",
    ))


def test_structure_aware_offsets_identify_repeated_section_units():
    records = [
        record("First section"),
        record("Second section", section_number="1.2", section_title="Second"),
        record("Back to first section", kind="footnote"),
    ]
    chunks = StructureAwareRecursiveChunker(include_footnotes=True).chunk(records)
    assert [chunk.metadata["section_unit_index"] for chunk in chunks] == [0, 1, 2]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["start_index"] == 0 for chunk in chunks)
    assert chunks[-1].metadata["sources"][0]["metadata"] == records[-1]["metadata"]


@pytest.mark.parametrize("chunker_class", CHUNKER_CLASSES)
def test_footnote_reference_metadata_is_preserved_without_aliasing(chunker_class):
    reference = {"marker": "1", "footnote_id": "note-1", "raw_offset": 5}
    item = record("Repeated sentence. " * 10, metadata={"footnote_refs": [reference]})
    chunks = chunker_class(chunk_size=70, chunk_overlap=20).chunk([item])
    sources = [source for chunk in chunks for source in chunk.metadata["sources"]]
    assert len(sources) > 1
    assert all(source["metadata"] == item["metadata"] for source in sources)
    sources[0]["metadata"]["footnote_refs"][0]["marker"] = "changed"
    assert sources[1]["metadata"]["footnote_refs"][0]["marker"] == "1"
    assert item["metadata"]["footnote_refs"][0]["raw_offset"] == 5
    assert item["metadata"]["footnote_refs"][0]["marker"] == "1"


@pytest.mark.parametrize("chunker_class", CHUNKER_CLASSES)
def test_identical_record_text_still_traces_every_non_whitespace_character(chunker_class):
    records = [record("foo " * 8), record("foo " * 8)]
    chunks = chunker_class(chunk_size=50, chunk_overlap=40).chunk(records)
    coverage = {item["id"]: set() for item in records}
    for chunk in chunks:
        for source in chunk.metadata["sources"]:
            coverage[source["record_id"]].update(range(source["source_start_index"], source["source_end_index"]))
    for item in records:
        assert {index for index, char in enumerate(item["text"]) if not char.isspace()} <= coverage[item["id"]]


@pytest.mark.parametrize("chunker_class", [GlobalRecursiveChunker, StructureAwareRecursiveChunker])
@pytest.mark.parametrize("separators", [[], ["\n", " "], "", [1, ""]])
def test_recursive_separator_configuration_requires_character_fallback(chunker_class, separators):
    with pytest.raises(ValueError, match="separators"):
        chunker_class(separators=separators)


@pytest.mark.parametrize("chunker_class", CHUNKER_CLASSES)
def test_empty_or_filtered_input_returns_no_chunks(chunker_class):
    assert chunker_class().chunk([]) == []
    assert chunker_class().chunk([record("  "), record("Note", kind="footnote")]) == []


def test_contract_example_output_and_header_only_chunks():
    item = record("A database stores related data.", id="example-document:p34:b3", pdf_page=34,
                  bbox=[117.0, 373.8, 467.4, 444.3])
    chunk = StructureAwareRecursiveChunker().chunk([item])[0]
    assert chunk.page_content == "[[SECTION 1.1: Introduction]]\nA database stores related data."
    assert chunk.metadata["end_index"] == 61
    assert chunk.metadata["sources"] == [{
        "record_id": item["id"], "pdf_page": 34, "type": "normal_text", "bbox": item["bbox"],
        "section_number": "1.1", "section_title": "Introduction",
        "source_start_index": 0, "source_end_index": 31,
        "chunk_start_index": 30, "chunk_end_index": 61, "metadata": {"footnote_refs": []},
    }]
    header_chunk = StrictFixedSizeChunker(chunk_size=10, chunk_overlap=0).chunk([item])[0]
    assert header_chunk.metadata["sources"] == []
    assert header_chunk.metadata["section_id"] == "sec_1.1"
