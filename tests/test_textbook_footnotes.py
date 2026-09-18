"""Layout-based regression tests; no textbook PDF is required for unit tests."""
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "src" / "ingestion" / "textbook_parser.py"
if not PARSER_PATH.exists():
    PARSER_PATH = Path(__file__).with_name("textbook_parser.py")
spec = importlib.util.spec_from_file_location("textbook_parser_under_test", PARSER_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TextbookParser = module.TextbookParser


def span(text, size=10.5, x=60, y=150, flags=0):
    return {
        "text": text, "size": size, "flags": flags,
        "origin": (x, y), "bbox": (x, y - size, x + max(1, len(text)) * size / 2, y),
    }


def block(number, *lines):
    spans = [item for line in lines for item in line]
    return {
        "number": number, "type": 0,
        "bbox": (
            min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
            max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans),
        ),
        "lines": [{"spans": line} for line in lines],
    }


def body(number=0, marker="1", y=150, flags=1):
    return block(number, [
        span("A database is related data.", y=y),
        span(marker, 7.875, 195, y - 4, flags),
        span(" By data, we mean facts.", x=202, y=y),
    ])


def note(number=1, marker="1", y=735):
    return block(number, [
        span(marker, 6, 60, y - 3),
        span("Data is both singular and plural.", 8, 64, y),
    ])


class Page:
    class Rect:
        height = 800

    rect = Rect()

    def __init__(self, blocks):
        self.blocks = blocks

    def get_text(self, kind):
        if kind == "dict":
            return {"blocks": self.blocks}
        return [
            (*b["bbox"], "\n".join("".join(s["text"] for s in line["spans"]) for line in b["lines"]), b["number"], 0)
            for b in self.blocks
        ]


def parser(*pages):
    instance = TextbookParser.__new__(TextbookParser)
    instance.doc = [Page(page) for page in pages]
    instance.document_id = "test-document"
    instance.current_chapter = instance.current_section = None
    instance.chapter_pattern = re.compile(r"\bChapter\s+(\d+)\s+(.+)$", re.I)
    instance.section_pattern = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(.+)$")
    return instance


def test_links_marker_preserves_original_and_removes_only_resolved_digit():
    records = parser([body(), note()]).parse_pages(0, 0)
    paragraph, footnote = records
    ref = paragraph["metadata"]["footnote_refs"][0]
    assert paragraph["type"] == "normal_text"
    assert "data.1 By" in paragraph["raw_text"]
    assert "data. By" in paragraph["text"]
    assert paragraph["raw_text"][ref["raw_offset"]] == "1"
    assert ref["status"] == "resolved" and ref["marker"] == "1"
    assert ref["footnote_id"] == footnote["id"]
    assert footnote["type"] == "footnote"
    assert footnote["text"] == "Data is both singular and plural."
    assert footnote["metadata"]["referenced_by"] == [paragraph["id"]]
    assert footnote["raw_text"].startswith("1")


def test_geometry_detects_marker_without_superscript_flag():
    records = parser([body(flags=0), note()]).parse_pages(0, 0)
    assert records[0]["metadata"]["footnote_refs"][0]["status"] == "resolved"


def test_smaller_body_font_on_code_heavy_page_does_not_hide_note():
    paragraph = body()
    for s in paragraph["lines"][0]["spans"]:
        s["size"] = 9 if s["size"] == 10.5 else 6.75
    records = parser([paragraph, note()]).parse_pages(0, 0)
    assert records[0]["metadata"]["footnote_refs"][0]["status"] == "resolved"


def test_footnote_content_can_start_with_a_symbol():
    footnote = note()
    footnote["lines"][0]["spans"][1]["text"] = "<TR> stands for table row."
    records = parser([body(), footnote]).parse_pages(0, 0)
    assert records[-1]["type"] == "footnote"
    assert records[-1]["text"].startswith("<TR>")
    assert records[0]["metadata"]["footnote_refs"][0]["status"] == "resolved"


def test_short_footnote_with_a_distinct_marker_is_retained():
    footnote = note()
    footnote["lines"][0]["spans"][1]["text"] = "See Section 1.2."
    records = parser([body(), footnote]).parse_pages(0, 0)
    assert records[-1]["type"] == "footnote"
    assert records[-1]["text"] == "See Section 1.2."
    assert records[0]["metadata"]["footnote_refs"][0]["status"] == "resolved"


def test_missing_footnote_keeps_marker_and_marks_unresolved():
    record = parser([body()]).parse_pages(0, 0)[0]
    assert "data.1 By" in record["text"]
    assert record["metadata"]["footnote_refs"][0]["status"] == "unresolved"
    assert record["metadata"]["footnote_refs"][0]["footnote_id"] is None


def test_duplicate_marker_is_ambiguous_and_has_unique_note_ids():
    records = parser([body(), note(), note(number=2, y=750)]).parse_pages(0, 0)
    assert records[0]["metadata"]["footnote_refs"][0]["status"] == "ambiguous"
    assert "data.1 By" in records[0]["text"]
    assert len({r["id"] for r in records}) == 3


def test_marker_numbers_are_scoped_to_page_and_ids_are_stable():
    instance = parser([body(), note()], [body(), note()])
    first = instance.parse_pages(0, 1)
    assert len({r["id"] for r in first}) == 4
    for r in first:
        for ref in r["metadata"].get("footnote_refs", []):
            target = next(n for n in first if n["id"] == ref["footnote_id"])
            assert target["pdf_page"] == r["pdf_page"]
    second_page = instance.parse_pages(1, 1)
    assert [r["id"] for r in second_page] == [r["id"] for r in first if r["pdf_page"] == 1]


def test_reference_does_not_resolve_to_next_page():
    next_page_body = block(0, [span("The discussion continues on this page.")])
    records = parser([body()], [next_page_body, note()]).parse_pages(0, 1)
    assert records[0]["metadata"]["footnote_refs"][0]["status"] == "unresolved"
    assert records[-1]["metadata"]["status"] == "unreferenced"


def test_note_section_comes_from_reference_before_later_heading():
    heading1 = block(0, [span("1.1 Introduction", y=100)])
    heading2 = block(2, [span("1.2 An Example", y=300)])
    records = parser([heading1, body(number=1), heading2, note(number=3)]).parse_pages(0, 0)
    assert records[-1]["section_number"] == "1.1"


def test_footnote_is_detected_before_footer_filter_and_continuation_is_merged():
    continuation = block(2, [span("This definition is important to retain.", 8, 60, 782)])
    records = parser([body(), note(y=770), continuation]).parse_pages(0, 0)
    assert len(records) == 2
    assert "important to retain" in records[-1]["text"]
    assert records[-1]["metadata"]["source_block_numbers"] == [1, 2]
    assert records[-1]["bbox"][3] == 782


def test_table_numbers_and_short_math_expression_are_preserved():
    table = block(0, [span("Smith", y=130)], [span("17", y=145)], [span("1", y=160)], [span("CS", y=175)])
    expression = block(1, [span("x", y=220), span("1", 7.875, 65, 216, 1), span(" + y", x=70, y=220)])
    records = parser([table, expression, note(number=2)]).parse_pages(0, 0)
    assert records[0]["type"] == "table_like"
    assert records[0]["text"] == "Smith 17 1 CS"
    assert records[1]["text"] == "x1 + y"
    assert records[1]["metadata"]["footnote_refs"] == []


def test_normal_digit_without_superscript_is_not_removed():
    paragraph = block(0, [span("There are 17 students in class 1.")])
    records = parser([paragraph, note()]).parse_pages(0, 0)
    assert records[0]["text"] == "There are 17 students in class 1."
    assert records[0]["metadata"]["footnote_refs"] == []


def test_variable_power_embedded_in_prose_is_preserved_even_with_matching_note():
    expression = block(0, [
        span("We compute the value x"), span("1", 7.875, 170, 146, 1),
        span(" for this example.", x=175),
    ])
    records = parser([expression, note()]).parse_pages(0, 0)
    assert "value x1" in records[0]["text"]
    assert records[0]["metadata"]["footnote_refs"] == []


# def test_export_preserves_links_and_is_json_serializable(tmp_path):
#     instance = parser([body(), note()])
#     records = instance.parse_pages(0, 0)
#     output = tmp_path / "records.jsonl"
#     instance.export_jsonl(records, output)
#     exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
#     assert exported[0]["metadata"]["footnote_refs"][0]["footnote_id"] == exported[1]["id"]


@pytest.mark.parametrize("bounds", [(-1, 0), (0, 1), (1, 0)])
def test_invalid_page_range_is_rejected(bounds):
    with pytest.raises(ValueError):
        parser([body()]).parse_pages(*bounds)


def test_actual_textbook_footnotes():
    path = Path(os.environ.get("TEXTBOOK_TEST_PDF", ROOT / "data/raw/Fundamentals_of_Database_System.pdf"))
    if not path.exists():
        pytest.skip("Private textbook PDF is not available")
    instance = TextbookParser(path)
    try:
        records = instance.parse_pages(34, 38)
        notes = [r for r in records if r["type"] == "footnote"]
        assert {(n["pdf_page"], n["metadata"]["marker"]) for n in notes} == {(34, "1"), (36, "2"), (37, "3")}
        assert all(n["metadata"]["status"] == "resolved" for n in notes)
        assert next(n for n in notes if n["metadata"]["marker"] == "2")["section_number"] == "1.1"
        note1 = next(n for n in notes if n["metadata"]["marker"] == "1")
        paragraph = next(r for r in records if r["id"] in note1["metadata"]["referenced_by"])
        assert "related data.1" in paragraph["raw_text"]
        assert "related data. By data" in paragraph["text"]
        assert "both singular and plural" in note1["text"]
        for page_idx, marker in [(215, "6"), (460, "2"), (945, "7")]:
            page_records = instance.parse_pages(page_idx, page_idx)
            target = next(r for r in page_records if r["type"] == "footnote" and r["metadata"]["marker"] == marker)
            assert target["metadata"]["status"] == "resolved"
    finally:
        instance.close()
