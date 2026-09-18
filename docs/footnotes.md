# Footnote records

`TextbookParser.parse_pages(start_page, end_page)` uses inclusive, zero-based PDF page indices.
Existing record fields are retained. Every record also has `id`, `raw_text`, `bbox` and `metadata`.

## Paragraph references

```json
{
  "type": "normal_text",
  "text": "A database is a collection of related data. By data, we mean ...",
  "metadata": {
    "footnote_refs": [{
      "marker": "1",
      "footnote_id": "<document>:p34:fn1:b8",
      "status": "resolved",
      "anchor_text": "A database is a collection of related data.",
      "raw_offset": 41,
      "bbox": [292, 449, 298, 458]
    }]
  }
}
```

`raw_offset` is a character offset in the record's `raw_text`, **not** in cleaned `text`.
`anchor_text` is up to 160 characters immediately before the marker, with whitespace cleaned.
Geometry is measured in PDF points. Example IDs, offsets and coordinates above are illustrative.

## Footnotes

Footnotes have `type: "footnote"`. Their cleaned `text` omits the leading marker;
`raw_text` retains it. Their metadata contains:

- `marker`: the printed marker, as a string.
- `referenced_by`: IDs of paragraphs referencing this note.
- `reference_contexts`: chapter/section context of each referencing paragraph.
- `status`: `resolved` or `unreferenced`.
- `recognition_basis`: `layout_and_marker` or `layout_only`.
- `source_block_numbers`: original blocks, including merged continuation blocks.

The footnote's chapter/section fields come from its linked paragraphs. If they disagree,
the corresponding field is `null`; all contexts remain in metadata. An unreferenced note
has no inferred section. Footnotes are appended after the body's records on each page.

## Detection and linking

The parser uses span font sizes, baselines and superscript flags before cleaning.
It compares small-font note candidates in the lower page region with inline markers,
and merges nearby same-column continuation blocks. It detects notes before applying
the existing footer exclusion rule.

Numeric markers (1–999) are linked **within the same page**. Document IDs are derived
from PDF content; block/page IDs are stable across file moves and parsing ranges.
When several candidate notes share a marker, the paragraph reference is `ambiguous`.
When no target is found, it is `unresolved`. Both statuses have `footnote_id: null` and
preserve the marker in cleaned text. Only a uniquely resolved marker is removed.

This is a layout heuristic for native text PDFs, not OCR. Symbol markers, cross-page
footnotes, multiple notes packed into one block and equations embedded in prose may
need additional rules. Review unresolved references before treating all numbers as
footnote markers. This change does not reconstruct table cells.

## Using references for RAG

When a retrieved chunk contains `footnote_refs`, fetch the corresponding `footnote_id`
records and include their text in the context. If combining paragraphs into chunks,
retain the union of their reference metadata and original record IDs.

## Verification

```powershell
python -m pytest tests/test_textbook_footnotes.py
```

Unit tests do not require the private textbook. The integration test runs when
`data/raw/Fundamentals_of_Database_System.pdf` exists, or when `TEXTBOOK_TEST_PDF`
points to that file; otherwise it is skipped.
