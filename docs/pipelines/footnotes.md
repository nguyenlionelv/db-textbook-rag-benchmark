# TextbookParser output schema and footnotes

`TextbookParser.parse_pages(start_page, end_page)` uses inclusive, zero-based PDF page indices.
Existing record fields are retained. Every record also has `id`, `raw_text`, `bbox` and `metadata`.

## Output format

`TextbookParser.export_jsonl(...)` writes **JSON Lines (JSONL)**, not one JSON array:
each non-empty line is one independent record. The current sample file is
`output/parsed_chapter1.jsonl` (not `parsed_chapter1.json`). Despite its legacy name,
the current file contains records from Chapters 1, 2 and 3, so downstream code must
filter the requested chapter instead of inferring it from the filename.

All records share this top-level schema:

| Field | Type | Nullable | Meaning |
| --- | --- | --- | --- |
| `id` | `string` | No | Stable record ID derived from PDF content, page and block. |
| `pdf_page` | `integer` | No | Zero-based PDF page index. |
| `chapter_number` | `string` | Yes | Detected chapter number, for example `"1"`. |
| `chapter_title` | `string` | Yes | Detected chapter title. |
| `section_number` | `string` | Yes | Detected section number, for example `"1.3.1"`; chapter introductions and unlinked notes may be `null`. |
| `section_title` | `string` | Yes | Section title; body records before the first section use `"Chapter Introduction"`. |
| `type` | `string` | No | `normal_text`, `table_like`, `figure_caption`, `table_caption`, or `footnote`. |
| `text` | `string` | No | Cleaned text used by downstream chunking. |
| `raw_text` | `string` | No | Text before cleaning; preserves line breaks and printed footnote markers. |
| `bbox` | `[number, number, number, number]` | No | PDF coordinates `[x0, y0, x1, y1]` in points. |
| `metadata` | `object` | No | Type-specific metadata described below. |

Equivalent JSON shape (values are illustrative):

```json
{
  "id": "<document>:p34:b1",
  "pdf_page": 34,
  "chapter_number": "1",
  "chapter_title": "Databases and Database Users",
  "section_number": "1.1",
  "section_title": "Introduction",
  "type": "normal_text",
  "text": "Cleaned content used for chunking.",
  "raw_text": "Content before whitespace and hyphenation cleanup.\n",
  "bbox": [117.0, 373.8, 467.4, 444.3],
  "metadata": {
    "footnote_refs": []
  }
}
```

For every non-footnote record, `metadata` has `footnote_refs`, which is an empty
array when no inline reference was detected. A reference object has this schema:

| Field | Type | Meaning |
| --- | --- | --- |
| `marker` | `string` | Printed numeric marker. |
| `footnote_id` | `string \| null` | Linked note ID only when resolution is unique. |
| `status` | `string` | `resolved`, `ambiguous`, or `unresolved`. |
| `anchor_text` | `string` | Cleaned text immediately before the marker. |
| `raw_offset` | `integer` | Marker offset in `raw_text`. |
| `bbox` | `[number, number, number, number]` | Marker coordinates in PDF points. |

For `type: "footnote"`, `metadata` instead contains `marker`, `referenced_by`,
`reference_contexts`, `status`, `recognition_basis`, and `source_block_numbers`,
as detailed in the Footnotes section below.

### Contract for chunking

The chunkers consume the parsed records in their original reading order and use
top-level `text`, `type`, `chapter_*`, and `section_*` fields. By default they discard
empty text and `type: "footnote"`. Callers may pass multiple chapters; all three
strategies handle each chapter independently. Structure-aware chunking also keeps
each consecutive section unit separate. See [chunker_contract.md](chunker_contract.md)
for the complete input/output contract.

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

Chunks retain original record IDs and reference metadata in `metadata.sources`.
Read each source's `metadata.footnote_refs`, fetch the corresponding `footnote_id`
records from the Parser output, and include their text in the context. References
are record-level context; a partial-record chunk does not necessarily contain every
anchor in that record. `raw_offset` still points into the original record's `raw_text`.

## Verification

```powershell
python -m pytest tests/test_textbook_footnotes.py
```

Unit tests do not require the private textbook. The integration test runs when
`data/raw/Fundamentals_of_Database_System.pdf` exists, or when `TEXTBOOK_TEST_PDF`
points to that file; otherwise it is skipped.
