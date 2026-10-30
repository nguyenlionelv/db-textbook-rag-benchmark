import re
import json
import hashlib
from collections import Counter
from pathlib import Path
from statistics import median

import pymupdf as fitz


class TextbookParser:
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(self.pdf_path)
        # Content-based IDs remain stable when the PDF is renamed or moved.
        with self.pdf_path.open("rb") as stream:
            self.document_id = hashlib.file_digest(stream, "sha256").hexdigest()[:16]

        self.current_chapter = None
        self.current_section = None

        self.chapter_pattern = re.compile(
            r"\bChapter\s+(\d+)\s+(.+)$",
            re.IGNORECASE
        )

        self.section_pattern = re.compile(
            r"^(\d+\.\d+(?:\.\d+)*)\s+(.+)$"
        )

    def clean_text(self, text):
        # Remove hyphenation
        text = re.sub(r"-\n(?=[a-z])", "", text)

        # Remove line breaks and extra whitespace
        text = text.replace("\n", " ")

        # Collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def classify_block(self, raw_text):
        text = raw_text.strip()

        lines = [
            line.strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]
        # Figure caption
        if re.match(
            r"^Figure\s+\d+\.\d+",
            text,
            re.IGNORECASE
        ):
            return "figure_caption"
        # Table caption
        if re.match(
            r"^Table\s+\d+\.\d+",
            text,
            re.IGNORECASE
        ):
            return "table_caption"
        # Table-like content
        if len(lines) >= 2:
            short_lines = sum(
                len(line) < 30
                for line in lines
            )

            numeric_lines = sum(
                any(char.isdigit() for char in line)
                for line in lines
            )

            short_ratio = short_lines / len(lines)
            numeric_ratio = numeric_lines / len(lines)

            if (
                short_ratio > 0.7
                and numeric_ratio > 0.3
            ):
                return "table_like"
        # Normal text
        return "normal_text"

    @staticmethod
    def _dominant_font(spans):
        weights = Counter()
        for span in spans:
            weight = len(span["text"].strip())
            if weight:
                weights[round(span["size"], 2)] += weight
        return weights.most_common(1)[0][0] if weights else 0

    def _layout_blocks(self, page):
        """Keep span geometry and offsets before cleaning changes the text."""
        results = []
        for index, block in enumerate(page.get_text("dict")["blocks"]):
            if block["type"] != 0:
                continue
            parts, lines, spans = [], [], []
            offset = 0
            for line in block["lines"]:
                line_spans = []
                for source_span in line["spans"]:
                    span = dict(source_span)
                    span["raw_offset"] = offset
                    parts.append(span["text"])
                    offset += len(span["text"])
                    spans.append(span)
                    line_spans.append(span)
                parts.append("\n")
                offset += 1
                lines.append(line_spans)
            raw_text = "".join(parts)
            if not raw_text.strip():
                continue
            results.append({
                "block_number": block.get("number", index),
                "raw_text": raw_text,
                "text": self.clean_text(raw_text),
                "bbox": tuple(block["bbox"]),
                "spans": spans,
                "lines": lines,
                "font_size": self._dominant_font(spans),
            })
        return results

    def _reference_candidates(self, block):
        candidates = []
        for spans in block["lines"]:
            line_font = self._dominant_font(spans)
            if not line_font:
                continue
            baseline = median(
                span["origin"][1] for span in spans
                if span["size"] >= line_font * 0.95
            )
            line_text = "".join(span["text"] for span in spans)
            # Short mathematical expressions and numeric table rows are not prose.
            if (
                len(re.findall(r"[^\W\d_]+", line_text)) < 3
                or any(operator in line_text for operator in "=+^×÷∑∫")
            ):
                continue
            for span in spans:
                match = re.fullmatch(r"\s*(\d{1,3})\s*", span["text"])
                if not match or span["size"] > line_font * 0.85:
                    continue
                raised = baseline - span["origin"][1] > line_font * 0.15
                if not (span["flags"] & 1 or raised):
                    continue
                start = span["raw_offset"] + match.start(1)
                prefix = block["raw_text"][:start].rstrip()
                if not prefix or not (prefix[-1].isalpha() or prefix[-1] in ".,;:!?)]"):
                    continue
                last_word = re.search(r"([^\W\d_]+)$", prefix)
                if last_word and len(last_word.group(1)) == 1:
                    # Be conservative around variable powers such as x² in prose.
                    continue
                candidates.append({
                    "marker": match.group(1),
                    "raw_offset": start,
                    "raw_end": span["raw_offset"] + match.end(1),
                    "bbox": tuple(span["bbox"]),
                    "anchor_text": self.clean_text(prefix)[-160:],
                })
        return candidates

    def _footnote_candidate(self, block, body_font, page_height, referenced_markers):
        if not body_font or block["bbox"][1] < page_height * 0.70:
            return None
        if block["font_size"] > body_font * 0.90:
            return None
        match = re.match(r"\s*(\d{1,3})(?!\d)", block["raw_text"])
        if not match or not re.search(r"[^\W\d_]", block["raw_text"]):
            return None
        first_span = next(span for span in block["spans"] if span["text"].strip())
        separate_marker = (
            first_span["text"].strip() == match.group(1)
            and first_span["size"] < block["font_size"] * 0.9
        )
        if not separate_marker:
            # Without a separate small marker, require a matching prose reference
            # and reject decimal section numbers or numeric table entries.
            if (
                match.group(1) not in referenced_markers
                or not re.match(r"\s|[^\W\d_]", block["raw_text"][match.end(1):])
                or len(re.findall(r"[^\W\d_]+", block["raw_text"])) < 3
            ):
                return None
        return {
            "marker": match.group(1),
            "marker_end": match.end(1),
            "recognition_basis": "layout_and_marker" if match.group(1) in referenced_markers else "layout_only",
        }

    def extract_content_blocks(self, page):
        blocks = self._layout_blocks(page)
        page_height = page.rect.height
        body_spans = [
            span for block in blocks
            if 50 <= block["bbox"][1] < page_height * 0.75
            for span in block["spans"]
        ]
        body_font = self._dominant_font(body_spans)
        referenced_markers = set()
        for block in blocks:
            block["reference_candidates"] = self._reference_candidates(block)
            referenced_markers.update(ref["marker"] for ref in block["reference_candidates"])
        for block in blocks:
            block["footnote"] = self._footnote_candidate(
                block, body_font, page_height, referenced_markers
            )

        # Join nearby, small-font continuation blocks in the same column.
        consumed = set()
        ordered = sorted(blocks, key=lambda block: (block["bbox"][1], block["bbox"][0]))
        for index, block in enumerate(ordered):
            if not block["footnote"] or block["block_number"] in consumed:
                continue
            block["source_block_numbers"] = [block["block_number"]]
            for following in ordered[index + 1:]:
                if following["footnote"]:
                    break
                gap = following["bbox"][1] - block["bbox"][3]
                aligned = abs(following["bbox"][0] - block["bbox"][0]) <= 12
                if not (
                    -2 <= gap <= max(16, block["font_size"] * 2)
                    and aligned
                    and following["font_size"] <= body_font * 0.90
                    and following["font_size"] <= block["font_size"] * 1.05
                    and len(re.findall(r"[^\W\d_]+", following["raw_text"])) >= 3
                ):
                    break
                block["raw_text"] += following["raw_text"]
                block["text"] = self.clean_text(block["raw_text"])
                block["bbox"] = (
                    min(block["bbox"][0], following["bbox"][0]),
                    min(block["bbox"][1], following["bbox"][1]),
                    max(block["bbox"][2], following["bbox"][2]),
                    max(block["bbox"][3], following["bbox"][3]),
                )
                block["source_block_numbers"].append(following["block_number"])
                consumed.add(following["block_number"])

        return [
            block for block in blocks
            if block["block_number"] not in consumed
            and block["bbox"][1] >= 50
            # Detect notes before excluding page footers.
            and (block["footnote"] or block["bbox"][3] <= page_height - 40)
        ]
    # Get the chapter number and title from the page header
    def detect_chapter_from_page(self, page):
        blocks = page.get_text("blocks")

        for block in blocks:
            (
                x0,
                y0,
                x1,
                y1,
                text,
                block_no,
                block_type
            ) = block

            if block_type != 0:
                continue

            # chỉ xét vùng header
            if y0 >= 50:
                continue

            # A running header may contain a separate line with the page number.
            header_lines = [line for line in text.splitlines() if not line.strip().isdigit()]
            cleaned_text = self.clean_text("\n".join(header_lines))

            match = self.chapter_pattern.search(
                cleaned_text
            )

            if match:
                return {
                    "number": match.group(1),
                    "title": match.group(2).strip()
                }

        return None

    def detect_section(self, text):
        match = self.section_pattern.match(text)

        if not match:
            return None

        # Decimal table cells such as "10.0 40.0" are not section headings.
        if not any(char.isalpha() for char in match.group(2)):
            return None
        if self.current_chapter and match.group(1).split(".")[0] != self.current_chapter["number"]:
            return None

        return {
            "number": match.group(1),
            "title": match.group(2).strip()
        }

    def parse_pages(self, start_page, end_page):
        """Parse an inclusive range; link numbered footnotes on the same page."""
        if not 0 <= start_page <= end_page < len(self.doc):
            raise ValueError("Expected 0 <= start_page <= end_page < PDF page count")
        records = []

        self.current_chapter = None
        self.current_section = None

        for page_idx in range(
            start_page,
            end_page + 1
        ):
            page = self.doc[page_idx]

            detected_chapter = (
                self.detect_chapter_from_page(page)
            )

            if detected_chapter is not None:
                if self.current_chapter and detected_chapter["number"] != self.current_chapter["number"]:
                    self.current_section = None
                self.current_chapter = detected_chapter

            blocks = self.extract_content_blocks(page)
            footnotes = []
            footnotes_by_marker = {}
            for block in blocks:
                if not block["footnote"]:
                    continue
                note = block["footnote"]
                record = self._make_record(page_idx, block, "footnote")
                record["id"] = f"{self.document_id}:p{page_idx}:fn{note['marker']}:b{block['block_number']}"
                record["text"] = self.clean_text(block["raw_text"][note["marker_end"]:])
                # A note's section comes from its references, not the last heading on the page.
                record["section_number"] = None
                record["section_title"] = None
                record["metadata"] = {
                    "marker": note["marker"],
                    "referenced_by": [],
                    "reference_contexts": [],
                    "status": "unreferenced",
                    "recognition_basis": note["recognition_basis"],
                    "source_block_numbers": block["source_block_numbers"],
                }
                footnotes.append(record)
                footnotes_by_marker.setdefault(note["marker"], []).append(record)

            for block in blocks:
                if block["footnote"]:
                    continue
                raw_text = block["raw_text"]
                cleaned_text = block["text"]

                detected_section = (
                    self.detect_section(cleaned_text)
                )

                if detected_section is not None:
                    self.current_section = (
                        detected_section
                    )
                    continue

                content_type = self.classify_block(
                    raw_text
                )

                record = self._make_record(page_idx, block, content_type)
                refs = []
                resolved = []
                for candidate in block["reference_candidates"]:
                    targets = footnotes_by_marker.get(candidate["marker"], [])
                    status = "resolved" if len(targets) == 1 else "ambiguous" if targets else "unresolved"
                    refs.append({
                        key: value for key, value in candidate.items() if key != "raw_end"
                    } | {
                        "footnote_id": targets[0]["id"] if status == "resolved" else None,
                        "status": status,
                    })
                    if status == "resolved":
                        resolved.append(candidate)
                        note_record = targets[0]
                        metadata = note_record["metadata"]
                        if record["id"] not in metadata["referenced_by"]:
                            metadata["referenced_by"].append(record["id"])
                            metadata["reference_contexts"].append({
                                "record_id": record["id"],
                                "chapter_number": record["chapter_number"],
                                "chapter_title": record["chapter_title"],
                                "section_number": record["section_number"],
                                "section_title": record["section_title"],
                            })
                        metadata["status"] = "resolved"
                record["metadata"]["footnote_refs"] = refs
                record["text"] = self._remove_resolved_markers(raw_text, resolved)
                records.append(record)

            for note_record in footnotes:
                contexts = note_record["metadata"]["reference_contexts"]
                if contexts:
                    for field in ("chapter_number", "chapter_title", "section_number", "section_title"):
                        values = {context[field] for context in contexts}
                        note_record[field] = next(iter(values)) if len(values) == 1 else None
                records.append(note_record)

        return records

    def _make_record(self, page_idx, block, content_type):
        return {
            "id": f"{self.document_id}:p{page_idx}:b{block['block_number']}",
            "pdf_page": page_idx,
            "chapter_number": self.current_chapter["number"] if self.current_chapter else None,
            "chapter_title": self.current_chapter["title"] if self.current_chapter else None,
            "section_number": self.current_section["number"] if self.current_section else None,
            "section_title": self.current_section["title"] if self.current_section else "Chapter Introduction",
            "type": content_type,
            "text": block["text"],
            "raw_text": block["raw_text"],
            "bbox": block["bbox"],
            "metadata": {"footnote_refs": []},
        }

    def _remove_resolved_markers(self, raw_text, candidates):
        text = raw_text
        for candidate in sorted(candidates, key=lambda item: item["raw_offset"], reverse=True):
            start, end = candidate["raw_offset"], candidate["raw_end"]
            # Preserve a word boundary when a PDF omitted surrounding spaces.
            space = " " if start and end < len(text) and not text[start - 1].isspace() and not text[end].isspace() else ""
            text = text[:start] + space + text[end:]
        return self.clean_text(text)

    def export_jsonl(self, records, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

    def close(self):
        self.doc.close()
