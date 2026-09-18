import re
import json
from pathlib import Path

import fitz


class TextbookParser:
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(self.pdf_path)

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

    def extract_content_blocks(self, page):
        blocks = page.get_text("blocks")
        page_height = page.rect.height

        results = []

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

            if y0 < 50:
                continue

            if y1 > page_height - 40:
                continue

            raw_text = text
            cleaned_text = self.clean_text(raw_text)

            if not cleaned_text:
                continue

            results.append({
                "raw_text": raw_text,
                "text": cleaned_text,
                "bbox": (x0, y0, x1, y1)
            })

        return results
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

            cleaned_text = self.clean_text(text)

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

        return {
            "number": match.group(1),
            "title": match.group(2).strip()
        }

    def parse_pages(self, start_page, end_page):
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
                self.current_chapter = detected_chapter

            blocks = self.extract_content_blocks(page)

            for block in blocks:
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

                records.append({
                    "pdf_page": page_idx,

                    "chapter_number": (
                        self.current_chapter["number"]
                        if self.current_chapter
                        else None
                    ),

                    "chapter_title": (
                        self.current_chapter["title"]
                        if self.current_chapter
                        else None
                    ),

                    "section_number": (
                        self.current_section["number"]
                        if self.current_section
                        else None
                    ),

                    "section_title": (
                        self.current_section["title"]
                        if self.current_section
                        else "Chapter Introduction"
                    ),

                    "type": content_type,
                    "text": cleaned_text
                })

        return records

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