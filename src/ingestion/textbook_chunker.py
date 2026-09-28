"""Chunk parsed textbook records into LangChain documents."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import groupby
import json
from math import isfinite
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
PARSER_FIELD_TYPES = {
    "id": (str,),
    "pdf_page": (int,),
    "chapter_number": (str, type(None)),
    "chapter_title": (str, type(None)),
    "section_number": (str, type(None)),
    "section_title": (str, type(None)),
    "type": (str,),
    "text": (str,),
    "raw_text": (str,),
    "bbox": (list, tuple),
    "metadata": (dict,),
}
CONTENT_TYPES = {
    "normal_text", "table_like", "figure_caption", "table_caption", "footnote",
}


class BaseChunker(ABC):
    """Base class dùng chung cho các chiến lược chunking.

    Input:
        ``records`` là danh sách dictionary do ``TextbookParser.parse_pages``
        trả về, với đủ schema trong ``docs/chunker_contract.md``. Đầu vào thuộc
        một PDF và giữ nguyên thứ tự đọc; chapter/section nằm ở top-level.

    Output:
        Danh sách ``Document``. Nội dung nằm trong ``page_content`` và thông tin
        nguồn nằm trong ``metadata``.
    """

    strategy: str

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        include_footnotes: bool = False,
    ) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size phải là số nguyên lớn hơn 0")
        if type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap phải là số nguyên trong [0, chunk_size)")
        if type(include_footnotes) is not bool:
            raise ValueError("include_footnotes phải là bool")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_footnotes = include_footnotes

    @abstractmethod
    def chunk(self, records: Sequence[Mapping[str, Any]]) -> list[Document]:
        """Chia các record thành những chunk có thể đưa sang bước embedding."""

    def _valid_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """Kiểm tra schema rồi lọc; giữ nguyên text và sao chép record nguồn."""
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("records phải là danh sách record từ Parser")

        valid, seen_ids = [], set()
        for index, record in enumerate(records):
            self._validate_record(record, index)
            if record["id"] in seen_ids:
                raise ValueError(f"record[{index}] id={record['id']!r}: id bị trùng")
            seen_ids.add(record["id"])

            # Chỉ dùng strip để kiểm tra rỗng, không thay đổi text nguồn.
            if not record["text"].strip():
                continue
            if record["type"] == "footnote" and not self.include_footnotes:
                continue
            if not record["chapter_number"] or not record["chapter_number"].strip():
                raise ValueError(
                    f"record[{index}] id={record['id']!r}: chapter_number bị thiếu"
                )
            valid.append(deepcopy(dict(record)))
        return valid

    @staticmethod
    def _validate_record(record: Mapping[str, Any], index: int) -> None:
        """Kiểm tra schema top-level; metadata nguồn được giữ nguyên như Parser."""
        if not isinstance(record, Mapping):
            raise ValueError(f"record[{index}]: cần một dictionary từ Parser")
        context = f"record[{index}] id={record.get('id')!r}"
        for field, allowed_types in PARSER_FIELD_TYPES.items():
            if field not in record:
                raise ValueError(f"{context}: thiếu trường {field}")
            # So sánh type để bool không bị nhận nhầm thành int.
            if type(record[field]) not in allowed_types:
                raise ValueError(f"{context}: sai kiểu dữ liệu của {field}")

        if not record["id"].strip():
            raise ValueError(f"{context}: id không được rỗng")
        if record["pdf_page"] < 0:
            raise ValueError(f"{context}: pdf_page phải lớn hơn hoặc bằng 0")
        if record["type"] not in CONTENT_TYPES:
            raise ValueError(f"{context}: type không thuộc các loại content của Parser")
        section_number = record["section_number"]
        if section_number is not None and not section_number.strip():
            raise ValueError(f"{context}: section_number phải là null hoặc chuỗi có nội dung")
        bbox = record["bbox"]
        if len(bbox) != 4 or any(
            type(value) not in (int, float) or not isfinite(value) for value in bbox
        ):
            raise ValueError(f"{context}: bbox phải chứa đúng 4 số hữu hạn")
        try:
            json.dumps(record["metadata"], allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context}: metadata phải serializable bằng JSON") from error

    @staticmethod
    def _consistent_title(
        records: Sequence[Mapping[str, Any]], field: str,
    ) -> str | None:
        """Lấy title hợp lệ duy nhất; bỏ giá trị null/rỗng, báo lỗi khi mâu thuẫn."""
        title = None
        for record in records:
            candidate = record[field]
            if candidate is None or not candidate.strip():
                continue
            if title is not None and candidate != title:
                raise ValueError(
                    f"record id={record['id']!r}: {field} mâu thuẫn trong cùng đơn vị"
                )
            title = candidate
        return title

    def _prepare_chapters(
        self, records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Chuẩn bị dữ liệu, chưa cắt chunk.

        Mỗi chapter có ``text``, ``metadata``, ``sections`` và ``source_spans``.
        Mỗi section có text riêng, metadata phân cấp và source_spans cục bộ.
        ``start``/``end`` của section và source_spans của chapter trỏ vào
        corpus chapter; source_spans của section trỏ vào text section.
        Mỗi source span giữ bản sao record và khoảng [start, end) của record.text.
        """
        valid_records = self._valid_records(records)
        chapters, seen_chapters = [], set()
        for chapter_number, chapter_group in groupby(
            valid_records, key=lambda record: record["chapter_number"],
        ):
            chapter_records = list(chapter_group)
            if chapter_number in seen_chapters:
                raise ValueError(
                    f"record id={chapter_records[0]['id']!r}: chapter_number "
                    f"{chapter_number!r} xuất hiện lại sau chapter khác"
                )
            seen_chapters.add(chapter_number)
            chapter_metadata = {
                "chapter_number": chapter_number,
                "chapter_title": self._consistent_title(chapter_records, "chapter_title"),
            }
            sections, chapter_sources, chapter_cursor = [], [], 0

            # groupby giữ các đoạn section liên tiếp, không gom lại rồi đảo thứ tự.
            for unit_index, (raw_number, section_group) in enumerate(groupby(
                chapter_records, key=lambda record: record["section_number"],
            )):
                section_records = list(section_group)
                section_number = raw_number if raw_number is not None else f"{chapter_number}.intro"
                section_title = self._consistent_title(section_records, "section_title")
                if section_title is None:
                    section_title = "Chapter Introduction" if raw_number is None else "Untitled Section"
                metadata = {
                    **chapter_metadata,
                    "section_id": self._section_id(chapter_number, raw_number),
                    "section_number": section_number,
                    "section_title": section_title,
                    "parent_section_id": self._parent_section_id(chapter_number, raw_number),
                    "section_unit_index": unit_index,
                }

                # Heading giống Experiment 1; chỉ record.text có record nguồn.
                heading = f"[[SECTION {section_number}: {section_title}]]\n"
                section_sources, section_cursor = [], len(heading)
                for record_index, record in enumerate(section_records):
                    if record_index:
                        section_cursor += 1  # Newline giữa hai record không thuộc nguồn.
                    end = section_cursor + len(record["text"])
                    section_sources.append({
                        "record": record, "start": section_cursor, "end": end,
                    })
                    section_cursor = end
                text = heading + "\n".join(record["text"] for record in section_records)

                if unit_index:
                    chapter_cursor += 2  # Hai newline giữa các đơn vị section.
                sections.append({
                    "text": text,
                    "metadata": metadata,
                    "source_spans": section_sources,
                    "start": chapter_cursor,
                    "end": chapter_cursor + len(text),
                })
                chapter_sources.extend({
                    **span,
                    "start": chapter_cursor + span["start"],
                    "end": chapter_cursor + span["end"],
                } for span in section_sources)
                chapter_cursor += len(text)

            chapters.append({
                "text": "\n\n".join(section["text"] for section in sections),
                "metadata": chapter_metadata,
                "sections": sections,
                "source_spans": chapter_sources,
            })
        return chapters

    def _make_chunk(
        self,
        chapter: dict[str, Any],
        start: int,
        end: int,
        chunk_index: int,
        section: dict[str, Any] | None = None,
    ) -> Document:
        """Tạo Document và ánh xạ phần text giao nhau về các record nguồn.

        Nếu truyền section, start/end là offset trong section đó; nếu không,
        chúng là offset trong corpus chapter. Heading/joiner không có nguồn.
        """
        unit = chapter if section is None else section
        sections = [section] if section is not None else [
            item for item in chapter["sections"]
            if start < item["end"] and end > item["start"]
        ]
        single_section = sections[0]["metadata"] if len(sections) == 1 else {}
        metadata = {
            **chapter["metadata"],
            "strategy": self.strategy,
            "chunk_index": chunk_index,
            "section_ids": list(dict.fromkeys(
                item["metadata"]["section_id"] for item in sections
            )),
            **{field: single_section.get(field) for field in (
                "section_id", "section_number", "section_title",
                "parent_section_id", "section_unit_index",
            )},
            "offset_scope": "chapter" if section is None else "section",
            "start_index": start,
            "end_index": end,
            "sources": [],
        }

        # Chỉ lấy record có text giao với chunk, không gắn toàn bộ section.
        for span in unit["source_spans"]:
            left, right = max(start, span["start"]), min(end, span["end"])
            if left >= right:
                continue
            record = span["record"]
            metadata["sources"].append({
                "record_id": record["id"],
                "pdf_page": record["pdf_page"],
                "type": record["type"],
                "bbox": list(record["bbox"]),
                "section_number": record["section_number"],
                "section_title": record["section_title"],
                "source_start_index": left - span["start"],
                "source_end_index": right - span["start"],
                "chunk_start_index": left - start,
                "chunk_end_index": right - start,
                # raw_offset của footnote_refs vẫn thuộc raw_text của record gốc.
                "metadata": deepcopy(record["metadata"]),
            })
        return Document(page_content=unit["text"][start:end], metadata=metadata)

    @staticmethod
    def _section_id(chapter_number: str, section_number: str | None) -> str:
        number = section_number if section_number is not None else f"{chapter_number}.intro"
        return f"sec_{number}"

    @staticmethod
    def _parent_section_id(chapter_number: str, section_number: str | None) -> str:
        if section_number is None or str(section_number).endswith(".intro"):
            return f"chapter_{chapter_number}"

        parts = str(section_number).split(".")
        if len(parts) <= 2:
            return f"chapter_{chapter_number}"
        return f"sec_{'.'.join(parts[:-1])}"


class StrictFixedSizeChunker(BaseChunker):
    """Cắt chuỗi theo cửa sổ ký tự cố định, không xét ranh giới câu/section.

    Input:
        Các parsed record của một PDF theo đúng thứ tự đọc, có thể gồm nhiều
        chapter. Section được nối thành corpus riêng cho từng chapter.

    Output:
        ``Document`` dài tối đa ``chunk_size``; các chunk liên tiếp trong cùng
        chapter overlap đúng cấu hình và có metadata truy vết nguồn.
    """

    strategy = "strict_fixed"

    def chunk(self, records: Sequence[Mapping[str, Any]]) -> list[Document]:
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for chapter in self._prepare_chapters(records):
            text = chapter["text"]
            for chunk_index, start in enumerate(range(0, len(text), step)):
                end = min(start + self.chunk_size, len(text))
                chunks.append(self._make_chunk(chapter, start, end, chunk_index))
                if end == len(text):
                    break
        return chunks


class GlobalRecursiveChunker(BaseChunker):
    """Chunk toàn corpus bằng RecursiveCharacterTextSplitter.

    Input:
        Các parsed record của một PDF theo thứ tự đọc. Split riêng từng chapter;
        section boundary chỉ được xem như separator ``\n\n`` trong corpus.

    Output:
        ``Document`` ưu tiên giữ paragraph/câu/từ, nhưng một chunk vẫn có thể
        chứa nội dung từ nhiều section; offset tính trong chapter.
    """

    strategy = "global_recursive"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        include_footnotes: bool = False,
        separators: Sequence[str] | None = None,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, include_footnotes)
        separators = DEFAULT_SEPARATORS if separators is None else separators
        if (
            not isinstance(separators, Sequence) or isinstance(separators, (str, bytes))
            or not separators or any(type(item) is not str for item in separators)
            or separators[-1] != ""
        ):
            raise ValueError("separators phải là dãy chuỗi kết thúc bằng '' để cắt theo ký tự")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
        )

    def chunk(self, records: Sequence[Mapping[str, Any]]) -> list[Document]:
        chunks = []
        for chapter in self._prepare_chapters(records):
            for chunk_index, (start, end) in enumerate(self._recursive_ranges(chapter["text"])):
                chunks.append(self._make_chunk(chapter, start, end, chunk_index))
        return chunks

    def _recursive_ranges(self, text: str) -> list[tuple[int, int]]:
        """Dùng splitter cho content, tìm offset theo thứ tự với overlap cho phép."""
        ranges, previous_start, previous_end = [], -1, 0
        for content in self.splitter.split_text(text):
            # Tiến về trước cả khi chunk ngắn/lặp lại, tránh gán về cùng vị trí.
            search_from = max(previous_start + 1, previous_end - self.chunk_overlap)
            start = text.find(content, search_from)
            if start < 0:
                raise ValueError("Không xác định được offset của chunk recursive trong text nguồn")
            end = start + len(content)
            ranges.append((start, end))
            previous_start, previous_end = start, end
        return ranges


class StructureAwareRecursiveChunker(GlobalRecursiveChunker):
    """Chia theo section trước, rồi recursive chunk bên trong từng section.

    Input:
        Các parsed record của một PDF, có chapter/section ở top-level. Record
        không có section dùng nhãn intro; section xuất hiện lại tạo đơn vị riêng.

    Output:
        ``Document`` luôn thuộc đúng một đơn vị section và giữ metadata nguồn;
        ``start_index``/``end_index`` là vị trí cục bộ trong đơn vị đó.
    """

    strategy = "structure_aware_recursive"

    def chunk(self, records: Sequence[Mapping[str, Any]]) -> list[Document]:
        chunks = []
        for chapter in self._prepare_chapters(records):
            chunk_index = 0
            for section in chapter["sections"]:
                for start, end in self._recursive_ranges(section["text"]):
                    chunks.append(self._make_chunk(chapter, start, end, chunk_index, section))
                    chunk_index += 1
        return chunks
