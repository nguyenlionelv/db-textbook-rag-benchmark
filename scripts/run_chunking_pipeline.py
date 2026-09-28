"""Pipeline thử đơn giản: JSONL từ Parser -> ba Chunker -> JSONL và mẫu in ra.

Chạy từ project root: python scripts/run_chunking_pipeline.py
Đổi các hằng số bên dưới để thử input hoặc kích thước chunk khác.
"""

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingestion.textbook_chunker import (  # noqa: E402
    StrictFixedSizeChunker,
    GlobalRecursiveChunker,
    StructureAwareRecursiveChunker,
)


INPUT_PATH = ROOT / "output" / "parsed_chapter1.jsonl"
OUTPUT_DIR = ROOT / "output" / "chunks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SAMPLE_COUNT = 1


def main():
    # Hiển thị đúng dấu tiếng Việt cả khi stdout được chuyển sang pipe trên Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    # 1. Nạp output Parser, giữ nguyên thứ tự record. Không cần đọc lại PDF.
    with INPUT_PATH.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    records_by_id = {record["id"]: record for record in records}
    print("Input:", INPUT_PATH)
    print("Số record:", len(records))
    print("Record theo chapter:", dict(Counter(record["chapter_number"] for record in records)))
    if records:
        print("\nMột record đầu vào:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for chunker_class in (StrictFixedSizeChunker, GlobalRecursiveChunker, StructureAwareRecursiveChunker):
        # 2. Truyền toàn bộ record cho Chunker; từng chapter được xử lý riêng.
        chunker = chunker_class(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = chunker.chunk(records)

        # 3. Kiểm tra nhanh kích thước, chapter và từng phần text nguồn.
        for chunk in chunks:
            metadata = chunk.metadata
            assert 0 < len(chunk.page_content) <= CHUNK_SIZE
            assert metadata["end_index"] - metadata["start_index"] == len(chunk.page_content)
            if metadata["offset_scope"] == "section":
                assert len(metadata["section_ids"]) == 1
            for source in metadata["sources"]:
                original = records_by_id[source["record_id"]]
                assert original["chapter_number"] == metadata["chapter_number"]
                assert original["text"][source["source_start_index"]:source["source_end_index"]] == (
                    chunk.page_content[source["chunk_start_index"]:source["chunk_end_index"]]
                )

        # 4. Lưu đầy đủ output và in mẫu gồm cả nội dung lẫn metadata.
        output_path = OUTPUT_DIR / f"{chunker.strategy}.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                item = {"page_content": chunk.page_content, "metadata": chunk.metadata}
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        print("\n" + "=" * 70)
        print(chunker_class.__name__, "-", len(chunks), "chunks")
        print("Chunk theo chapter:", dict(Counter(chunk.metadata["chapter_number"] for chunk in chunks)))
        print("Chunk trộn section:", sum(len(chunk.metadata["section_ids"]) > 1 for chunk in chunks))
        print("Kiểm tra nguồn: OK")
        print("Output:", output_path)
        for chunk in chunks[:SAMPLE_COUNT]:
            print(json.dumps({"page_content": chunk.page_content, "metadata": chunk.metadata},
                             ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
