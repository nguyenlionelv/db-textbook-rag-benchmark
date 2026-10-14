"""JSONL Chunker -> E5Embedder -> Qdrant Local -> manifest đã xác minh.

Chạy từ project root: python scripts/run_indexing_pipeline.py
Đổi các hằng số để dùng corpus khác; --help liệt kê tùy chọn cache và đường dẫn.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingestion.textbook_embedder import E5Embedder
from src.ingestion.textbook_indexer import build_qdrant_index

CHUNK_PATH = ROOT / "output/chunks/structure_aware_recursive.jsonl"
PARSER_PATH = ROOT / "output/parsed_chapter1.jsonl"
DB_PATH = ROOT / "output/qdrant_db"
MANIFEST_PATH = ROOT / "output/index/manifest.json"
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
INDEX_BATCH_SIZE = 64


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Chỉ dùng model đã cache, không tải model")
    parser.add_argument("--model-cache", help="Thư mục Hugging Face Hub cache đã có model E5")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Thư mục Qdrant Local")
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH, help="File manifest đầu ra")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        client = QdrantClient(path=str(args.db_path))
    except RuntimeError as error:
        if "already accessed by another instance" in str(error):
            parser.exit(2, f"Qdrant đang khóa {args.db_path}. Đóng client đang dùng hoặc chọn "
                        "--db-path và --manifest-path riêng.\n")
        raise
    try:
        embedder = E5Embedder(model_name=MODEL_NAME, revision=MODEL_REVISION, batch_size=8,
                              cache_folder=args.model_cache, local_files_only=args.offline)
        profile = build_qdrant_index(client, embedder, CHUNK_PATH, args.manifest_path,
                                     parser_path=PARSER_PATH, batch_size=INDEX_BATCH_SIZE)
    finally:
        client.close()
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("Manifest:", args.manifest_path.resolve())


if __name__ == "__main__":
    main()
