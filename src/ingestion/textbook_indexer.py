"""Tạo index E5/Qdrant từ JSONL Chunker, giữ ID và collection của notebook 03."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from langchain_core.documents import Document
from qdrant_client import models


def _write_manifest(path, profile):
    """Thay manifest bằng một file JSON hoàn chỉnh trên cùng filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(profile, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_chunks(raw):
    # Path.read_text trong notebook chuyển CRLF/CR sang LF trước khi tạo tên.
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            json.dumps(row, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Dòng {line_number}: JSON không hợp lệ") from error
        content = row.get("page_content") if isinstance(row, dict) else None
        metadata = row.get("metadata") if isinstance(row, dict) else None
        sources = metadata.get("sources") if isinstance(metadata, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Dòng {line_number}: page_content phải là chuỗi không rỗng")
        if not isinstance(sources, list):
            raise ValueError(f"Dòng {line_number}: metadata.sources phải là list")
        for source in sources:
            if (not isinstance(source, dict)
                    or not isinstance(source.get("record_id"), str)
                    or not source["record_id"].strip()
                    or type(source.get("pdf_page")) is not int or source["pdf_page"] < 0):
                raise ValueError(f"Dòng {line_number}: nguồn thiếu record_id/pdf_page hợp lệ")
        rows.append(row)
    if not any(row["metadata"]["sources"] for row in rows):
        raise ValueError("Không có chunk có nguồn để index")
    return text, rows


def build_qdrant_index(client, embedder, chunk_path, manifest_path, *, batch_size=64,
                       parser_path=None):
    """Embed/upsert và xác minh toàn bộ index; trả manifest khi status=complete.

    Caller quản lý client. Nhận E5Embedder với revision cố định. Point ID là vị
    trí record JSONL (bỏ dòng trắng), gồm cả các chunk không được chọn. Chỉ index
    chunk có sources như notebook 03; báo số chunk bị bỏ trong manifest.
    Đổi input/model/revision tạo collection khác, không xóa collection cũ.
    Chạy lại encode và upsert cùng ID; chưa có cache embedding hoặc resume.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size phải là int dương")
    if not isinstance(embedder.revision, str) or not embedder.revision.strip():
        raise ValueError("Cần revision cố định để định danh index")
    chunk_path, manifest_path = Path(chunk_path), Path(manifest_path)
    protected = {chunk_path.resolve()}
    if parser_path is not None:
        parser_path = Path(parser_path)
        protected.add(parser_path.resolve())
    if manifest_path.resolve() in protected:
        raise ValueError("manifest_path không được ghi đè file đầu vào")

    raw = chunk_path.read_bytes()
    text, rows = _read_chunks(raw)
    selected = [(i, row) for i, row in enumerate(rows) if row["metadata"]["sources"]]
    dimension = embedder.model.get_sentence_embedding_dimension()
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Số chiều embedding phải là int dương")
    # Preflight toàn corpus trước khi ghi batch đầu tiên, tránh cắt ngầm token.
    for point_id, row in selected:
        count = len(embedder.model.tokenizer(
            "passage: " + row["page_content"], truncation=False,
        )["input_ids"])
        if count > embedder.model.max_seq_length:
            raise ValueError(f"Chunk {point_id}: {count} token, vượt {embedder.model.max_seq_length}")

    key = embedder.model_name + embedder.revision + text
    collection = "e5_" + hashlib.sha256(key.encode()).hexdigest()[:12]
    profile = {
        "collection": collection, "model": embedder.model_name,
        "model_revision": embedder.revision, "dimension": dimension, "distance": "Cosine",
        "chunk_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_path": str(chunk_path.resolve()), "input_chunks": len(rows),
        "chunks": len(selected), "skipped_without_sources": len(rows) - len(selected),
        "selection_policy": "sources must be non-empty", "point_id_policy": "JSONL record index",
        "batch_size": batch_size, "upserted": 0, "status": "in_progress",
    }
    storage_path = client.init_options.get("path")
    if storage_path is not None:
        profile["qdrant_path"] = str(Path(storage_path).resolve())
    if parser_path is not None:
        profile["parser_sha256"] = hashlib.sha256(parser_path.read_bytes()).hexdigest()
        profile["parser_path"] = str(parser_path.resolve())

    if client.collection_exists(collection):
        config = client.get_collection(collection).config.params.vectors
        if (not isinstance(config, models.VectorParams) or config.size != dimension
                or config.distance != models.Distance.COSINE):
            raise ValueError("Collection đã có phải dùng vector cùng số chiều và cosine")
        # Không xóa point lạ trong collection đã có. Dừng trước khi upsert.
        expected_ids = {i for i, _ in selected}
        offset = None
        while True:
            points, offset = client.scroll(collection_name=collection, offset=offset,
                                           limit=batch_size, with_payload=False, with_vectors=False)
            if any(point.id not in expected_ids for point in points):
                raise ValueError("Collection có ID ngoài corpus; cần kiểm tra index trước khi ghi")
            if offset is None:
                break

    _write_manifest(manifest_path, profile)
    try:
        if not client.collection_exists(collection):
            client.create_collection(collection_name=collection, vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE,
            ))
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            chunks = [Document(page_content=row["page_content"], metadata=row["metadata"])
                      for _, row in batch]
            vectors = np.asarray(embedder.embed_documents(chunks), dtype=np.float32)
            if (vectors.shape != (len(batch), dimension) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
                raise ValueError(f"Batch bắt đầu tại chunk {batch[0][0]}: vector sai shape/giá trị/chuẩn L2")
            points = [models.PointStruct(id=i, vector=vector.tolist(), payload=row)
                      for (i, row), vector in zip(batch, vectors)]
            client.upsert(collection_name=collection, points=points, wait=True)
            saved = client.retrieve(collection_name=collection, ids=[p.id for p in points],
                                    with_payload=True, with_vectors=True)
            expected = {point.id: point for point in points}
            if len(saved) != len(points) or {p.id for p in saved} != set(expected):
                raise ValueError("Không đọc lại đủ ID đã upsert")
            for point in saved:
                original = expected[point.id]
                if (point.payload != original.payload or point.vector is None
                        or not np.allclose(point.vector, original.vector, atol=1e-6)):
                    raise ValueError(f"Point {point.id}: payload/vector đọc lại không khớp")
            profile["upserted"] += len(points)
            _write_manifest(manifest_path, profile)
        if client.count(collection_name=collection, exact=True).count != len(selected):
            raise ValueError("Count trong collection không khớp corpus")
        profile["status"] = "complete"
        _write_manifest(manifest_path, profile)
    except Exception as error:
        profile["status"] = "failed"
        profile["error"] = str(error)
        _write_manifest(manifest_path, profile)
        raise
    return profile
