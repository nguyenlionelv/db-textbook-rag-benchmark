"""Hybrid retrieval trên collection Qdrant có payload page_content và metadata."""

import re
from copy import deepcopy

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _validate_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")


def _id_key(point_id):
    # ID số sắp theo giá trị số; UUID sắp theo chuỗi, sau ID số.
    return (isinstance(point_id, str), point_id)


class CrossEncoderReranker:
    """Chấm lại các cặp query–chunk; báo lỗi nếu vượt max_length.

    Điểm trả về là logit, không phải xác suất. Không sửa hits đầu vào.
    """

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        revision=None,
        device="cpu",
        batch_size=2,
        max_length=512,
        cache_folder=None,
    ):
        for name, value in (("batch_size", batch_size), ("max_length", max_length)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} phải là int dương")
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name, revision=revision, device=device, max_length=max_length,
            cache_folder=cache_folder,
        )

    def rerank(self, query, hits):
        """Trả list dict đã sắp xếp, thay score và score_type bằng điểm reranker."""
        _validate_query(query)
        if not hits:
            return []
        pairs = [[query, hit["page_content"]] for hit in hits]
        for index, (question, passage) in enumerate(pairs):
            tokens = self.model.tokenizer(question, passage, truncation=False)["input_ids"]
            if len(tokens) > self.max_length:
                raise ValueError(
                    f"Cặp query–chunk[{index}] có {len(tokens)} token, "
                    f"vượt giới hạn {self.max_length}"
                )
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
        )
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(hits) or not np.isfinite(scores).all():
            raise ValueError("Reranker phải trả một điểm hữu hạn cho mỗi chunk")
        results = [
            {**deepcopy(hit), "score": float(score), "score_type": "reranker"}
            for hit, score in zip(hits, scores)
        ]
        return sorted(results, key=lambda hit: (-hit["score"], _id_key(hit["point_id"])))


class TextbookRetriever:
    """BM25 + dense → RRF → reranker → top-k.

    Đọc toàn bộ payload và chuẩn bị BM25 một lần. Khởi tạo lại nếu collection
    thay đổi; dùng embedder khớp model/revision đã tạo vector tài liệu.
    Truyền embedding_profile từ manifest để kiểm tra cấu hình lúc khởi tạo.
    Caller quản lý và đóng client. Không tạo hoặc sửa index.
    """

    def __init__(self, client, collection_name, embedder, reranker, embedding_profile=None):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.reranker = reranker
        if embedding_profile is not None:
            self._check_embedding_profile(embedding_profile)
        self.chunks = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name, offset=offset, limit=256,
                with_payload=True, with_vectors=False,
            )
            for point in points:
                self._validate_payload(point.id, point.payload)
                self.chunks[point.id] = point.payload
            if offset is None:
                break
        self.point_ids = list(self.chunks)
        tokens = [_tokenize(self.chunks[i]["page_content"]) for i in self.point_ids]
        self.bm25 = BM25Okapi(tokens) if any(tokens) else None

    def _check_embedding_profile(self, profile):
        """Đối chiếu manifest đã xác minh khi tạo/thực nghiệm index."""
        if "status" in profile and profile["status"] != "complete":
            raise ValueError("Index trong embedding_profile chưa hoàn tất")
        if (profile["collection"] != self.collection_name
                or profile["model"] != self.embedder.model_name
                or profile["model_revision"] != self.embedder.revision):
            raise ValueError("Collection/model/revision không khớp embedding_profile")
        config = self.client.get_collection(self.collection_name).config.params.vectors
        dimension = self.embedder.model.get_sentence_embedding_dimension()
        if config.size != dimension or config.distance != "Cosine":
            raise ValueError("Index phải dùng cosine và cùng số chiều với embedder")

    def _validate_payload(self, point_id, payload):
        text = payload.get("page_content") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Point {point_id}: page_content phải là chuỗi không rỗng")
        metadata = payload.get("metadata")
        sources = metadata.get("sources") if isinstance(metadata, dict) else None
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Point {point_id}: metadata.sources phải có nguồn")
        for source in sources:
            if (not isinstance(source, dict) or not isinstance(source.get("record_id"), str)
                    or not source["record_id"].strip()
                    or type(source.get("pdf_page")) is not int or source["pdf_page"] < 0):
                raise ValueError(f"Point {point_id}: nguồn phải có record_id và pdf_page hợp lệ")

    def retrieve(self, query, top_k=5, candidate_k=20):
        """Trả tối đa top_k chunk kèm point_id, nội dung, metadata và điểm cuối."""
        _validate_query(query)
        for name, value in (("top_k", top_k), ("candidate_k", candidate_k)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} phải là int dương")
        if candidate_k < top_k:
            raise ValueError("candidate_k phải >= top_k")
        if not self.chunks:
            return []

        lexical = self._bm25_search(query, candidate_k)
        dense = self._dense_search(query, candidate_k)
        candidates = self._rrf(lexical, dense, candidate_k)
        hits = [
            {"point_id": i, "page_content": self.chunks[i]["page_content"],
             "metadata": deepcopy(self.chunks[i]["metadata"]),
             "score": score, "score_type": "rrf"}
            for i, score in candidates
        ]
        return self.reranker.rerank(query, hits)[:top_k]

    def _bm25_search(self, query, limit):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        hits = [(i, float(score)) for i, score in zip(self.point_ids, scores) if score > 0]
        return sorted(hits, key=lambda hit: (-hit[1], _id_key(hit[0])))[:limit]

    def _dense_search(self, query, limit):
        vector = np.asarray(self.embedder.embed_query(query))
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Vector query phải là vector hữu hạn, khác zero")
        response = self.client.query_points(
            collection_name=self.collection_name, query=vector.tolist(),
            limit=limit, with_payload=False,
        )
        hits = [(point.id, point.score) for point in response.points]
        return sorted(hits, key=lambda hit: (-hit[1], _id_key(hit[0])))

    def _rrf(self, lexical_hits, dense_hits, limit):
        # Cộng theo ID để chunk xuất hiện ở cả hai nhánh chỉ có một kết quả.
        scores = {}
        for hits in (lexical_hits, dense_hits):
            for rank, (point_id, _) in enumerate(hits, 1):
                scores[point_id] = scores.get(point_id, 0) + 1 / (60 + rank)
        return sorted(scores.items(), key=lambda hit: (-hit[1], _id_key(hit[0])))[:limit]
