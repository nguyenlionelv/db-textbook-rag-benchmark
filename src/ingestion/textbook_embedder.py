import numpy as np
from sentence_transformers import SentenceTransformer


class E5Embedder:
    """Dùng cho multilingual E5 small/base/large với passage/query prefix.

    Nhận Document từ Chunker, trả về vector theo đúng thứ tự đầu vào.
    Caller chọn các chunk cần nhúng; class không lọc chunk hoặc sửa metadata.
    Khi cần tái lập kết quả, truyền revision cụ thể của model.
    local_files_only=True chỉ đọc model đã cache, không tải model từ mạng.
    """

    def __init__(
        self,
        model_name="intfloat/multilingual-e5-small",
        revision=None,
        device="cpu",
        batch_size=8,
        cache_folder=None,
        local_files_only=False,
    ):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size phải là số nguyên lớn hơn 0")

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name, revision=revision, device=device,
            cache_folder=cache_folder, local_files_only=local_files_only,
        )

    def embed_documents(self, chunks):
        """Trả về ma trận float32 (số chunk, số chiều), chuẩn hóa L2."""
        texts = [chunk.page_content for chunk in chunks]
        return self._encode(texts, prefix="passage: ")

    def embed_query(self, text):
        """Trả về một vector câu hỏi; chưa thực hiện tìm kiếm."""
        return self._encode([text], prefix="query: ")[0]

    def _encode(self, texts, prefix):
        if not texts:
            size = self.model.get_sentence_embedding_dimension()
            return np.empty((0, size), dtype=np.float32)

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"text[{index}] phải là chuỗi không rỗng")

        # Chỉ thêm prefix vào input model, giữ nguyên nội dung chunk gốc.
        inputs = [prefix + text for text in texts]
        for index, text in enumerate(inputs):
            count = len(self.model.tokenizer(text, truncation=False)["input_ids"])
            if count > self.model.max_seq_length:
                raise ValueError(
                    f"text[{index}] có {count} token (gồm prefix và special tokens), "
                    f"vượt giới hạn {self.model.max_seq_length}"
                )

        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            precision="float32",
            normalize_embeddings=True,
            convert_to_numpy=True,
            prompt="",
            show_progress_bar=False,
        )
