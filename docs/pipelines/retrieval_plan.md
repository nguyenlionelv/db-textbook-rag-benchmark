# Retrieval — phạm vi gọn đã thống nhất

Plan này thay cho kiến trúc nhiều lớp ban đầu. Mục tiêu là một pipeline Hybrid Retrieval dùng được ngoài notebook, có nguồn và kiểm chứng được trên corpus mẫu hiện tại.

## 1. Pipeline và kiến trúc

**Query Processing → BM25 + E5/Qdrant → RRF giữ 20 ứng viên → CrossEncoder reranker → top-5 có nguồn.**

Một file `src/retrieval/textbook_retrieval.py`, hai class:

- `TextbookRetriever`: đọc payload một lần, chuẩn bị BM25, tìm kiếm và gộp RRF, gọi reranker.
- `CrossEncoderReranker`: load model một lần, kiểm tra token của cặp query–chunk, chấm và sắp xếp lại.

API chính:

```python
retriever = TextbookRetriever(client, collection_name, embedder, reranker,
                             embedding_profile=profile)
hits = retriever.retrieve(query, top_k=5, candidate_k=20)
```

Kết quả là `list[dict]` với `point_id`, `page_content`, `metadata`, `score`, `score_type`. Caller quản lý client và biết collection đang dùng. Nguồn và offsets được giữ nguyên. Điểm reranker là logit, không phải xác suất đúng.

## 2. Cấu hình và điều kiện đầu vào

- Corpus mẫu hiện tại: 233 chunk có nguồn từ PDF index 34–100, không đại diện toàn bộ sách.
- E5: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Qdrant Local, vector 384 chiều, cosine; BM25 và dense dùng cùng collection.
- BM25: `k1=1.5`, `b=0.75`, `epsilon=0.25`; tokenizer Unicode `\w+` + lowercase; chỉ lấy điểm dương.
- RRF: `1 / (60 + rank)`, rank bắt đầu từ 1, hai nhánh trọng số bằng nhau.
- Reranker: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, CPU, batch 2, giới hạn 512 token/cặp.
- Vượt giới hạn token thì báo lỗi trước inference, không cắt ngầm. Cần đổi giới hạn có chủ đích nếu thử câu hỏi/chunk dài hơn.
- Khi đồng điểm: ID số tăng dần; ID chuỗi theo thứ tự chuỗi, sau ID số.

Khi khởi tạo, class kiểm tra nội dung không rỗng và mỗi nguồn có `record_id`, `pdf_page` hợp lệ. Nếu truyền `embedding_profile` từ manifest, class đối chiếu collection, model, revision, số chiều và cosine.

Notebook/script kiểm tra payload khớp corpus. Script còn kiểm tra hash corpus và parser. Manifest phải đến từ index đã xác minh; cùng số chiều không chứng minh hai model tương đương. Khởi tạo lại retriever nếu collection thay đổi.

## 3. Notebook và kiểm thử

- Notebook 05 giữ kết quả đối chứng BM25/dense/hybrid đã chạy trên 32 câu; không tuyên bố R4/R5 đã chạy trong notebook này.
- Notebook 06 giữ các bước ngắn để học, thêm kiểm tra token và ví dụ gọi class; đối chiếu ID, điểm, nội dung, metadata với pipeline từng bước.
- Unit test dùng model giả: BM25, RRF, đồng điểm, token, input, payload, profile và điểm model.
- Integration test dùng Qdrant thật trong bộ nhớ và model giả: toàn pipeline, nguồn, giới hạn ứng viên, corpus rỗng.
- Chạy model thật riêng qua notebook 06 và script đánh giá. Unit test không tải model lớn.

## 4. Đánh giá vừa đủ

Chạy `python scripts/evaluate_retrieval.py` từ thư mục dự án. Lần đầu cần Internet để tải model; reranker lưu tại `output/model_cache/reranker_hub/`.

Dùng lại 32 câu đã có (16 dev, 16 eval, mỗi nhóm gồm Anh và Việt). Không chọn cấu hình mới bằng kết quả eval. So sánh Hybrid trước và sau reranking, cố định candidate-k 20 và top-k 5.

- Hit@5, MRR@5: bằng chứng là `record_id` và khoảng ký tự đã gán; không gọi đây là Recall đầy đủ.
- Candidate Hit@20: phân biệt mất bằng chứng ở bước tìm ứng viên với lỗi xếp hạng.
- Thời gian: một warm-up, một lượt mỗi câu/phương pháp, gồm query embedding nhưng không gồm load model/index. Chỉ báo thời gian quan sát trên máy hiện tại.
- Lưu `per_query.csv`, `summary.csv`, `topk.jsonl`, `manifest.json`, `report.md` vào `output/retrieval/hybrid_reranking/`, tách khỏi kết quả notebook 05.

Lượt kiểm chứng notebook dùng bản sao SQLite nhất quán của index đang được mở; toàn bộ payload đã đối chiếu với corpus.

## 5. Tiêu chí hoàn thành

- [x] Hai class triển khai pipeline và giữ nguồn.
- [x] Token quá dài báo lỗi; kiểm tra dữ liệu và profile khi chuẩn bị.
- [x] 49 test logic, metric evidence và integration với Qdrant trong bộ nhớ đạt.
- [x] Notebook và class khớp khi chạy model thật (top-k, điểm, nội dung, nguồn).
- [x] Đã đánh giá Hybrid trước/sau reranking trên 32 câu và ghi kết quả.
- [x] Plan và README mô tả phạm vi gọn, cách chạy và giới hạn.

Không cần thêm BaseRetriever, storage wrapper, mỗi phương pháp một class, evaluator framework hoặc CLI nhiều chế độ. Filter chương/section chỉ bổ sung khi app cần. Generation, app và deploy thuộc bước tiếp theo.

Lý thuyết: [RetrievalMaterials.md](../materials/RetrievalMaterials.md).

## Kết quả lượt chạy đã xác minh

49 test đạt; notebook/class khớp với model thật. Baseline Hybrid khớp kết quả notebook 05. Trên 8 câu eval tiếng Việt, Hit@5 tăng từ 0,50 lên 0,875 và MRR@5 từ 0,344 lên 0,875. Reranker chạy khoảng 23–27 giây/câu trung bình theo nhóm trên CPU hiện tại. Đây là kết quả pilot, cần cân nhắc tốc độ khi làm app.

Báo cáo: [report.md](../../output/retrieval/hybrid_reranking/report.md).
