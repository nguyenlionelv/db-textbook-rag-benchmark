# TextbookRetriever và CrossEncoderReranker

Mã nguồn: [textbook_retrieval.py](../../src/retrieval/textbook_retrieval.py).

Luồng truy xuất là BM25 + dense search → Reciprocal Rank Fusion → cross-encoder reranking → top-k. BM25 và dense sử dụng cùng tập chunk trong collection Qdrant.

## Hàm hỗ trợ

| Hàm | Xử lý |
| --- | --- |
| `_tokenize()` | Chuyển chữ thường và tách token Unicode bằng biểu thức `\w+`. |
| `_validate_query()` | Yêu cầu query là chuỗi có nội dung. |
| `_id_key()` | Tạo khóa sắp xếp ổn định: ID số trước ID chuỗi. |

## Khởi tạo TextbookRetriever

`__init__()` nhận client, collection, embedder và reranker. Khi có `embedding_profile`, `_check_embedding_profile()` đối chiếu tên collection, model, revision, dimension và cosine. Profile có `status` phải mang giá trị `complete`.

Client `scroll()` đọc payload theo lô 256 point. `_validate_payload()` yêu cầu text có nội dung và từng nguồn có `record_id`, `pdf_page` hợp lệ. Payload được lưu vào `self.chunks` theo point ID; BM25 được dựng một lần từ nội dung đó.

`self.chunks` và BM25 là dữ liệu trong bộ nhớ tại thời điểm khởi tạo. Khi collection thay đổi, cần tạo lại retriever để đồng bộ tập tài liệu.

## Hai nhánh tìm kiếm

`_bm25_search()` tính điểm lexical trên các token của query. Chỉ các điểm lớn hơn 0 được giữ; kết quả sắp giảm dần theo điểm rồi theo ID.

`_dense_search()` gọi `embed_query()`, kiểm tra vector một chiều, hữu hạn và khác zero, rồi gửi `query_points()` tới Qdrant. Qdrant trả điểm tương đồng và ID; nội dung được lấy từ `self.chunks`.

Hai nhánh đều lấy tối đa `candidate_k` kết quả. Điểm BM25 và cosine không được cộng trực tiếp vì khác thang đo.

## Fusion bằng RRF

`_rrf()` cộng đóng góp theo point ID:

```text
RRF(d) = tổng 1 / (60 + rank(d)) trên các nhánh chứa d
```

Hạng bắt đầu từ 1. Chunk có mặt ở cả hai nhánh nhận hai đóng góp nhưng chỉ tạo một kết quả. Danh sách sau fusion được cắt còn `candidate_k` ứng viên.

## CrossEncoderReranker

`__init__()` nạp `CrossEncoder`; mặc định model là `BAAI/bge-reranker-v2-m3`, CPU, batch 2 và giới hạn 512 token cho một cặp.

`rerank(query, hits)` tạo từng cặp `[query, page_content]`. Tokenizer đếm cả cặp với `truncation=False`; cặp vượt giới hạn gây lỗi trước khi gọi model.

`predict()` dùng `torch.nn.Identity()` làm activation, nên điểm lưu là logit. Hàm yêu cầu một điểm hữu hạn cho mỗi hit, sao chép các hit và thay `score_type` thành `reranker`. Khi điểm bằng nhau, kết quả được sắp theo ID.

## retrieve()

`retrieve(query, top_k=5, candidate_k=20)` yêu cầu hai tham số dương và `candidate_k >= top_k`. Collection không có chunk trả danh sách rỗng.

Hàm lần lượt gọi hai nhánh, fusion, dựng hit có nội dung/metadata, rerank rồi lấy top-k. Kết quả trả về gồm:

| Trường | Ý nghĩa |
| --- | --- |
| `point_id` | ID point trong collection. |
| `page_content` | Nội dung chunk nguyên bản. |
| `metadata` | Metadata nguồn được sao chép. |
| `score` | Điểm sau reranking. |
| `score_type` | Loại điểm, bằng `reranker` ở đầu ra cuối. |

Điểm reranker biểu thị mức liên quan để xếp hạng, không phải xác suất đáp án đúng. Hàm không tự đặt ngưỡng từ chối và không tạo câu trả lời.

Caller đóng Qdrant client. [Generation](textbook_generation.md) nhận trực tiếp danh sách hit này; [output retrieval](../output_docs/retrieval.md) mô tả các file xếp hạng và metric.
