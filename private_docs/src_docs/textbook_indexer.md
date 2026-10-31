# TextbookIndexer

Mã nguồn: [textbook_indexer.py](../../src/ingestion/textbook_indexer.py).

`build_qdrant_index()` đọc JSONL Chunker, tạo embedding, ghi point vào Qdrant và trả manifest của index. Caller cung cấp `client` và `embedder`, đồng thời quản lý việc đóng client.

## Đọc và kiểm tra chunk

`_read_chunks(raw)` giải mã UTF-8, chuyển CRLF/CR về LF và đọc các dòng JSON không rỗng. Mỗi dòng phải có `page_content` không rỗng và `metadata.sources` là danh sách.

Mỗi nguồn cần `record_id` có nội dung và `pdf_page` là số nguyên không âm. `allow_nan=False` loại các giá trị JSON không hợp lệ cho dữ liệu lưu.

Các chunk có `sources=[]` vẫn nằm trong danh sách dòng, nhưng không được chọn để encode. Point ID là vị trí record JSONL sau khi bỏ dòng trắng, không phải số thứ tự sau khi lọc nguồn.

## Kiểm tra trước khi ghi

- `batch_size` phải là số nguyên dương; embedder phải có revision cố định.
- `manifest_path` không được trùng file chunk hoặc Parser.
- Số chiều embedding phải là số nguyên dương.
- Toàn bộ chunk được chọn phải vừa giới hạn token sau khi thêm `passage: `.

Kiểm tra token trên toàn bộ dữ liệu diễn ra trước batch ghi đầu tiên. Lỗi ở một chunk dài không để các batch trước đó bị ghi trong cùng lượt xử lý.

## Định danh collection

```text
key = model_name + revision + normalized_jsonl_text
collection = "e5_" + SHA256(key)[:12]
```

Tên collection phụ thuộc model, revision và nội dung JSONL sau chuẩn hóa newline. `chunk_sha256` trong manifest dùng byte gốc của file để kiểm tra đúng đầu vào.

Collection tồn tại phải có số chiều tương ứng và khoảng cách cosine. Hàm đọc ID trong collection và từ chối point nằm ngoài tập ID dự kiến, thay vì tự xóa dữ liệu.

## Vòng lặp batch

1. Dựng `Document` từ các dòng được chọn.
2. Gọi `embedder.embed_documents()`.
3. Kiểm tra shape, giá trị hữu hạn và chuẩn L2 của vector.
4. Tạo `PointStruct(id, vector, payload)`; payload giữ nguyên dòng JSONL.
5. Gọi `upsert(..., wait=True)`.
6. Đọc lại toàn bộ ID trong batch và đối chiếu payload/vector.
7. Ghi số point kiểm tra được vào `upserted`.

`batch_size` của Indexer điều khiển batch ghi và xác minh. `batch_size` của Embedder điều khiển batch inference bên trong model; hai tham số có vai trò khác nhau.

## Manifest và trạng thái

| Trạng thái | Ý nghĩa |
| --- | --- |
| `in_progress` | Index đang được ghi và kiểm tra theo batch. |
| `complete` | Mọi batch khớp dữ liệu đầu vào và count collection khớp số chunk được chọn. |
| `failed` | Có lỗi trong quá trình ghi hoặc xác minh; trường `error` chứa thông báo. |

Manifest chứa model/revision, dimension, distance, collection, hash đầu vào, đường dẫn, số chunk, quy tắc chọn nguồn và quy tắc point ID. Khi có `parser_path`, hash và đường dẫn Parser cũng được ghi.

`_write_manifest()` ghi JSON vào file tạm cùng thư mục rồi thay thế file đích. Cách ghi này tránh để manifest ở trạng thái JSON chỉ được ghi một phần.

## Chạy lại dữ liệu

Cùng input và revision tạo cùng collection và point ID, nên upsert cập nhật các point đó. Hàm encode lại các chunk được chọn; không đọc cache vector theo nội dung và không tiếp tục từ số `upserted` của lượt trước.

Tham số CLI và cách đọc storage nằm trong [output indexing](../output_docs/indexing.md).
