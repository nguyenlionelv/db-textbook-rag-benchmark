# E5Embedder

Mã nguồn: [textbook_embedder.py](../../src/ingestion/textbook_embedder.py).

`E5Embedder` tạo vector cho chunk và câu hỏi bằng `SentenceTransformer`. Vector trả về được chuẩn hóa L2 và giữ thứ tự của dữ liệu đầu vào.

## Khởi tạo

| Tham số | Giá trị mặc định | Vai trò |
| --- | --- | --- |
| `model_name` | `intfloat/multilingual-e5-small` | Model tạo embedding. |
| `revision` | `None` | Revision của model được nạp. |
| `device` | `cpu` | Thiết bị inference. |
| `batch_size` | `8` | Số text trong một batch encode. |
| `cache_folder` | `None` | Thư mục cache model. |
| `local_files_only` | `False` | Khi True, chỉ đọc model từ cache. |

`batch_size` phải là số nguyên dương. Model được nạp một lần trong `__init__()` và được dùng lại cho các lần encode.

## embed_documents()

`embed_documents(chunks)` lấy `page_content` từ từng `Document`, rồi gọi `_encode()` với prefix `passage: `.

Đầu ra là ma trận NumPy `float32` có dạng `(n_documents, dimension)`. Số chiều được xác định bởi model; E5-small dùng 384 chiều. Metadata không được encode và không bị sửa.

## embed_query()

`embed_query(text)` gọi `_encode()` với danh sách một phần tử và prefix `query: `, sau đó lấy vector đầu tiên. Đầu ra có dạng `(dimension,)`.

Prefix query và passage phân biệt hai vai trò văn bản trong E5. Chúng chỉ được thêm vào input của model; nội dung lưu trong chunk giữ nguyên.

## Kiểm tra và encode trong _encode()

1. Danh sách rỗng trả ma trận rỗng có dạng `(0, dimension)`.
2. Mỗi text phải là chuỗi có nội dung.
3. Prefix được thêm trước khi đếm token.
4. Tokenizer đếm với `truncation=False`, gồm prefix và special token.
5. Text vượt `model.max_seq_length` gây lỗi trước inference.
6. `model.encode()` chạy theo batch, dùng `float32`, chuẩn hóa vector và trả NumPy array.

`prompt=""` ngăn model áp thêm prompt cấu hình ngoài prefix được tạo trong hàm. Kiểm tra độ dài trước encode tránh mất nội dung do truncation.

## Chuẩn hóa và phép so sánh

Với vector L2 có norm bằng 1, tích vô hướng tương ứng cosine similarity. Vector tài liệu và query phải dùng cùng model, revision và cách thêm prefix để cùng thuộc một không gian biểu diễn.

Class không chọn chunk có nguồn và không ghi vector xuống đĩa. Việc chọn dữ liệu và lưu vector thuộc [Indexer](textbook_indexer.md). Các biểu diễn dùng trong notebook so sánh được mô tả trong [output embedding](../output_docs/embedding.md).
