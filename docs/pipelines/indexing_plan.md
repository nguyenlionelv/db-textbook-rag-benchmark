# Kế hoạch Indexing

Ngày: 14/10/2026.

Điểm xuất phát: output Chunker có 241 chunk, trong đó 233 chunk có nguồn.
Sử dụng E5-small và Qdrant Local.

## 1. Chuẩn bị dữ liệu

- Đọc JSONL từ Chunker.
- Kiểm tra `page_content`, metadata và nguồn `record_id`, `pdf_page`.
- Chọn chunk có nguồn; thống kê chunk bị bỏ qua.
- Giữ nguyên nội dung, metadata và offsets.

## 2. Tạo embedding

- Dùng `intfloat/multilingual-e5-small`, cố định revision.
- Thêm tiền tố `passage: ` và kiểm tra giới hạn token trước khi encode.
- Báo lỗi khi vượt giới hạn; không cắt ngầm nội dung.
- Tạo vector 384 chiều, chuẩn hóa L2, xử lý theo batch.

## 3. Lưu vào Qdrant

- Dùng Qdrant Local với cosine similarity.
- Đặt tên collection theo hash của model, revision và dữ liệu đầu vào.
- Dùng vị trí record JSONL làm point ID.
- Upsert theo batch; chạy lại cùng dữ liệu không tạo point trùng.
- Đóng client sau khi xử lý; dùng database riêng nếu database hiện tại đang khóa.

## 4. Xác minh và ghi manifest

- Đọc lại ID, payload và vector; đối chiếu số point với dữ liệu đã chọn.
- Ghi model/revision, collection, đường dẫn database, hash đầu vào và số chunk.
- Dùng ba trạng thái: `in_progress`, `failed`, `complete`.
- Chỉ đưa index sang retrieval khi manifest là `complete`.
- Giữ riêng manifest indexing và kết quả thực nghiệm retrieval.

## 5. Kiểm thử

- Lưu, đóng và mở lại database; kiểm tra payload/vector và query.
- Chạy lại cùng input; thử input hoặc revision khác.
- Kiểm tra schema, token, vector và cấu hình collection không hợp lệ.
- Kiểm tra lỗi giữa batch và chạy lại sau lỗi.
- Đối chiếu manifest với cấu hình retriever.
