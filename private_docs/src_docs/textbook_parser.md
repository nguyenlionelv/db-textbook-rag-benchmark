# TextbookParser

Mã nguồn: [textbook_parser.py](../../src/ingestion/textbook_parser.py).

`TextbookParser` chuyển các block văn bản PDF thành record có chapter, section, vị trí trang và liên kết footnote. Thư viện PyMuPDF cung cấp văn bản, font, tọa độ block và tọa độ span.

## Khởi tạo và trạng thái

`__init__(pdf_path)` mở PDF bằng `fitz.open()` và tạo `document_id` từ 16 ký tự đầu của SHA-256 nội dung file. Đổi tên hoặc di chuyển cùng file PDF không làm đổi ID tài liệu.

`current_chapter` và `current_section` giữ ngữ cảnh khi đọc liên tiếp các trang. Mỗi lần gọi `parse_pages()` đặt lại hai biến này; chapter được nhận diện trong lần đọc đó.

## Chuẩn hóa và phân loại văn bản

| Hàm | Xử lý |
| --- | --- |
| `clean_text()` | Ghép từ bị ngắt bằng dấu gạch nối trước chữ thường, chuyển xuống dòng thành khoảng trắng và gộp khoảng trắng thừa. |
| `classify_block()` | Nhận diện caption hình, caption bảng, block giống bảng và văn bản thông thường. |
| `_dominant_font()` | Chọn cỡ font xuất hiện nhiều nhất theo tổng số ký tự có nội dung. |

`classify_block()` nhận `raw_text` để giữ thông tin dòng. Block được gán `table_like` khi tỷ lệ dòng ngắn lớn hơn 0,7 và tỷ lệ dòng có chữ số lớn hơn 0,3. Nhãn này là phân loại hình thức văn bản, không phải cấu trúc hàng/cột của bảng.

## Đọc layout và nhận diện footnote

`_layout_blocks()` đọc cấu trúc `dict` của trang. Mỗi block giữ `raw_text`, `text`, `bbox`, danh sách line, span và cỡ font chính. `raw_offset` của span được tính trước khi làm sạch văn bản.

`_reference_candidates()` tìm chữ số nhỏ nằm cao hơn baseline hoặc có cờ superscript. Hàm loại các biểu thức toán học ngắn và một số dạng số mũ để giảm nhầm lẫn với tham chiếu chú thích.

`_footnote_candidate()` xét block ở vùng dưới trang, font nhỏ hơn thân bài và có marker số ở đầu. `recognition_basis` phân biệt nhận diện có tham chiếu trong thân bài với nhận diện chỉ dựa trên layout.

`extract_content_blocks()` thực hiện theo thứ tự:

1. Dựng block và ước lượng font thân bài.
2. Thu thập marker tham chiếu.
3. Nhận diện block footnote.
4. Ghép các block tiếp nối gần nhau, cùng cột và có font phù hợp.
5. Loại vùng header/footer theo tọa độ; footnote được xét trước khi loại footer.

## Nhận diện chapter và section

`detect_chapter_from_page()` tìm mẫu `Chapter <number> <title>` trong block header có `y0 < 50`. Dòng chỉ chứa số trang được bỏ trước khi ghép tiêu đề.

`detect_section()` tìm số mục như `1.3` hoặc `1.3.1`. Phần tiêu đề phải có chữ cái; chữ số đầu của section phải khớp chapter khi đã có chapter. Heading section cập nhật trạng thái và không trở thành record nội dung riêng.

## Tạo record trong parse_pages()

`parse_pages(start_page, end_page)` đọc cả hai trang biên. Chỉ số trang bắt đầu từ 0. Khi chapter thay đổi, section cũ được xóa trước khi đọc nội dung chapter mới.

Mỗi trang có hai lượt xử lý: tạo các record footnote trước, sau đó liên kết marker trong thân bài với các record đó. Marker và footnote chỉ được ghép trong cùng trang.

| Trạng thái tham chiếu | Điều kiện |
| --- | --- |
| `resolved` | Có đúng một footnote cùng marker. |
| `ambiguous` | Có nhiều footnote cùng marker. |
| `unresolved` | Không có footnote cùng marker. |

Marker `resolved` được bỏ khỏi `text` qua `_remove_resolved_markers()`. `raw_text` vẫn giữ nguyên; marker chưa giải quyết vẫn nằm trong văn bản để tránh xóa nhầm nội dung.

## Cấu trúc record

`_make_record()` tạo record từ trạng thái chapter/section và block đang đọc. Hàm giữ `raw_text`, `bbox`, tạo ID theo tài liệu/trang/block và khởi tạo `metadata.footnote_refs` rỗng; nhánh xử lý footnote bổ sung metadata riêng sau bước này.

| Trường | Ý nghĩa |
| --- | --- |
| `id` | ID ổn định theo tài liệu, trang và block. Footnote có thêm marker trong ID. |
| `pdf_page` | Chỉ số trang PDF bắt đầu từ 0. |
| `chapter_number`, `chapter_title` | Chapter đang áp dụng cho block. |
| `section_number`, `section_title` | Section đang áp dụng; phần mở đầu có thể chưa có số section. |
| `type` | `normal_text`, `figure_caption`, `table_caption`, `table_like` hoặc `footnote`. |
| `text`, `raw_text` | Văn bản sạch và văn bản giữ xuống dòng/layout text. |
| `bbox` | Hộp bao block trong hệ tọa độ PDF. |
| `metadata` | Tham chiếu footnote hoặc thông tin liên kết ngược của footnote. |

Footnote giữ `referenced_by`, `reference_contexts`, `source_block_numbers` và trạng thái liên kết. Khi có tham chiếu, chapter/section của footnote được suy từ các record tham chiếu; nếu các ngữ cảnh không thống nhất, trường tương ứng nhận `None`. Footnote không có tham chiếu giữ chapter từ header và không được gán section.

## Xuất dữ liệu và đóng tài liệu

`export_jsonl()` ghi mỗi record thành một dòng JSON UTF-8. `close()` đóng tài liệu PyMuPDF và giải phóng tài nguyên file.

Đầu ra được sử dụng bởi [Chunker](textbook_chunker.md); cấu trúc file được mô tả trong [output parsing](../output_docs/parsing.md).
