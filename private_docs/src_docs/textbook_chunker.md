# TextbookChunker

Mã nguồn: [textbook_chunker.py](../../src/ingestion/textbook_chunker.py).

Module nhận record từ Parser và trả danh sách LangChain `Document`. `page_content` chứa text của chunk; `metadata` chứa cấu trúc tài liệu và các khoảng văn bản nguồn.

## Các class và tham số

| Class | Đơn vị cắt | Ranh giới section |
| --- | --- | --- |
| `BaseChunker` | Chuẩn bị chapter, section và source span. | Cung cấp metadata chung. |
| `StrictFixedSizeChunker` | Cửa sổ ký tự trên từng chapter. | Có thể cắt ngang section. |
| `GlobalRecursiveChunker` | Recursive splitter trên từng chapter. | Section là một phần của chuỗi văn bản. |
| `StructureAwareRecursiveChunker` | Recursive splitter trên từng đơn vị section. | Một chunk thuộc một đơn vị section. |

`chunk_size` và `chunk_overlap` đo bằng ký tự, không phải token. Điều kiện tham số là `chunk_size > 0` và `0 <= chunk_overlap < chunk_size`. `include_footnotes=False` loại record footnote trước khi dựng văn bản.

## Kiểm tra đầu vào

`_validate_record()` kiểm tra các trường Parser, kiểu dữ liệu, bbox hữu hạn và metadata có thể biểu diễn bằng JSON. `_valid_records()` kiểm tra ID trùng, bỏ text rỗng, áp dụng quy tắc footnote và yêu cầu chapter cho record được giữ.

Việc kiểm tra text rỗng dùng `strip()`, nhưng text nguồn không bị thay đổi. Record hợp lệ được sao chép để các bước sau không sửa đầu vào của Parser.

## Dựng chapter và section

`_prepare_chapters()` nhóm các record liên tiếp theo chapter, sau đó theo section. `groupby()` giữ thứ tự đọc; section xuất hiện lại sau section khác tạo một đơn vị mới với `section_unit_index` riêng.

`_consistent_title()` lấy tiêu đề có nội dung duy nhất trong nhóm. Tiêu đề mâu thuẫn gây lỗi. Chapter xuất hiện lại sau chapter khác cũng gây lỗi vì thứ tự đầu vào không liên tục.

Mỗi section có heading dạng `[[SECTION <number>: <title>]]`. Các record trong section nối bằng một newline; các section trong chapter nối bằng hai newline. Heading và ký tự nối không được gán record nguồn.

Section thiếu số dùng nhãn `<chapter>.intro`. `_section_id()` tạo `sec_<number>`; `_parent_section_id()` suy chapter cha hoặc section cha từ số mục phân cấp.

## Cắt chunk

`StrictFixedSizeChunker.chunk()` dùng bước trượt `chunk_size - chunk_overlap`. Chunk cuối dừng tại cuối chapter; các chunk trong cùng chapter có overlap theo cửa sổ ký tự.

`GlobalRecursiveChunker` dùng các separator mặc định theo thứ tự: đoạn văn, dòng, câu, từ, ký tự. Separator cuối là chuỗi rỗng để có thể cắt đến mức ký tự khi cần.

`_recursive_ranges()` tìm vị trí text do splitter trả về trong văn bản nguồn. Điểm tìm bắt đầu sau vị trí chunk trước, có xét overlap, để tránh ánh xạ chuỗi lặp về cùng một vị trí.

`StructureAwareRecursiveChunker.chunk()` gọi cùng splitter cho từng section. Chỉ số chunk tăng trong chapter; offset của chunk thuộc văn bản section tương ứng.

## Ánh xạ nguồn trong _make_chunk()

Hàm xét phần giao giữa khoảng chunk và từng source span:

```text
left  = max(chunk_start, source_span_start)
right = min(chunk_end, source_span_end)
```

Chỉ phần có `left < right` tạo một mục trong `sources`.

| Trường nguồn | Hệ quy chiếu |
| --- | --- |
| `source_start_index`, `source_end_index` | Khoảng `[start, end)` trong `record.text`. |
| `chunk_start_index`, `chunk_end_index` | Khoảng `[start, end)` trong `chunk.page_content`. |
| `record_id`, `pdf_page`, `bbox` | Record và block PDF gốc. |
| `metadata` | Bản sao metadata của record, gồm tham chiếu footnote. |

Hai lát cắt ở phía record và phía chunk phải cho cùng chuỗi. `bbox` vẫn là hộp bao record, không phải hộp bao chính xác của lát text bị cắt. `footnote_refs.raw_offset` vẫn thuộc `record.raw_text`.

## Metadata của chunk

`offset_scope` nhận `chapter` hoặc `section`, xác định hệ quy chiếu của `start_index` và `end_index`. `section_ids` liệt kê các section giao với chunk.

Khi chunk giao nhiều section, các trường section đơn như `section_id` hoặc `section_title` có thể là `None`. Một chunk chỉ có heading có thể có `sources=[]`; bước indexing lọc theo nguồn thay vì gán nguồn cho heading.

Tệp JSONL của ba chiến lược được giải thích trong [output chunking](../output_docs/chunking.md).
