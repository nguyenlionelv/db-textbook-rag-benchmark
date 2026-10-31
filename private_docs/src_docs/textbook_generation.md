# Context assembly và OllamaGenerator

Mã nguồn: [textbook_generation.py](../../src/generation/textbook_generation.py).

Module nhận câu hỏi và các chunk có nguồn, tạo prompt theo token budget, gọi Ollama local và ánh xạ citation về nguồn PDF.

## ANSWER_SCHEMA và SYSTEM_PROMPT

`ANSWER_SCHEMA` định nghĩa ba trường `answer`, `status`, `citations` và không cho thêm trường khác. `status` nhận `answered` hoặc `insufficient_evidence` trong output model.

`SYSTEM_PROMPT` yêu cầu trả lời dựa trên nguồn, phân biệt kiến thức chung với thông tin về hệ thống/người cụ thể, trả lời ngắn và dẫn nguồn cho các phát biểu. Chỉ dẫn tiếng Việt hoặc tiếng Anh được bổ sung cạnh context theo tham số `language`.

## build_context()

`build_context(query, hits, tokenizer, num_ctx, max_new_tokens, language)` trả `prompt`, `context_text`, `sources`, `prompt_tokens` và `omitted`.

Hàm kiểm tra query, ngôn ngữ và điều kiện `0 < max_new_tokens < num_ctx`. Token của câu hỏi và chỉ dẫn được kiểm tra trước khi thêm nguồn.

Hàm con `render()` dùng `apply_chat_template()` với `enable_thinking=False`, sau đó đếm token của toàn bộ prompt bằng `encode(..., add_special_tokens=False)`. Cách đếm bao gồm chat template nhưng không thêm special token lần thứ hai.

## Chọn chunk theo token budget

Các hit được xét theo thứ hạng đầu vào. Mỗi hit cần text không rỗng và nguồn có ID record, chỉ số trang hợp lệ.

- Chunk trùng cả text và danh sách nguồn được bỏ với lý do `duplicate`.
- Nguồn được gắn nhãn `S1`, `S2`, ...; trang hiển thị trong prompt là `pdf_page + 1`.
- Toàn bộ prompt được đếm lại sau mỗi lần thử thêm chunk.
- Chunk không vừa ngân sách được bỏ với lý do `token_budget`; các chunk sau vẫn được xét.

Điều kiện nhận chunk là `prompt_tokens + max_new_tokens <= num_ctx`. Nội dung chunk không bị cắt giữa chừng.

Trong `sources`, `pdf_pages` vẫn bắt đầu từ 0. Mỗi nhãn nguồn liên kết với `point_id` và bản sao metadata của chunk.

## Khởi tạo và kiểm tra Ollama

`OllamaGenerator` mặc định dùng `qwen3:1.7b`, context 4.096 token, tối đa 256 token đầu ra, timeout 180 giây và `keep_alive="2m"`. Notebook 07 truyền model 4B Instruct bằng tham số riêng.

Constructor chấp nhận HTTP localhost, không cho thông tin đăng nhập, path, query hoặc fragment trong `base_url`. Tên model không được rỗng hoặc chứa tag cloud.

`_request()` gửi JSON UTF-8 qua HTTP và giải mã JSON phản hồi. Lỗi HTTP, kết nối và timeout được chuyển thành `RuntimeError` có thông báo tương ứng.

`check_model()` đọc `/api/tags`, `/api/show` và `/api/version`. Hàm yêu cầu đúng tag model trong danh sách local và từ chối metadata trỏ tới model remote.

## generate()

`generate(query, hits, language=...)` gọi `build_context()` và chuẩn bị kết quả mặc định thiếu bằng chứng. Nếu không có nguồn được chọn, hàm trả ngay với `called_model=False`.

Với context có nguồn, schema được tách thành hai nhánh:

| Status | Yêu cầu citation |
| --- | --- |
| `answered` | Có ít nhất một ID nguồn thuộc context. |
| `insufficient_evidence` | Danh sách citation rỗng. |

Cả hai nhánh yêu cầu `answer` có nội dung. Các ID cho phép được đưa vào schema của lượt gọi đó.

Request tới `/api/generate` dùng `raw=True`, `stream=False`, `think=False`, temperature 0, seed 42, CPU và 4 thread. Prompt đã có chat template nên `raw=True` tránh áp template lần nữa.

## Kiểm tra phản hồi

Hàm kiểm tra phản hồi hoàn tất, không bị cắt do giới hạn token và số token input Ollama khớp tokenizer. JSON phải có đúng ba trường; answer không rỗng; citation phải là danh sách ID duy nhất, thuộc context và phù hợp status.

Citation hợp lệ được ánh xạ thành đầy đủ metadata trong `sources`. Lỗi request hoặc lỗi output được xử lý thành `status="error"`, answer rỗng và `citation_valid=False`; input không hợp lệ ở bước dựng context gây `ValueError` trước lượt gọi.

## Các trường chẩn đoán

| Trường | Ý nghĩa |
| --- | --- |
| `raw_output` | Chuỗi JSON gốc từ model. |
| `usage` | Token input/output, thông tin tải model, thời gian runtime và lý do dừng. |
| `seconds` | Thời gian dựng context, gọi model và kiểm tra kết quả. |
| `called_model` | Có gửi request generation hay không. |
| `omitted` | Các hit bị bỏ và lý do. |
| `citation_valid` | Hình thức và ID citation hợp lệ; không xác minh đúng kiến thức hoặc hỗ trợ ngữ nghĩa. |

Các nhãn đúng, đủ ý, bám nguồn và citation hỗ trợ được lưu riêng trong bảng review. Cấu trúc dữ liệu được giải thích trong [output generation](../output_docs/generation.md).
