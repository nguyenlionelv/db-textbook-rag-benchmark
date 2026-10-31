# Generation eval — corpus pilot

## Phạm vi đánh giá

Corpus gồm 233 chunk có nguồn từ chương 1–3. Tập dev có 46 kết quả: 24 với context chọn sẵn và 22 với context retrieval. Tập eval gồm 16 câu hỏi, mỗi câu có hai loại context, tổng cộng 32 kết quả; mỗi nhóm context/ngôn ngữ có 8 câu.

Context retrieval là [snapshot top-5](../../retrieval/generation_contexts/manifest.json) từ pipeline BM25 + E5 → RRF (20 ứng viên) → BGE reranker. Các nhãn dựa trên đối chiếu đáp án với context theo [tiêu chí đánh giá](review_rubric.md).

## Cấu hình

- Model: `qwen3:4b-instruct-2507-q4_K_M`, CPU, 4 thread, temperature 0, seed 42, không thinking.
- Context window: 4.096 token; tối đa 256 token đầu ra.
- Tokenizer: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Cấu hình eval: [frozen_config.json](frozen_config.json). Digest model và hash code, tokenizer, dữ liệu: [manifest.json](manifest.json).

## Kết quả eval

| Context | Ngôn ngữ | Đúng | Đủ ý | Bám nguồn | Citation hỗ trợ | Generation trung bình |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| retrieved | en | 8/8 | 8/8 | 8/8 | 8/8 | 35.9 s |
| retrieved | vi | 7/8 | 7/8 | 7/8 | 7/8 | 36.9 s |
| selected_context | en | 8/8 | 8/8 | 8/8 | 8/8 | 15.0 s |
| selected_context | vi | 7/8 | 8/8 | 7/8 | 7/8 | 17.8 s |

Độ đúng, đủ ý, bám nguồn, citation và trạng thái là các chỉ số riêng. Đáp án có thể đủ định nghĩa nhưng sai ở phần giải thích thêm; citation ID hợp lệ chưa chứng minh nguồn hỗ trợ đáp án. Nhận xét từng trường hợp nằm trong [manual_review.csv](manual_review.csv).

## So sánh dev với cấu hình 1.7B

| Context | Ngôn ngữ | Cấu hình 1.7B đúng | Cấu hình 4B đúng |
| --- | --- | ---: | ---: |
| retrieved | en | 10/11 (90.9%) | 10/11 (90.9%) |
| retrieved | vi | 7/11 (63.6%) | 10/11 (90.9%) |
| selected_context | en | 11/12 (91.7%) | 11/12 (91.7%) |
| selected_context | vi | 7/12 (58.3%) | 9/12 (75.0%) |

Hai cấu hình dùng cùng câu hỏi dev và tiêu chí chấm. Nhóm selected_context gồm hai ca context rỗng, nên có 24 kết quả thay vì 22. Số liệu cấu hình 1.7B: [summary.csv](../20261030_113151_222095/summary.csv).

Độ đúng trên dev tăng ở tiếng Việt và giữ nguyên ở tiếng Anh. Cấu hình 1.7B chưa có kết quả Generation eval để so sánh trên cùng tập eval.

## Lỗi và giới hạn

- Eval `q23/selected_context`: định nghĩa transaction đủ ý, nhưng phần giải thích thêm làm lẫn consistency và atomicity; độ đúng, bám nguồn và citation đều False.
- Eval `q19/retrieved`: context có tiêu đề liên quan đến data abstraction và mô tả insulation, nhưng thiếu giải thích data abstraction hỗ trợ cả hai dạng độc lập. Đáp án nhắc lại sự tách biệt thay vì xác định data abstraction và vẫn trả `answered`; `review_answerable=False`.
- Các lỗi dev gồm nhầm schema với schema diagram hoặc conceptual schema, nhầm logical/physical independence và suy vai trò DBA chung thành thông tin về một trường cụ thể.
- Một số đáp án tiếng Việt giữ thuật ngữ tiếng Anh hoặc dùng cách dịch chưa tự nhiên. Lỗi diễn đạt được ghi trong nhận xét; nhầm các khái niệm khác nhau bị chấm sai kiến thức.
- Mỗi nhóm eval chỉ có 8 câu đơn có đáp án trong corpus; câu nhiều nguồn và probe thuộc dev. Chủ đề eval từng xuất hiện trong các thực nghiệm trước, nên tập này chưa hoàn toàn độc lập.
- Phạm vi kết quả giới hạn ở 233 chunk pilot, không đại diện toàn bộ sách.

## Thời gian

Thời gian Generation gồm ghép context, gọi HTTP local, kiểm tra output và có thể nạp lại Ollama; không tính các lượt warm-up riêng. Đây là thời gian quan sát trên CPU, không phải benchmark cold-cache hoặc p95.

`retrieval_seconds` và `total_seconds` để trống vì context retrieval lấy từ snapshot. Bảng không biểu thị thời gian end-to-end của pipeline RAG.

## Dữ liệu kết quả

- [per_query.jsonl](per_query.jsonl): đáp án, context, nguồn, usage và thời gian từng kết quả.
- [manual_review.csv](manual_review.csv): 78 bộ nhãn và nhận xét, gắn với output bằng `review_id`.
- [summary.csv](summary.csv): chỉ số tổng hợp theo dev/eval, context và ngôn ngữ.
- [manifest.json](manifest.json), [frozen_config.json](frozen_config.json): cấu hình và thông tin dữ liệu/model.
- [review_rubric.md](review_rubric.md): tiêu chí và quy ước chấm.
