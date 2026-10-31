# Tiêu chí đánh giá Generation

| Cột | Tiêu chí |
| --- | --- |
| review_answerable | Context có đủ bằng chứng để trả lời. |
| review_correct | Đáp án đúng câu hỏi và kiến thức, hoặc thừa nhận thiếu thông tin đúng. Lỗi status được đo riêng. |
| review_complete | Đáp án có đủ ý cần thiết; câu hỏi gọi tên có thể trả lời bằng một cụm từ. |
| review_supported | Các phát biểu được context hỗ trợ trực tiếp hoặc qua suy luận ngắn. |
| review_citations | Các nguồn được dẫn hỗ trợ đầy đủ phát biểu trong đáp án. |

## Quy ước

- Dùng True/False; để trống complete, supported và citations khi insufficient_evidence.
- Xác định answerable từ context thực tế, kể cả nguồn thay thế đoạn gold.
- Context rỗng kiểm tra nhánh xử lý không gọi model.
- Nguồn có trong context nhưng không được dẫn có thể tạo supported=True và citations=False.
- Văn bản đúng nhưng status sai được ghi riêng ở status_match_rate.
- Khi answered, chấm đủ complete, supported và citations, kể cả context không đủ trả lời.
- Chấm toàn bộ phát biểu, gồm ví dụ và lợi ích được thêm; ý chính đúng chưa đủ để chấm supported=True.
- Tiêu đề hoặc câu hỏi ôn tập không thay thế bằng chứng giải thích nội dung được hỏi.
- Lỗi diễn đạt hoặc thuật ngữ Việt chưa chuẩn nhưng vẫn xác định đúng khái niệm được ghi trong review_notes. Nhầm các khái niệm khác nhau, như consistency với atomicity, bị chấm sai kiến thức.

## Các trường hợp cần phân biệt

- unknown_person: tài liệu mô tả vai trò DBA chung, không xác định người hoặc đơn vị quản trị trường cụ thể.
- unknown_hardware: context không có RAM của máy chủ trường. Nội dung báo thiếu thông tin phù hợp với insufficient_evidence và citations rỗng.
- q3 retrieved: nguồn thay thế vẫn mô tả schema, nên answerable=True; cần phân biệt schema nói chung với conceptual schema.
- multi_vi retrieved: nguồn chỉ có câu hỏi ôn tập, thiếu giải thích state để so sánh; answerable=False.
- q15 selected_context: câu hỏi cần tên thuộc tính khóa; tên tính chất duy nhất không thay thế tên thuộc tính.

SHA-256 per_query.jsonl: `852d89da2532e3aebe592860f6fdca8fd55609e10aa519d1ba0b33084ee20a11`.
