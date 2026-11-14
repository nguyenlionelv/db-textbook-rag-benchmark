# Retrieval trên toàn sách: 100 câu Database Systems/DBMS

## Phạm vi

50 câu hỏi gốc, mỗi câu có bản Việt và Anh: 100 câu, gồm 60 dev và 40 eval. Các cặp dịch và nhóm chủ đề được giữ cùng split. Câu hỏi lấy bằng chứng từ chương 1–2; tìm kiếm trên toàn bộ 6.040 chunk của 30 chương và 3 phụ lục, không lọc chương.

[Bộ câu hỏi](../../../../data/benchmarks/dbms_foundations_100/questions.md) và [nhãn bằng chứng](queries.jsonl) đã được chốt trước khi chạy retrieval. Mỗi câu có đáp án tham chiếu, record ID, trang PDF và khoảng ký tự của các nhóm bằng chứng cần tìm.

## Cấu hình

BM25 + multilingual E5 small → RRF giữ 20 ứng viên → BGE reranker v2-m3 lấy top-5. CPU, 4 thread, reranker batch 2 và giới hạn 512 token/cặp. Model/revision, hash input/code và phiên bản thư viện nằm trong [frozen_config.json](frozen_config.json).

Hybrid và Hybrid + reranker dùng cùng tập ứng viên. Thời gian sau rerank cộng thời gian retrieval trực tiếp với thời gian reranker; không gọi lại retrieval lần thứ hai cho cùng câu.

## Kết quả trên 100 câu

| Ngôn ngữ | Số câu | Hybrid Hit@5 | Sau rerank Hit@5 | Sau rerank MRR@5 | Đủ nhóm bằng chứng@5 sau rerank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anh | 50 | 86% | 92% | 0,877 | 84% |
| Việt | 50 | 36% | 60% | 0,567 | 54% |

Hit@5 ghi nhận tìm được ít nhất một đoạn bằng chứng đã gán nhãn. MRR@5 dùng vị trí đầu tiên có đoạn như vậy. Chỉ số đủ nhóm bằng chứng yêu cầu mọi nhóm cần thiết đều có mặt trong top-5; một nhóm có thể có nhiều nguồn thay thế. Coverage@5 trong các file CSV là tỷ lệ nhóm bằng chứng được tìm thấy.

## Tập eval: 40 câu, tương ứng 20 cặp

| Ngôn ngữ | Số câu | Hybrid Hit@5 | Sau rerank Hit@5 | Sau rerank MRR@5 | Đủ nhóm bằng chứng@5 sau rerank | Candidate Hit@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anh | 20 | 90% | 95% | 0,925 | 80% | 95% |
| Việt | 20 | 35% | 50% | 0,500 | 40% | 55% |

Các bảng theo dev/eval, ngôn ngữ và phương pháp nằm trong [summary.csv](summary.csv). Kết quả gộp nằm trong [overall.csv](overall.csv).

## Các trường hợp chưa tìm đủ bằng chứng

Sau rerank, 24/100 câu không tìm được đoạn đã gán nhãn nào trong top-5. Trong đó, 23 câu không có đoạn đã gán nhãn trong 20 ứng viên. Có 31 câu chưa đủ mọi nhóm bằng chứng; 28 câu đã thiếu nhóm ngay ở tập ứng viên. Chi tiết: [failure_analysis.csv](failure_analysis.csv).

Theo các nhãn hiện có, bước tạo ứng viên là điểm cần ưu tiên trên tập dev, đặc biệt với câu tiếng Việt. Reranker không thể đưa nguồn lên top-5 nếu nguồn không có trong 20 ứng viên. Một số top-5 chứa câu hỏi ôn tập, văn bản trích từ hình/bảng hoặc nội dung chuyên sâu ở chương khác thay cho định nghĩa nền tảng cần tìm.

## Giới hạn của nhãn

Các đoạn đã gán nhãn chưa bao phủ mọi cách trả lời đúng trong toàn sách. Vì vậy, một trường hợp trượt nhãn chưa chắc là truy xuất sai về nội dung. Ví dụ:

- `f05_en`: top-1 là chunk 135, phần tổng kết chương 1, nêu đúng rằng cơ sở dữ liệu và phần mềm hợp thành hệ cơ sở dữ liệu. Đoạn được gán nhãn ban đầu ở phần nhập môn không xuất hiện nên Hit@5 vẫn bằng 0.
- `f30_en`: top-5 có chunk 280 mô tả schema và state, bổ sung cho chunk 168 định nghĩa state; nhóm schema đã gán nhãn vẫn bị tính là thiếu.
- `f14_vi`: chunk 56 có ví dụ đặt ghế, chunk 109 nhắc phân hệ điều khiển đồng thời; nguồn thứ nhất đã gán nhãn vẫn không có trong top-5.

[qualitative_examples.json](qualitative_examples.json) lưu bốn ví dụ đã đọc lại và các câu trích khớp với output. Đây là rà soát bổ sung có chọn mẫu, không phải chấm lại ngữ nghĩa toàn bộ 100 câu. Không sửa nhãn hoặc điểm chính sau khi xem kết quả.

Bộ câu hỏi và lượt rà soát do coding agent thực hiện, chưa có chấm độc lập của người dùng. Chủ đề có giao với các thực nghiệm trước, nên eval không phải một tập kiểm định hoàn toàn chưa từng được quan sát. Hai ngôn ngữ là bản dịch của 50 tình huống, không phải 100 tình huống độc lập. Không so sánh trực tiếp tỷ lệ này với benchmark pilot 32 câu vì câu hỏi, chunking và corpus đã khác.

## Thời gian và kiểm tra

Lượt chạy mất 1.452,96 giây, khoảng 24 phút 13 giây, gồm setup, kiểm tra token, warm-up và đánh giá. Trung bình sau rerank là 13,30 giây/câu Anh và 14,83 giây/câu Việt; Hybrid khoảng 0,08 giây/câu. Thời gian từng câu không gồm setup, kiểm tra token và warm-up. Đây là một lượt quan sát trên CPU, không phải p95 hoặc thử nghiệm cold-cache.

Không có lỗi runtime. Cặp query–candidate dài nhất 382 token, dưới giới hạn 512. Đã tính lại chỉ số từ 200 ranking đã lưu, kiểm tra 1.000 payload top-5 với corpus và đối chiếu các hash. Bộ test đạt 322 passed, 1 skipped. Chi tiết: [verification.json](verification.json).

Lượt này đánh giá retrieval; chưa chạy Generation hoặc LLM API. Các nguồn tương đương được phát hiện cần được rà soát thêm trước khi dùng nhãn làm chuẩn chính thức cho các vòng so sánh tiếp theo.
