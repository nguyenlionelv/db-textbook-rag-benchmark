# Hybrid trước và sau reranking

32 câu hỏi cố định: 16 dev, 16 eval; Anh và Việt.
Hit@5 và MRR@5 dựa trên khoảng evidence đã gán, không phải Recall đầy đủ.

| Split | Ngôn ngữ | Phương pháp | Hit@5 | MRR@5 | Candidate Hit@20 | Giây/query |
| --- | --- | --- | --- | --- | --- | --- |
| dev | en | hybrid | 1.000 | 1.000 | 1.000 | 0.07 |
| dev | en | hybrid_rerank | 1.000 | 1.000 | 1.000 | 26.68 |
| dev | vi | hybrid | 0.625 | 0.500 | 0.875 | 0.07 |
| dev | vi | hybrid_rerank | 0.875 | 0.750 | 0.875 | 27.17 |
| eval | en | hybrid | 1.000 | 0.938 | 1.000 | 0.05 |
| eval | en | hybrid_rerank | 1.000 | 1.000 | 1.000 | 22.80 |
| eval | vi | hybrid | 0.500 | 0.344 | 0.875 | 0.05 |
| eval | vi | hybrid_rerank | 0.875 | 0.875 | 0.875 | 24.27 |

Thời gian là trung bình một lượt sau warm-up trên máy hiện tại, không phải benchmark p95.
Reranker chỉ sắp xếp 20 ứng viên; không tìm lại bằng chứng ngoài tập ứng viên.
Xem per_query.csv và topk.jsonl để đối chiếu từng câu và nguồn.

Trên 32 câu: 9 câu tăng MRR@5, 0 câu giảm, 23 câu giữ nguyên.

- Dữ liệu trong cơ sở dữ liệu tại một thời điểm được gọi là gì? — MRR@5: 0.000 → 1.000.
- Thông tin lưu trong catalog của DBMS được gọi là gì? — MRR@5: 0.000 → 1.000.

Lượt kiểm chứng dùng bản sao SQLite của index đang được mở; toàn bộ payload, IDs và hash corpus/parser đã đối chiếu. Baseline Hybrid khớp notebook 05.
