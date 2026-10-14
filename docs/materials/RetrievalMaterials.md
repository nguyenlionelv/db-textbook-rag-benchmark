# Retrieval Materials — lý thuyết truy xuất thông tin

Retrieval nhận câu hỏi/query và tìm các đoạn liên quan từ corpus. Trong RAG, nó cung cấp bằng chứng cho bước sinh câu trả lời. Embedding tạo biểu diễn; retrieval sử dụng biểu diễn, index và quy tắc ranking để chọn kết quả.

```text
Query → lấy ứng viên → kết hợp/xếp hạng lại nếu cần → top-k chunk + nguồn
```

## 1. Có bao nhiêu phương pháp?

**Không có một số lượng cố định.** Có thể học qua bốn nhóm lấy ứng viên bên dưới; hybrid, reranking và query expansion là những cách xây dựng pipeline, còn exact/ANN là cách thực thi tìm kiếm. Chúng không phải các mục cùng một cấp phân loại.

| Nhóm lấy ứng viên | Đại diện | Dựa vào đâu? |
| --- | --- | --- |
| Lexical truyền thống | Boolean, TF-IDF, BM25 | Từ/thuật ngữ xuất hiện trong query và tài liệu |
| Dense single-vector | E5, DPR, BGE dense | Độ gần của một vector query với một vector mỗi đoạn |
| Neural sparse | SPLADE, BGE-M3 sparse | Token/term với trọng số được model học |
| Multi-vector / late interaction | ColBERT, BGE-M3 multi-vector | Tương tác giữa nhiều vector token query và đoạn |

Một pipeline có thể kết hợp nhiều nhóm. Model E5 là lựa chọn tạo biểu diễn dense; Qdrant là công cụ lưu/tìm kiếm; hai cái không phải hai phương pháp cạnh tranh.

## 2. Lexical retrieval

### Boolean và TF-IDF

Boolean dùng điều kiện AND/OR/NOT để xác định tài liệu khớp. Phù hợp yêu cầu từ khóa chính xác; tự nó không xếp hạng theo mức liên quan. Inverted index lưu term → danh sách tài liệu chứa term. [Stanford — Boolean retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/boolean-retrieval-1.html)

TF-IDF tăng trọng số từ thường gặp trong một đoạn nhưng hiếm trên corpus; cosine giữa vector query và tài liệu cho ranking. Notebook 04 đã minh họa. Tokenization, vocabulary và IDF phải nhất quán khi index/query.

### BM25

BM25 thêm cơ chế bão hòa tần suất từ và điều chỉnh độ dài tài liệu:

$$
s(q,d)=\sum_{t\in q} IDF(t)\,
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/avgdl)}.
$$

`f(t,d)` là tần suất term, `|d|` là độ dài theo token, `avgdl` là độ dài trung bình corpus. `k1` điều khiển bão hòa; `b` điều khiển mức chuẩn hóa độ dài. Một từ lặp nhiều lần không làm điểm tăng tuyến tính mãi. [Stanford — BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)

IDF có nhiều biến thể. `rank_bm25.BM25Okapi` dùng log-ratio có smoothing 0.5 và thay IDF âm bằng `epsilon × average_idf`; vì vậy không lấy công thức IDF của một thư viện khác rồi kỳ vọng điểm giống hệt. Thư viện cũng không tự quyết định cách tokenize. [Mã nguồn rank_bm25](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py)

Lexical hữu ích với thuật ngữ, tên và mã. Query Việt trên sách Anh thường thiếu term chung; lowercase/tokenize không tạo khả năng dịch hay khớp xuyên ngôn ngữ.

## 3. Dense retrieval — tìm theo vector ngữ nghĩa

Bi-encoder encode query và tài liệu riêng. Với vector chuẩn hóa L2:

$$
s(q,d)=\hat v_q^\top\hat v_d=\cos(v_q,v_d),\qquad
R_k(q)=\operatorname{TopK}_{d\in\mathcal D}s(q,d).
$$

Vector tài liệu tính trước; khi hỏi chỉ encode query. Với E5, dùng `query: ` cho câu hỏi và `passage: ` cho tài liệu, giữ cùng model/revision và preprocessing. Không đủ để kiểm tra chỉ bằng số chiều. [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md)

Dense hỗ trợ tìm đoạn diễn đạt khác từ khóa; khả năng xuyên ngôn ngữ phụ thuộc model. Nó vẫn có thể nhầm con số, phủ định hoặc khái niệm gần nhau. Một vector gộp cả đoạn có thể làm mất chi tiết.

## 4. Neural sparse retrieval

Biểu diễn là các chiều term/token với trọng số học được; điểm thường là:

$$
s(q,d)=\sum_t w_t(q)w_t(d).
$$

Chiều không lưu có trọng số 0. SPLADE có thể mở rộng sang term không xuất hiện trực tiếp trong văn bản, thông qua dự đoán vocabulary. Đây là điểm khác với chỉ đếm term. [SPLADE paper](https://arxiv.org/abs/2107.05720)

BGE-M3 sparse trong notebook 04 chấm lexical weights theo token ID; không nên coi mọi neural sparse đều có cơ chế expansion giống SPLADE. Sparse cũng không tự bảo đảm tìm xuyên ngôn ngữ tốt.

## 5. Multi-vector / late interaction

Thay vì gộp cả đoạn thành một vector, giữ vector của token. Với query vectors `u_i`, document vectors `v_j`, ColBERT-style dùng MaxSim:

$$
s(q,d)=\sum_i\max_j u_i^\top v_j.
$$

Mỗi query token tìm token document phù hợp nhất rồi tổng hợp. Giữ tương tác chi tiết nhưng tốn bộ nhớ và scoring hơn single-vector. Có thể dùng cho lấy ứng viên với index chuyên dụng hoặc xếp hạng lại một tập nhỏ. [ColBERT paper](https://arxiv.org/abs/2004.12832)

BGE-M3 trong notebook 04 lấy **trung bình** các MaxSim query token; không đồng nhất điểm số với công thức tổng của ColBERT. Late interaction cũng khác cross-encoder: token được encode riêng trước khi tương tác.

## 6. Hybrid retrieval — kết hợp danh sách

Ví dụ BM25 tìm thuật ngữ, dense tìm ngữ nghĩa; fusion dùng thông tin cả hai. RRF tránh cộng trực tiếp các điểm BM25/cosine khác thang:

$$
s_{RRF}(d)=\sum_{r:d\in L_r}\frac{1}{c+rank_r(d)}.
$$

Rank bắt đầu 1; `c` là hằng số fusion, không phải số kết quả top-k. Tài liệu vắng ở một nhánh đóng góp 0. Hai nhánh phải dùng cùng corpus và identity để gộp đúng. [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

Ví dụ với `c=60`: tài liệu A đứng hạng 1 và 3 ở hai nhánh có điểm `1/61 + 1/63`; tài liệu B chỉ đứng hạng 1 một nhánh có `1/61`. RRF ưu tiên sự đồng thuận theo thứ hạng. Nhánh yếu vẫn có thể đưa nhiễu lên; hybrid không mặc định thắng dense.

## 7. Reranking — chấm kỹ tập ứng viên

Retriever lấy `candidate_k` đoạn; cross-encoder đọc **cặp query–đoạn** để chấm lại, rồi chọn top-k. Nó có tương tác attention giữa query và document, nên thường đắt hơn encode query một lần. Candidate 20 và top-k 5 là hai tham số khác nhau. [Sentence Transformers — Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

Reranker không thể lấy lại bằng chứng đã bị bỏ khỏi tập ứng viên. Vì vậy cần đo candidate hit trước rerank và thứ hạng sau rerank. Điểm raw/logit hoặc sigmoid không phải xác suất câu trả lời đúng đã được hiệu chuẩn. Reranker phải phù hợp ngôn ngữ, kiểm tra giới hạn token cho cặp input và đo thời gian thực tế.

## 8. Những trục khác cần biết

| Kỹ thuật | Vai trò | Phạm vi dự án hiện tại |
| --- | --- | --- |
| Exact kNN | Chấm toàn bộ vector hợp lệ | Dùng làm baseline |
| ANN, như HNSW/IVF | Tìm gần đúng để giảm công tìm trên corpus lớn | Học lý thuyết, chưa benchmark |
| Metadata filtering | Giới hạn sách/chapter/section trước lấy ứng viên | Triển khai |
| Query expansion/rewrite | Thêm term hoặc diễn đạt lại query | Mở rộng sau |
| MMR/diversification | Cân bằng liên quan với ít trùng lặp | Mở rộng sau |
| Hierarchical/multi-hop | Tìm theo cấp tài liệu hoặc nhiều bước | Mở rộng sau |

Exact/ANN không đổi model embedding: chúng đổi cách tìm láng giềng. Cần phân biệt **ANN recall so với exact neighbors** với **retrieval recall so với tài liệu liên quan**. Qdrant Local thực hiện brute-force/exact; có Qdrant không có nghĩa đang đo HNSW. [Qdrant Local source](https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/local/qdrant_local.py)

Filter phải áp vào tập hợp lệ trước chọn top-k; lấy top-5 rồi mới lọc có thể bỏ lỡ đoạn phù hợp ở hạng 6. Qdrant hỗ trợ query theo vector và filter payload; đọc bằng `retrieve(ids=...)` không phải tìm theo câu hỏi. [Qdrant Search](https://qdrant.tech/documentation/search/search/)

Query expansion có thể dựa vào feedback hoặc term liên quan; việc sửa query cũng có thể làm lệch ý ban đầu. Nó là chiến lược tác động đầu vào, không phải một embedding model mới. [Stanford — Query expansion](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-query-expansion-1.html)

## 9. Đánh giá Retrieval đúng cách

Với pilot có evidence gán trước, gọi `r_q` là hạng đầu tiên của chunk chứa trọn evidence:

$$
Hit@k=\frac1{|Q|}\sum_q\mathbf1[r_q\le k],\qquad
MRR@k=\frac1{|Q|}\sum_q\frac{\mathbf1[r_q\le k]}{r_q}.
$$

Nếu không có hit, đóng góp MRR bằng 0. Hit hỏi có tìm được ít nhất một bằng chứng không; MRR xét bằng chứng xuất hiện sớm đến đâu. Cần lưu top-k và nguồn để biết nguyên nhân miss, không chỉ nhìn số trung bình.

`Precision@k` đo tỷ lệ kết quả liên quan trong top-k; `Recall@k` đo tỷ lệ tập relevant đã tìm được. Muốn kết luận recall phải định nghĩa tập relevant đầy đủ trong phạm vi đánh giá. Gold chỉ gồm một evidence chọn trước không đại diện mọi đoạn trả lời hợp lệ. [Stanford — Precision/Recall](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)

`nDCG@k` phù hợp khi đánh giá mức liên quan theo nhiều cấp và muốn thưởng kết quả tốt ở hạng cao. Bộ câu hỏi phải đủ đa dạng; đừng tune rồi báo kết quả trên cùng bộ như một kiểm tra độc lập. [Stanford — Ranked evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)

Trong dự án: tách Anh/Việt, dev/eval; đo cả chất lượng và thời gian. Query ngoài corpus dùng để quan sát: dense top-k vẫn có thể trả đoạn gần nhất, không tự bảo đảm trả `[]`. Truy xuất đúng nguồn cũng chưa chứng minh LLM sẽ trả lời đúng.

## 10. Học và làm gì tiếp trong repo này?

1. BM25 để hiểu lexical ranking.
2. E5 + Qdrant để hiểu dense retrieval chạy từ index đã lưu.
3. BM25 + dense + RRF để hiểu fusion.
4. Dense/hybrid + cross-encoder để hiểu reranking và candidate ceiling.
5. Đối chiếu quality, latency và nguồn; đóng gói cùng API Retriever.

Đó là phần thực nghiệm riêng của Retrieval trong [retrieval_plan.md](../pipelines/retrieval_plan.md). Không cần thử lại tất cả model embedding hoặc hiện thực mọi kỹ thuật trong tài liệu này để hoàn thành phạm vi ấy.
