# Embedding Materials — lý thuyết các phương pháp biểu diễn văn bản

Embedding biến văn bản thành biểu diễn số để máy có thể so sánh, tìm kiếm, phân nhóm hoặc phân loại. Tài liệu trình bày các nhóm biểu diễn chính, công thức, cách hoạt động và đánh đổi; E5 và BGE-M3 được dùng làm ví dụ cho những kiến trúc cụ thể.

## Nền tảng chung

### Văn bản, token và vector

- **Token:** đơn vị do tokenizer tạo ra; có thể là một từ, một phần của từ hoặc ký hiệu.
- **Vector:** dãy tọa độ số. Số chiều $D$ không phải số token của đầu vào.
- **Pooling:** gộp nhiều vector token thành một vector đại diện văn bản.
- **Encode:** tạo biểu diễn; **retrieval:** tính điểm rồi lấy các văn bản phù hợp.

Dùng $q$ cho query, $d$ cho văn bản; $t_i$ là token ở vị trí $i$, $h_i$ là vector của token sau xử lý ngữ cảnh. Văn bản có thể là một câu, một đoạn hoặc tài liệu.

Có ba trục phân biệt:

| Cách phân biệt | Ý nghĩa |
| --- | --- |
| Sparse / dense | Nhiều chiều bằng 0 / thông tin phân bố trên các tọa độ thường khác 0 |
| Tĩnh / theo ngữ cảnh | Vector token cố định / vector token thay đổi theo chuỗi đầu vào |
| Single-vector / multi-vector | Một vector / nhiều vector để biểu diễn một văn bản |

Chiều sparse thường gắn với từ/token cụ thể. Chiều dense là đặc trưng được học, không mặc định tương ứng với một từ hay khái niệm có tên rõ ràng.

### Chuẩn hóa và độ tương đồng

Với vector khác 0:

$$
\|v\|_2=\sqrt{\sum_{j=1}^{D}v_j^2},\qquad
\hat v=\frac{v}{\|v\|_2}.
$$

Chuẩn hóa L2 đưa độ dài vector về 1, không đưa tổng tọa độ về 1. Ví dụ $[3,4]$ trở thành $[0.6,0.8]$.

$$
\cos(q,d)=\frac{v_q^\top v_d}{\|v_q\|_2\|v_d\|_2}.
$$

Nếu cả hai vector đã chuẩn hóa thì cosine bằng tích vô hướng. Khi đó $\|\hat v_q-\hat v_d\|_2^2=2-2\cos(q,d)$: cosine càng cao, khoảng cách Euclidean càng nhỏ. Nếu chưa chuẩn hóa, tích vô hướng còn phụ thuộc độ lớn vector. Xem [cosine và mô hình không gian vector](https://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html).

### Vector được học như thế nào?

TF-IDF lấy trọng số từ thống kê corpus. Các model neural học tham số từ dữ liệu: dự đoán từ xung quanh, học quan hệ giữa câu, hoặc phân biệt cặp query–document liên quan và không liên quan.

Một mục tiêu contrastive thường dùng trong retrieval là:

$$
\mathcal L(q)=-\log
\frac{\exp(s(q,d^+)/\tau)}
{\sum_{d\in\mathcal C_q}\exp(s(q,d)/\tau)}.
$$

$d^+$ là văn bản liên quan; $\mathcal C_q$ gồm văn bản đó và các văn bản đối chứng; $\tau>0$ điều khiển độ sắc của phân phối điểm. Giảm loss khuyến khích tăng điểm của cặp đúng so với các cặp còn lại. Cách học này được sử dụng trong [Multilingual E5](https://arxiv.org/abs/2402.05672).

**Pretraining** học biểu diễn trên dữ liệu lớn; **fine-tuning** điều chỉnh cho nhiệm vụ hoặc miền dữ liệu cụ thể; **inference** sử dụng tham số đã học để encode. Chất lượng phụ thuộc cả dữ liệu và mục tiêu huấn luyện, không chỉ kiến trúc hay số chiều.

## Chương 1. TF-IDF — biểu diễn sparse truyền thống

### Mô tả và công thức

TF-IDF tăng trọng số cho term xuất hiện nhiều trong một văn bản nhưng ít phổ biến trên toàn corpus. Term có thể là từ hoặc n-gram; mỗi term là một chiều vector.

Với $N$ văn bản, $DF(t)$ là số văn bản chứa term $t$, một dạng cơ bản là:

$$
TF(t,d)=\operatorname{count}(t,d),\qquad
IDF(t)=\log\frac{N}{DF(t)},\qquad
w_t(d)=TF(t,d)\,IDF(t).
$$

TF phản ánh sự xuất hiện trong văn bản; IDF phản ánh khả năng phân biệt giữa các văn bản. Từ xuất hiện ở hầu hết tài liệu thường ít hữu ích cho việc chọn tài liệu nào. Log hạn chế mức tăng của trọng số từ hiếm. Xem [lý thuyết IDF](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html).

Có nhiều biến thể. Chẳng hạn scikit-learn mặc định dùng IDF làm trơn:

$$
IDF(t)=\ln\frac{N+1}{DF(t)+1}+1.
$$

Một lựa chọn giảm ảnh hưởng việc lặp từ là $TF(t,d)=1+\log\operatorname{count}(t,d)$ khi term xuất hiện, và 0 khi không xuất hiện. Vector có thể được chuẩn hóa L2 rồi so bằng cosine. Xem [các lựa chọn TF-IDF trong scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).

### Đặc điểm, ưu điểm và nhược điểm

**Ưu điểm:** đơn giản, dễ giải thích, không cần model neural; phù hợp khớp thuật ngữ và làm baseline. Bigram giữ một phần thứ tự cục bộ, chẳng hạn phân biệt `database system` với các từ đứng riêng.

**Nhược điểm:** phụ thuộc từ/ngram chung; không tự nối từ đồng nghĩa, nghĩa đa ngôn ngữ hoặc ngữ cảnh dài. Từ hiếm có thể chỉ là nhiễu. Đếm tuyến tính làm từ lặp nhiều có ảnh hưởng lớn; chuẩn hóa chỉ giảm tác động của độ lớn toàn vector.

**Khi dùng:** tìm kiếm cùng ngôn ngữ với query bám sát thuật ngữ, phân loại văn bản cơ bản hoặc cần giải thích chiều nào đóng góp. Vocabulary và IDF được xác định trên corpus rồi dùng lại cho query; term ngoài vocabulary bị bỏ qua. Đổi corpus có thể thay IDF.

## Chương 2. Dense tĩnh — static embeddings

### Mô tả và cách học

Dense tĩnh gán một vector cố định cho mỗi từ/token. Cùng token ID được biểu diễn giống nhau trong mọi câu. “Tĩnh” nói về vector token, không phải vector cả câu luôn giống nhau, cũng không chỉ có nghĩa trọng số model được giữ nguyên lúc inference.

Các cách học tiêu biểu:

- **Word2Vec:** CBOW dự đoán từ từ ngữ cảnh lân cận; Skip-gram dự đoán các từ lân cận từ một từ. Xem [Word2Vec](https://arxiv.org/abs/1301.3781).
- **GloVe:** học từ thống kê đồng xuất hiện của các từ trên corpus. Xem [GloVe](https://nlp.stanford.edu/projects/glove/).
- **fastText:** kết hợp vector các n-gram ký tự để biểu diễn từ, hỗ trợ suy ra vector cho nhiều từ chưa gặp. Xem [fastText](https://arxiv.org/abs/1607.04606).

Các model static hiện đại còn có thể học trực tiếp từ cặp câu/query–document. Vì vậy static là một nhóm kiến trúc, không đồng nghĩa với riêng Word2Vec.

### Công thức và diễn giải

Với bảng $E\in\mathbb R^{V\times D}$, vector token là $e_i=E[t_i]$. Một cách tạo vector đoạn là trung bình có trọng số:

$$
v(d)=\frac{\sum_{i=1}^{n}\alpha_i e_i}{\sum_{i=1}^{n}\alpha_i}.
$$

$\alpha_i\ge0$ điều khiển đóng góp của token; đặt mọi $\alpha_i=1$ cho mean pooling. Sau đó có thể chuẩn hóa L2 và dùng cosine. Ví dụ hai vector $[0.8,0.2]$ và $[0.6,0.4]$ có trung bình $[0.7,0.3]$.

Mean pooling tạo một vector kích thước cố định dù độ dài văn bản thay đổi. `StaticEmbedding` là một cách hiện thực tra bảng và lấy trung bình các token đã học; xem [một model static dành cho retrieval](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1).

### Ưu điểm, nhược điểm và khi dùng

**Ưu điểm:** encode ít phép tính, tận dụng quan hệ ngữ nghĩa đã học; thuận tiện cho CPU hoặc xử lý nhiều văn bản. Có thể dùng vector từ trực tiếp hoặc gộp thành vector câu.

**Nhược điểm:** token đa nghĩa như `bank` có cùng vector dù chỉ ngân hàng hay bờ sông. Mean pooling không phân biệt thứ tự của cùng một tập token và có thể hòa loãng thông tin trong đoạn dài. Từ ngoài vocabulary phụ thuộc cách xử lý của model; không phải model nào cũng có cơ chế subword.

**Khi dùng:** ưu tiên tốc độ, tài nguyên hạn chế hoặc nhiệm vụ chủ yếu dựa vào nội dung từ vựng. Khả năng ngôn ngữ và chất lượng vector câu phụ thuộc model và cách pooling; vector từ tốt chưa bảo đảm truy xuất câu tốt.

## Chương 3. Dense theo ngữ cảnh — E5

### Mô tả và cơ chế

Contextual embedding tạo biểu diễn token dựa trên chuỗi và vị trí. E5 là một họ model sử dụng cơ chế này và được huấn luyện để biểu diễn các văn bản liên quan gần nhau. Các biến thể multilingual phục vụ nhiều ngôn ngữ; không phải mọi model contextual đều đa ngôn ngữ.

$$
(h_1,\ldots,h_n)=Transformer(t_1,\ldots,t_n).
$$

Một thành phần của Transformer là self-attention. Với một attention head, dạng cơ bản là:

$$
Attention(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_k}}\right)V.
$$

$Q,K,V$ được tạo từ các biểu diễn đầu vào; $D_k$ là số chiều của key. Điểm giữa query và key quyết định mức đóng góp của value từ các vị trí. Đây là attention bên trong một văn bản, khác việc so query tìm kiếm với document. Kết hợp attention, thông tin vị trí và các lớp biến đổi tạo nên vector theo ngữ cảnh. Xem [Transformer](https://arxiv.org/abs/1706.03762).

### Công thức biểu diễn và truy xuất

Một cách pooling của các model multilingual E5 là trung bình có mask:

$$
v(d)=\frac{\sum_i m_i h_i}{\sum_i m_i},\qquad
\hat v(d)=\frac{v(d)}{\|v(d)\|_2},\qquad
s(q,d)=\hat v(q)^\top\hat v(d).
$$

$m_i=0$ cho padding và 1 cho token hợp lệ. Khác static, phép trung bình ở đây gộp các vector **đã xử lý ngữ cảnh**; đổi câu có thể đổi $h_i$ của cùng token.

Query và document được encode riêng bằng bi-encoder; vector document có thể lưu trước. Các model `multilingual-e5-small/base/large` sử dụng tiền tố `query: ` và `passage: ` để phân biệt vai trò, kể cả với nội dung không phải tiếng Anh. Các biến thể E5 khác có thể có quy tắc instruction riêng; cần theo hướng dẫn model. Xem [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md).

### Ưu điểm, nhược điểm và khi dùng

**Ưu điểm:** biểu diễn cách dùng từ trong câu; phù hợp semantic retrieval, paraphrase và truy xuất xuyên ngôn ngữ với model hỗ trợ. Một vector mỗi văn bản giúp lưu và tìm kiếm thuận tiện.

**Nhược điểm:** encode tốn tài nguyên hơn static; một vector có thể mất chi tiết. Giới hạn token, pooling và miền dữ liệu ảnh hưởng chất lượng; ngữ cảnh không bảo đảm model hiểu đúng mọi câu.

**Khi dùng:** query diễn đạt khác tài liệu hoặc cần tìm theo nghĩa. Phải giữ đúng model, tokenizer, tiền tố và normalization ở hai phía. Kiểm tra độ dài đầu vào trước khi cắt; số chiều vector không phải giới hạn token.

## Chương 4. BGE-M3 dense — đại diện văn bản bằng token đặc biệt

### Mô tả và công thức

BGE-M3 dùng một backbone theo ngữ cảnh cho ba nhánh dense, sparse và multi-vector. Nhánh dense lấy hidden state của token đặc biệt `[CLS]` làm đại diện văn bản, rồi chuẩn hóa:

$$
H(d)=Transformer(d),\qquad
v(d)=\frac{h_{\mathrm{CLS}}(d)}{\|h_{\mathrm{CLS}}(d)\|_2},\qquad
s_{dense}(q,d)=v(q)^\top v(d).
$$

`[CLS]` tiếp nhận thông tin qua các lớp contextual và được huấn luyện để phục vụ biểu diễn toàn văn bản. Đây là cách pooling khác mean pooling của E5; cả hai vẫn thuộc nhóm single-vector dense theo ngữ cảnh. Xem [kiến trúc BGE-M3](https://arxiv.org/html/2402.03216v2#S3.SS2).

### Ưu điểm, nhược điểm và khi dùng

**Ưu điểm:** một vector mỗi văn bản; hỗ trợ semantic retrieval đa ngôn ngữ và có thể phối hợp các nhánh khác của cùng model.

**Nhược điểm:** vẫn nén nhiều thông tin vào một vector; tài nguyên encode và chất lượng phụ thuộc model. Dùng `[CLS]` không mặc định tốt hơn lấy trung bình, và số chiều lớn hơn không bảo đảm chính xác hơn.

**Khi dùng:** cần contextual dense retrieval hoặc một nền tảng có nhiều nhánh biểu diễn. BGE-M3 có hướng dẫn đầu vào riêng, không áp dụng máy móc tiền tố của E5; xem [model card BGE-M3](https://huggingface.co/BAAI/bge-m3).

Khác biệt kết quả giữa hai model còn đến từ dữ liệu, tokenizer và cách huấn luyện; không thể quy tất cả cho pooling.

## Chương 5. Neural sparse — BGE-M3 sparse

### Mô tả và công thức

Neural sparse tạo vector thưa với trọng số được học. Trong BGE-M3, trọng số token phụ thuộc hidden state theo ngữ cảnh, thay vì thống kê TF × IDF.

Theo dạng công thức của BGE-M3, trọng số ở vị trí $i$ là:

$$
a_i(d)=\operatorname{ReLU}(w_\ell^\top h_i(d)),\qquad
\operatorname{ReLU}(x)=\max(0,x).
$$

$w_\ell$ là vector tham số được học. Nếu cùng token ID $t$ xuất hiện nhiều lần thì giữ trọng số lớn nhất:

$$
w_t(d)=\max_{i:t_i=t}a_i(d),\qquad
s_{sparse}(q,d)=\sum_{t\in T(q)\cap T(d)}w_t(q)w_t(d).
$$

$T(x)$ là tập token được lưu. Chiều không lưu có trọng số 0; token không xuất hiện được xem là 0. Công thức scoring nối các token ID chung, không tự nối hai từ chỉ vì chúng đồng nghĩa. Xem [nhánh lexical của BGE-M3](https://arxiv.org/html/2402.03216v2#S3.SS2).

### Đặc điểm, ưu điểm và nhược điểm

**Ưu điểm:** tận dụng ngữ cảnh để học mức quan trọng nhưng vẫn xem được token nào đóng góp vào điểm. Đầu ra có thể lưu thành cặp token ID–weight và tìm bằng index lexical.

**Nhược điểm:** encode vẫn cần model neural; sparse không đồng nghĩa encode nhẹ. Khớp token chung có thể yếu khi query và document khác ngôn ngữ. Điểm tích vô hướng không mặc định là cosine nếu không chuẩn hóa.

**Khi dùng:** cần tín hiệu thuật ngữ có trọng số học được, hoặc làm nhánh lexical trong hybrid. Token ID phải thuộc cùng vocabulary; không trộn trực tiếp vector TF-IDF với vector BGE sparse vì các chiều không tương ứng.

**Phân biệt với SPLADE:** BGE-M3 gán trọng số cho token xuất hiện trong đầu vào. SPLADE có thể mở rộng sang term không xuất hiện trực tiếp, giúp giảm vấn đề lệch từ vựng; vì vậy neural sparse không chỉ có một cách biểu diễn. Xem [SPLADE](https://arxiv.org/abs/2107.05720).

## Chương 6. Multi-vector — ColBERT và BGE-M3

### Mô tả và công thức

Multi-vector giữ nhiều vector theo token để tránh gộp mọi thông tin quá sớm. Query và document được encode riêng; khi search, các vector token tương tác. Cách này gọi là **late interaction**, tiêu biểu là [ColBERT](https://arxiv.org/abs/2004.12832).

Gọi $P$ là phép chiếu đã học. Một dạng biểu diễn token chuẩn hóa là:

$$
u_i=\frac{P(h_i(q))}{\|P(h_i(q))\|_2},\qquad
v_j=\frac{P(h_j(d))}{\|P(h_j(d))\|_2}.
$$

ColBERT dùng tổng MaxSim; BGE-M3 dùng trung bình MaxSim:

$$
s_{ColBERT}(q,d)=\sum_{i=1}^{n_q}\max_{1\le j\le n_d}u_i^\top v_j,
$$

$$
s_{BGE\text{-}multi}(q,d)=\frac{1}{n_q}\sum_{i=1}^{n_q}\max_{1\le j\le n_d}u_i^\top v_j.
$$

Mỗi query token chọn document token có độ tương đồng cao nhất. Sau đó tổng hợp những điểm tối đa; không cộng điểm mọi cặp và không lấy trung bình vector trước khi so. Biến thể BGE-M3 được mô tả trong [bài báo BGE-M3](https://arxiv.org/html/2402.03216v2#S3.SS2).

Với một query cố định, tổng và trung bình MaxSim xếp hạng giống nhau nếu dùng đúng cùng các vector và token hợp lệ; chia cho $n_q$ thay thang điểm giữa các query có độ dài khác nhau. Hai model ColBERT và BGE-M3 vẫn khác nhau về huấn luyện và biểu diễn.

### Ưu điểm, nhược điểm và khi dùng

**Ưu điểm:** đối chiếu chi tiết từng phần query với văn bản; document vẫn có thể encode trước. Có thể chấm lại một tập ứng viên từ retriever nhanh hơn.

**Nhược điểm:** lưu khoảng $n_dD$ giá trị thay vì $D$ cho mỗi văn bản; scoring trực tiếp một cặp tốn khoảng $O(n_qn_dD)$. MaxSim không yêu cầu các token được chọn nằm cạnh nhau, nên điểm cao chưa bảo đảm có bằng chứng trả lời hoàn chỉnh.

**Khi dùng:** cần truy xuất chi tiết hoặc rerank tập ứng viên vừa phải và chấp nhận thêm tài nguyên. Ma trận nhiều vector không tự động giữ toàn bộ cấu trúc văn bản; pooling theo MaxSim vẫn là một quy tắc tổng hợp cụ thể.

## Chương 7. Hybrid retrieval — kết hợp các tín hiệu

### Mô tả và công thức

Hybrid kết hợp các retriever như dense và lexical để tận dụng tín hiệu bổ sung. Đây là chiến lược truy xuất, không phải một họ embedding mới; có thể dùng những model độc lập.

Hai cách phổ biến:

**Cộng điểm có trọng số:**

$$
s(q,d)=\lambda s_{dense}(q,d)+(1-\lambda)s_{lexical}(q,d),\qquad 0\le\lambda\le1.
$$

Cần hiệu chỉnh các thang điểm và chọn trọng số phù hợp. Nếu một nhánh có điểm lớn hơn về mặt số học, nó có thể lấn át nhánh còn lại dù không chính xác hơn.

**Reciprocal Rank Fusion — RRF:**

$$
s_{RRF}(q,d)=\sum_{r\in\mathcal R_d}\frac{1}{k+\operatorname{rank}_r(q,d)}.
$$

$\mathcal R_d$ gồm các danh sách có văn bản $d$; rank bắt đầu từ 1, $k>0$ làm giảm chênh lệch đóng góp giữa các hạng. Nhánh không có $d$ đóng góp 0. RRF dùng thứ hạng nên không cần cộng trực tiếp điểm gốc. Xem [bài báo RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

### Ưu điểm, nhược điểm và khi dùng

**Ưu điểm:** phối hợp khớp thuật ngữ với khớp ngữ nghĩa; có thể tăng độ bao phủ nếu các nhánh tìm được những kết quả hữu ích khác nhau.

**Nhược điểm:** cần chạy và quản lý nhiều nhánh; nhánh yếu có thể đưa nhiễu lên cao. RRF mất thông tin về khoảng cách điểm, còn weighted sum cần hiệu chỉnh. Cả hai không bảo đảm tốt hơn retriever mạnh nhất.

**Khi dùng:** query có cả thuật ngữ chính xác và cách diễn đạt lại, hoặc từng retriever bỏ sót những trường hợp khác nhau. Chọn số ứng viên mỗi nhánh, quy tắc fusion và số kết quả cuối cùng như các quyết định riêng.

## Chương 8. So sánh, phân biệt và lựa chọn

### So sánh các nhóm

| Phương pháp | Biểu diễn | Tín hiệu chính | Ưu thế | Hạn chế |
| --- | --- | --- | --- | --- |
| TF-IDF | Sparse term/ngram | Tần suất và độ hiếm | Đơn giản, dễ giải thích | Phụ thuộc từ chung |
| Static dense | Một vector sau pooling | Quan hệ token đã học | Encode ít phép tính | Không đổi nghĩa token theo câu |
| Contextual dense, như E5 | Một vector sau pooling | Ngữ nghĩa theo ngữ cảnh | Paraphrase, semantic retrieval | Chi phí encode; mất chi tiết khi gộp |
| BGE-M3 dense | Một vector `[CLS]` | Ngữ nghĩa theo ngữ cảnh | Cùng nền tảng với nhiều nhánh | Có đánh đổi của single-vector dense |
| Neural sparse | Token IDs và trọng số | Lexical có trọng số học được | Có thể xem đóng góp token | Chi phí neural; phụ thuộc cách khớp/expansion |
| Multi-vector | Nhiều vector token | Tương tác chi tiết | Khớp từng thành phần query | Dung lượng và scoring cao |
| Hybrid | Kết hợp danh sách/điểm | Nhiều tín hiệu | Tận dụng sự bổ sung | Nhiễu từ nhánh yếu, nhiều thành phần |

E5 và BGE dense là các model cùng nhóm contextual dense; BGE dense/sparse/multi là ba nhánh của một model. Một model đa ngôn ngữ không làm cho mọi kiểu scoring đều truy xuất xuyên ngôn ngữ tốt như nhau.

TF-IDF thuộc nhóm lexical; [BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) là một hàm lexical khác, có điều chỉnh bão hòa tần suất và độ dài tài liệu, không đồng nhất với TF-IDF. Lexical retrieval cũng không phải lúc nào được gọi là embedding theo nghĩa model neural.

### Khi nên dùng gì?

- **Khớp thuật ngữ, mã, tên riêng:** bắt đầu với lexical retrieval; TF-IDF là lựa chọn đơn giản để hiểu trọng số.
- **Ưu tiên encode nhanh trên CPU:** cân nhắc static với ngôn ngữ và miền dữ liệu được hỗ trợ.
- **Query diễn đạt khác tài liệu:** cân nhắc contextual dense được huấn luyện cho retrieval.
- **Query và tài liệu khác ngôn ngữ:** chọn model multilingual phù hợp; khả năng đó đến từ dữ liệu huấn luyện, không chỉ vì vector là dense.
- **Cần khớp chi tiết từng phần query:** cân nhắc multi-vector, thường trên tập ứng viên nhỏ hơn.
- **Cần cả lexical và semantic:** kết hợp các nhánh bằng hybrid khi chúng bổ sung nhau.

Các lựa chọn này suy ra từ cấu trúc và đánh đổi của phương pháp. Lựa chọn cuối cần dựa vào chất lượng trên dữ liệu mục tiêu, độ trễ, tài nguyên và khả năng vận hành.

### Những phân biệt quan trọng

1. **Embedding không phải vector database:** model tạo biểu diễn; database/index lưu và tìm trên biểu diễn đó.
2. **Bi-encoder không phải cross-encoder:** bi-encoder encode query/document riêng; cross-encoder đọc từng cặp để chấm điểm, thường dùng reranking. Late interaction giữ vector token riêng rồi tương tác khi scoring.
3. **Similarity không phải xác suất đúng:** từ gần nhau về ngữ nghĩa vẫn có thể khác về phủ định, con số hoặc quan hệ. Nhìn vài tọa độ không đủ chứng minh model hiểu văn bản.
4. **Cùng số chiều không có nghĩa cùng không gian:** query và document phải có biểu diễn tương thích. Đổi model, tokenizer hoặc pooling có thể cần encode lại corpus.
5. **Vector nhỏ không có nghĩa model nhẹ:** dung lượng output khác với trọng số model, bộ nhớ inference và chi phí index. Đo encode và search như hai bước riêng.
6. **Input dài không bảo đảm biểu diễn tốt:** giới hạn token là khả năng nhận đầu vào; pooling, độ dài và cấu trúc nội dung vẫn ảnh hưởng tín hiệu. Không âm thầm cắt phần có thông tin cần tìm.
7. **Retrieval đúng chưa bảo đảm câu trả lời đúng:** trong RAG, văn bản truy xuất còn phải được đưa vào mô hình sinh và sử dụng đúng khi tạo câu trả lời.
