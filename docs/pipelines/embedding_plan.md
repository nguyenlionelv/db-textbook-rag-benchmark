# Kế hoạch Embedding và Vector Storage

Trạng thái: đề xuất để xem xét, chưa chốt công nghệ và chưa triển khai code.

Ngày đối chiếu tài liệu công nghệ: 28/09/2026. Các mức giá bên dưới là giá tham khảo tại thời điểm này, không phải dự toán chi phí toàn bộ sách.

## 1. Mục tiêu của phần này

Hoàn thiện phần còn lại của ingestion theo pipeline:

```text
PDF → Parser → Chunker → Embedding → Vector Storage
                                      ↓
                             Kiểm tra lưu và tìm kiếm
```

Kết quả cần đạt:

1. Nhận đúng `Document` từ ba Chunker, không làm mất nội dung hoặc metadata truy vết.
2. Biến mỗi chunk thành một vector bằng model được chọn.
3. Lưu vector cùng nội dung, ID sách và metadata để mở lại ở lần chạy sau.
4. Chạy lại cùng dữ liệu/cấu hình không tạo thêm bản sao chunk.
5. Nhập câu hỏi, tìm những chunk liên quan và hiển thị được nguồn PDF.
6. Chạy ổn trên dữ liệu mẫu, sau đó mới mở rộng sang toàn bộ sách.
7. Có đủ hàm để một giao diện nhỏ gọi lại pipeline, không phải viết lại logic.

Embedding model và vector database là hai lựa chọn độc lập. Model tạo vector; database lưu và tìm kiếm vector. Kiểm thử retrieval ban đầu được đưa vào kế hoạch để xác nhận ingestion hữu dụng; phần sinh câu trả lời bằng LLM chưa thuộc phạm vi này.

Chưa làm ở phiên bản đầu: hybrid search, reranker, fine-tuning, OCR, xử lý bảng/hình chuyên sâu, chatbot hoàn chỉnh, hàng đợi công việc, phân quyền nhiều người dùng hoặc triển khai cloud.

## 2. Điểm xuất phát của dự án

- Parser đã có schema record và `document_id` dựa trên nội dung PDF; đổi tên file không đổi ID sách.
- Chunker đã có contract tại [chunker_contract.md](chunker_contract.md), đầu ra là `Document(page_content, metadata)`.
- Ba chiến lược: `strict_fixed`, `global_recursive`, `structure_aware_recursive`.
- Cấu hình hiện tại: `chunk_size=1000` ký tự, `chunk_overlap=150` ký tự, `include_footnotes=False`.
- `output/parsed_chapter1.jsonl` là tên cũ nhưng chứa dữ liệu của Chapter 1, 2 và 3, không phải toàn bộ sách.
- Output mẫu hiện có lần lượt 189, 241 và 241 chunk cho ba chiến lược.
- Metadata đã có chapter, section, offset và `sources`; chưa có ID chunk duy nhất toàn sách hoặc thông tin embedding.
- Chưa có code embedding, vector storage hoặc retrieval.
- Môi trường `.venv` đang chạy Python 3.13.0; README hướng dẫn Python 3.11. Khi triển khai cần kiểm tra khả năng cài dependency trên môi trường thực tế, không tự thay môi trường hiện có.

Hai điểm cần giữ đúng khi nối pipeline:

1. `chunk_size` hiện tính bằng **ký tự**, trong khi model giới hạn theo **token**. Phải đếm token bằng tokenizer của model được chọn.
2. Chunker hiện báo lỗi nếu record còn lại sau lọc thiếu `chapter_number`. Trước khi ingest toàn sách cần rà soát và chốt cách xử lý nội dung ngoài chapter hoặc chapter chưa nhận diện được.

## 3. Các lựa chọn để quyết định

### 3.1. Embedding model

Sách hiện là tiếng Anh. Đề xuất bộ kiểm tra có cả câu hỏi tiếng Anh và tiếng Việt, vì tìm câu hỏi tiếng Việt trong nội dung tiếng Anh là một tình huống khác với tìm cùng ngôn ngữ. Nếu chỉ cần tiếng Anh, có thể thu hẹp sau khi chốt.

| Mã | Model | Chạy ở đâu | Số chiều | Giới hạn đầu vào | Khi nào nên chọn |
| --- | --- | --- | --- | --- | --- |
| M1 | `intfloat/multilingual-e5-small` | Local | 384 | 512 token | Muốn học pipeline, không gửi nội dung lên API, bắt đầu với model nhỏ. |
| M2 | `BAAI/bge-m3`, chỉ dùng dense embedding | Local | 1024 | 8192 token | Muốn thử model đa ngôn ngữ với giới hạn dài hơn; sẵn sàng đo tải model và thời gian chạy trên máy. |
| M3 | `text-embedding-3-small` | OpenAI API | 1536 mặc định | 8192 token | Muốn giảm việc vận hành model local; chấp nhận API key, gửi text ra ngoài và chi phí theo token. |
| M4 | `text-embedding-3-large` | OpenAI API | 3072 mặc định | 8192 token | Lựa chọn đối chứng sau khi đo retrieval; chưa cần bắt đầu bằng model này. |

Thông số và cách dùng đối chiếu từ [model card E5](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md), [model card BGE-M3](https://huggingface.co/BAAI/bge-m3) và [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings).

Lưu ý triển khai theo model:

- M1: document dùng tiền tố `passage: `, query dùng `query: `; chuẩn hóa vector theo hướng dẫn model. Đếm token sau khi thêm tiền tố và tính cả special token. Không để thư viện âm thầm cắt phần vượt giới hạn. [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md).
- M2: phiên bản đầu chỉ lấy dense vector; chưa dùng sparse hoặc ColBERT. Không áp dụng tiền tố E5 cho BGE-M3. [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
- M3/M4: bắt đầu với số chiều mặc định; giảm chiều là một cấu hình thí nghiệm khác, không tự thay giữa ingestion và query. [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings).
- Tốc độ thực tế phải đo trên máy; hiện chưa có thông tin RAM/GPU để cam kết thời gian chạy model local.
- Không kết luận model nào tốt nhất cho sách này chỉ từ tên, kích thước hoặc benchmark chung. Quyết định chất lượng dựa vào bộ câu hỏi của dự án.

Giá API tham khảo cho request tiêu chuẩn:

| Model | USD / 1 triệu input token |
| --- | --- |
| `text-embedding-3-small` | 0.02 |
| `text-embedding-3-large` | 0.13 |

Nguồn: [OpenAI small model](https://developers.openai.com/api/docs/models/text-embedding-3-small), [OpenAI large model](https://developers.openai.com/api/docs/models/text-embedding-3-large).

Dự toán phải tính trên tổng token của text thực sự gửi, gồm heading và phần overlap xuất hiện nhiều lần. Ví dụ giả định 1 triệu token cần embed mới có chi phí tương ứng 0.02 hoặc 0.13 USD; không suy ra rằng toàn bộ sách hiện tại có đúng lượng token này. Query phát sinh thêm token. Retry sau lỗi không rõ trạng thái có thể tạo thêm lượt tính phí; cache chỉ giảm việc gửi lại text đã lưu kết quả.

Local không có phí API theo token, nhưng vẫn có chi phí tải model, dung lượng, thời gian chạy và tài nguyên máy. Nếu chọn API, cần xác nhận chính sách dữ liệu/quyền sử dụng nội dung và ngân sách trước khi gửi cả sách; không ghi API key vào code hoặc Git.

### 3.2. Vector database / thư viện index

| Mã | Lựa chọn | Điểm thuận lợi cho dự án | Phần phải chấp nhận | Đánh giá đề xuất |
| --- | --- | --- | --- | --- |
| V1 | Qdrant local, lưu trên đĩa | Không cần server lúc thử nghiệm; payload JSON giữ được metadata lồng nhau của Chunker. | Phù hợp prototype; khi cần truy cập đồng thời hoặc mở rộng phải chuyển sang server và kiểm tra lại. | Ưu tiên hiện tại. |
| V2 | Chroma `PersistentClient` | API Python trực tiếp, lưu persistent; hợp với notebook/prototype. | Metadata `sources` dạng list các dictionary cần chuyển đổi để lưu, không truyền nguyên cấu trúc vào các trường metadata thông thường. | Hợp nếu ưu tiên Chroma và chấp nhận adapter metadata. |
| V3 | PostgreSQL + pgvector | Lưu text, metadata JSONB và vector trong hệ SQL; dễ nối với bảng sách/thí nghiệm. | Cần PostgreSQL, extension, kết nối và schema SQL; nhiều thiết lập hơn local Python client. | Hợp nếu muốn dùng SQL lâu dài hoặc đã có PostgreSQL. |
| V4 | FAISS + file metadata riêng | Hợp nghiên cứu thuật toán tìm kiếm vector và làm baseline. | FAISS là thư viện index; cần tự ghép quản lý text, metadata, filter, cập nhật và persistence. | Không ưu tiên cho ingestion đơn giản hiện tại. |

Cơ sở so sánh: [Qdrant local client](https://github.com/qdrant/qdrant-client#local-mode), [Qdrant payload](https://qdrant.tech/documentation/manage-data/payload/), [Chroma client](https://docs.trychroma.com/reference/python), [Chroma metadata](https://docs.trychroma.com/docs/collections/add-data), [pgvector](https://github.com/pgvector/pgvector), [PostgreSQL JSONB](https://www.postgresql.org/docs/current/datatype-json.html), [FAISS](https://github.com/facebookresearch/faiss).

Các điểm cần biết trước khi chọn:

- V1: dùng đường dẫn lưu trên đĩa, không dùng `:memory:` cho dữ liệu sách. Local mode có khóa thư mục; không mở cùng storage từ hai client/process độc lập. Giao diện local ban đầu nên chạy ingestion và query tuần tự trong một process, dùng lại client; nếu CLI và UI phải truy cập đồng thời thì chọn Qdrant server. [Client source](https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/local/qdrant_local.py).
- V2: tài liệu hiện cho phép scalar và array scalar đồng nhất; array rỗng không được chấp nhận. `sources` không phải array scalar. Đề xuất lưu metadata dễ lọc riêng và lưu toàn bộ metadata dưới dạng một chuỗi JSON, giải mã khi đọc; bỏ trường filter có giá trị không được backend hỗ trợ, không thay `null` bằng dữ liệu giả. [Chroma metadata](https://docs.trychroma.com/docs/collections/add-data).
- V3: `text-embedding-3-large` mặc định 3072 chiều cần lưu ý nếu muốn HNSW/IVFFlat: index trên kiểu `vector` hiện hỗ trợ tối đa 2000 chiều; có lựa chọn `halfvec` đến 4000 chiều hoặc giảm chiều qua API. Đây là giới hạn của cách index, không có nghĩa PostgreSQL không lưu được vector 3072 chiều. [pgvector index types](https://github.com/pgvector/pgvector#hnsw).
- Không dùng kết quả tốc độ Qdrant local để kết luận hiệu năng Qdrant server. Nếu sau này đánh giá nhiều vector database, dùng cùng tập vector đã cache để tách ảnh hưởng của model khỏi database.

### 3.3. Các cấu hình khởi đầu

| Phương án | Model + storage | Phù hợp với ưu tiên |
| --- | --- | --- |
| A — local, dễ học | M1 + V1 | Không phí API theo token, giữ nội dung local sau khi tải model, pipeline có thể quan sát từng bước. |
| B — API, giảm vận hành model | M3 + V1 | Muốn tập trung vào ingestion/retrieval, chấp nhận gửi text qua API. |
| C — local, thử model lớn hơn | M2 + V1 | Muốn dùng BGE-M3 và có thời gian/tài nguyên thử nghiệm. |
| D — nền tảng SQL | M1 hoặc M3 + V3 | Muốn quản lý sách và thí nghiệm trên PostgreSQL từ đầu. |

Đề xuất cho nhu cầu hiện tại: **A nếu ưu tiên local; B nếu ưu tiên triển khai nhanh bằng API**. Đây là đánh giá về độ phù hợp và công triển khai, chưa phải kết luận chất lượng retrieval. Chỉ hiện thực một model và một storage đã chọn trước; interface nhỏ cho phép thêm lựa chọn khác sau này.

### 3.4. Các quyết định không phải công nghệ

| Quyết định | Đề xuất ban đầu | Lựa chọn khác |
| --- | --- | --- |
| Chiến lược chunk chạy trước | `structure_aware_recursive` làm baseline dễ truy vết. | Chạy cả ba ngay, nhưng tách tập index khi tìm kiếm. |
| Dữ liệu thử đầu tiên | JSONL mẫu hiện có, Chapter 1–3. | Một chapter nhỏ được chọn lại. |
| Ngôn ngữ query | Tiếng Anh và tiếng Việt. | Chỉ tiếng Anh. |
| Footnote | Giữ mặc định `False`; vẫn giữ toàn bộ Parser output để tra note. | Bật `True` và đánh giá như một cấu hình khác. |
| Nội dung thiếu chapter | Chưa tự bỏ hoặc tự gán; báo cáo để bạn duyệt trước khi mở rộng. | Loại rõ ràng; bổ sung context đúng; hoặc thiết kế nhóm ngoài chapter và sửa contract/test. |
| Giao diện | Làm sau khi pipeline CLI và persistence đã đạt. | Làm sớm để quan sát, nhưng vẫn gọi cùng các hàm đã test. |

Không coi structure-aware là chiến lược tốt nhất khi chưa đánh giá retrieval. Giữ nguyên ba cách đã khớp notebook để có baseline so sánh.

## 4. Contract tối thiểu Chunker → Embedding → Storage

Mục này là đề xuất cần duyệt ở Bước 1, chưa sửa contract Chunker đã chốt.

### 4.1. Input embedding

- Nhận danh sách `Document` của Chunker.
- Text đưa vào model lấy từ `page_content`, giữ cách dựng text hiện tại của thực nghiệm.
- Tiền tố bắt buộc của model chỉ thêm trong bước encode. Không ghi tiền tố vào `page_content` hoặc thay các offset đã có.
- Chưa thêm lại chapter title vào từng chunk, dịch text, bỏ heading hoặc làm sạch lần nữa. Những thay đổi đó ảnh hưởng thí nghiệm và cần cấu hình/test riêng nếu muốn làm sau.
- Metadata không được đưa nguyên JSON vào model; nó phục vụ lưu trữ, filter và truy vết.
- `document_id`/tên sách/đường dẫn Parser output do caller truyền cùng dữ liệu, không lấy `chunk_index` làm ID sách. Với pipeline PDF lấy `parser.document_id`; nếu bắt đầu từ JSONL thì cần manifest kèm ID đã xác nhận.

### 4.2. Output để lưu

| Trường | Vai trò |
| --- | --- |
| `chunk_id` | UUID xác định, duy nhất trong tập chunk; dùng làm ID upsert. |
| `document_id` | ID sách từ Parser, không phụ thuộc tên file. |
| `chunk_set_id` | Fingerprint của input Parser, phạm vi/policy chọn dữ liệu, phiên bản contract và cấu hình Chunker. |
| `embedding_profile_id` | Fingerprint của model/provider, revision nếu có, số chiều, tiền tố và quy tắc normalization. |
| `page_content` | Nội dung chunk gốc để hiển thị và đưa vào retrieval sau này. |
| `metadata` | Toàn bộ metadata Chunker, gồm `sources`; không cắt bỏ để tiết kiệm công adapter. |
| `vector` | Dãy số hữu hạn, đúng số chiều model/profile, norm khác 0. |

`chunk_index` hiện bắt đầu lại theo chapter nên không đủ duy nhất. Đề xuất tạo `chunk_id` bằng UUID5 từ chuỗi xác định gồm `document_id`, `chunk_set_id`, chapter, phạm vi offset, section unit nếu có, chunk index, start/end và hash nội dung. Namespace UUID phải cố định, không tạo ngẫu nhiên mỗi lần chạy.

ID chunk không cần chứa tên model: cùng chunk có thể embed bằng nhiều model trong các index/profile khác nhau. Cache vector dùng khóa riêng là `embedding_profile_id + hash(text thực sự encode)`; hai chunk có cùng text vẫn có ID và metadata riêng.

Lưu toàn bộ cấu hình/fingerprint đầu vào trong một manifest JSON nhỏ. Không cần xây hệ thống registry hoặc version manager ở phiên bản đầu.

### 4.3. Collection và quản lý phiên bản

Đề xuất đơn giản cho một sách ban đầu:

- Một collection cho một sách + một `chunk_set_id` + một `embedding_profile_id`; không tách collection theo từng chapter.
- Tên kỹ thuật có thể dùng các hash rút gọn; manifest giữ giá trị đầy đủ để kiểm tra khớp, không dựa riêng vào tên.
- Cùng input/cấu hình thì dùng lại collection và cùng IDs; upsert không tăng số bản ghi.
- Đổi model, chiều vector hoặc tiền xử lý embedding thì dùng profile/index mới và embed lại hoặc dùng cache đúng profile.
- Đổi cấu hình chunking, parser output hoặc policy chọn dữ liệu thì dùng tập chunk/index mới.
- Chưa tự xóa index cũ. Chỉ query index đã hoàn tất và được chọn; index dở dang không được công bố là ingest thành công.
- Khi so sánh ba chiến lược, tìm trên từng collection riêng. Không lấy top-k từ một collection trộn cả ba để đánh giá riêng từng chiến lược.

Cách này tránh vấn đề: upsert cập nhật các ID có trong lần chạy mới nhưng **không tự xóa** các chunk cũ đã biến mất sau khi đổi cách parse/chunk. Nếu về sau muốn gom nhiều sách vào một collection chung, cần thiết kế replace-by-document/version rõ ràng ở bước mở rộng, không làm ngầm ngay từ đầu.

### 4.4. Truy vết và kết quả tìm kiếm

- Kết quả gồm `chunk_id`, điểm tương đồng, `page_content` và metadata đủ để xem chapter, section, record nguồn và trang PDF.
- Trang lấy từ `sources[].pdf_page`; có thể có nhiều trang và nhiều record. Metadata section có thể là `null` với chunk giao nhiều section, không giả định luôn có một section.
- `pdf_page` vẫn đánh số từ 0. Giao diện có thể hiển thị `pdf_page + 1` và ghi rõ đây là số trang PDF, không phải số trang in trong sách.
- Không diễn giải `bbox` toàn record thành bbox chính xác của phần chunk bị cắt.
- Không đổi `footnote_refs.raw_offset` thành offset của chunk. Giữ Parser output để tra `footnote_id` kể cả khi không embed note.
- Chunk chỉ chứa heading có thể có `sources=[]`. Báo cáo số trường hợp; nếu gặp thì duyệt policy trước khi index, không gán nguồn giả hoặc âm thầm bỏ.
- Chọn cosine cho phiên bản đầu. Adapter trả về một quy ước điểm thống nhất, điểm cao hơn là gần hơn; không gọi điểm đó là xác suất câu trả lời đúng và không so trực tiếp điểm tuyệt đối giữa các model.

## 5. Kế hoạch triển khai từng bước và điểm dừng

### Bước 0 — Chốt cấu hình khởi đầu

Việc cần quyết định:

1. Chọn model M1–M4 hoặc một phương án A–D.
2. Chọn storage V1–V4 và local/server nếu dùng Qdrant.
3. Xác nhận ngôn ngữ query, chiến lược baseline và footnote policy.
4. Nếu local: cung cấp RAM/GPU hoặc cho phép kiểm tra phần cứng để chọn batch thử. Nếu API: chốt quyền gửi nội dung và ngân sách; chưa cần đưa key vào chat.
5. Chốt bắt đầu từ dữ liệu mẫu, chưa ingest cả sách ngay.

Đầu ra: ghi các quyết định vào phần cuối tài liệu này. Điểm dừng: bạn duyệt lựa chọn trước khi cài dependency/tải model hoặc gọi API.

### Bước 1 — Chốt contract và kiểm tra dữ liệu trước embedding

Việc làm:

1. Chuyển mục 4 thành contract ngắn đã thống nhất, có một ví dụ đầu vào/đầu ra.
2. Chuẩn bị manifest chứa ID sách, file Parser, phạm vi trang, policy lọc, cấu hình chunk và profile embedding.
3. Tạo chunk IDs xác định; kiểm tra trùng ID, nội dung rỗng, metadata JSON và khả năng tra record nguồn.
4. Đếm token thực tế bằng tokenizer của model, gồm phần tiền tố/special token liên quan. In min/median/max và danh sách chunk vượt giới hạn.
5. Với chunk quá dài: báo lỗi có ID và số token, không tự truncate. Nếu cần giảm `chunk_size` thì đổi cấu hình được ghi nhận và chạy lại cả các chiến lược dùng để so sánh.
6. Báo cáo chunk không có nguồn, chapter/section thiếu, ký tự lỗi đáng chú ý. Không tự sửa Parser trong bước embedding.

Đầu ra: contract, manifest dự kiến và báo cáo preflight. Tiêu chí đạt: toàn bộ chunk dự kiến encode hợp lệ, nguồn còn truy vết được, không có truncation âm thầm. Điểm dừng: bạn xem schema và báo cáo trước khi chạy model thật.

### Bước 2 — Hiện thực Embedder đơn giản

Thiết kế đề xuất:

```python
class BaseEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Base có thể là abstract class; chỉ thêm một class con cho provider đã chọn. Với local có thể dùng `LocalTextEmbedder`; với API có thể dùng `OpenAITextEmbedder`. Không viết tất cả provider ở lần đầu.

Việc làm:

1. Tải/load model một lần hoặc khởi tạo API client một lần, không làm lại cho từng chunk.
2. Áp dụng đúng encode document/query của model.
3. Chạy theo batch nhỏ; có thể thử 8 rồi 16/32 tùy máy hoặc giới hạn request. Đây là mức thử, không là cam kết batch tối ưu.
4. Giữ thứ tự vector trùng thứ tự input; với API dùng trường index của response để kiểm tra ánh xạ.
5. Kiểm tra số vector, số chiều, giá trị hữu hạn và norm khác 0; không dùng vector rỗng/zero thay cho batch lỗi.
6. API dùng timeout và retry có giới hạn cho lỗi tạm thời, ví dụ rate limit/network; lỗi text/cấu hình không retry vô hạn.
7. Cache các vector đã thành công bằng profile + hash text encode. Ghi cache theo batch, validate cache trước khi dùng lại.
8. In tiến độ số chunk hoàn tất, thời gian và token/cost khi có; không in API key hay toàn bộ vector dài.

Test cần có: input rỗng trả về `[]`; query rỗng báo lỗi; thứ tự không đảo; không sửa text; đúng tiền tố; sai chiều/NaN/zero norm bị từ chối; cache không dùng lẫn profile.

Đầu ra: class có docstring/comment về input/output, test bằng fake model/client và một smoke test model thật trên vài chunk. Test thường không cần mạng hoặc API trả phí; smoke test là bước chạy riêng có lựa chọn rõ ràng. Điểm dừng: bạn xem input encode, chiều vector và kết quả chạy nhỏ.

### Bước 3 — Hiện thực Vector Store cho backend đã chọn

API nhỏ đủ dùng:

```python
store.upsert(items)
store.count()
store.search(query_vector, top_k=5, filters=None)
store.close()
```

Không cần tạo nhiều lớp factory, plugin hoặc backend chưa dùng. Có thể dùng trực tiếp client chính thức bên trong một class wrapper.

Việc làm:

1. Tạo/mở collection với số chiều và distance đã chốt; kiểm tra manifest/profile khớp trước khi ghi hoặc query.
2. Upsert bằng chunk IDs xác định; lưu text, ID sách và đầy đủ metadata.
3. Cung cấp filter chapter/section cần thiết bằng metadata dễ truy vấn. Nếu Chroma, giữ bản JSON đầy đủ song song với các trường filter hợp lệ.
4. Batch ghi dữ liệu; xác nhận thành công rồi mới cập nhật tiến độ lưu.
5. Đếm số point/bản ghi; đọc lại vài ID và đối chiếu text/metadata với đầu vào.
6. Đóng client, mở lại storage và kiểm tra persistence.

Test cần có: upsert hai lần cùng input không tăng count; dữ liệu sách/cấu hình khác không đè nhầm; sai chiều/profile bị báo lỗi; metadata lồng nhau round-trip đúng; search/filter/count hoạt động sau mở lại.

Đầu ra: một wrapper storage và test persistence dùng vector giả, không cần chạy model thật. Điểm dừng: bạn xem một bản ghi đã lưu và bằng chứng mở lại vẫn đọc được.

### Bước 4 — Pipeline nhỏ từ input đến index

Tạo `scripts/run_embedding_pipeline.py`, phong cách tương tự script Chunker hiện có. Trước tiên nhận Parser JSONL mẫu; chưa cần parse lại PDF cho mỗi lần thử.

Thứ tự chạy:

1. Nạp record + thông tin sách.
2. Chạy Chunker baseline với cấu hình đã ghi.
3. Preflight, tạo IDs và manifest trạng thái đang chạy.
4. Encode/cache theo batch rồi upsert vào storage.
5. Kiểm tra count, đọc lại sample, đóng và mở lại để kiểm tra lưu bền vững.
6. Chỉ đánh dấu hoàn tất khi mọi chunk dự kiến đã được lưu đúng.
7. In vài chunk mẫu và tổng kết; vector chỉ in số chiều, norm và vài giá trị đầu nếu cần.

Nội dung báo cáo:

- Input, số chapter/record/chunk, chunk strategy và config.
- Model, provider, chiều vector, normalization, collection/path lưu.
- Token thống kê; cache hit/miss; số encode và số upsert thành công.
- Thời gian, token billed nếu API trả về, lỗi và ID liên quan nếu có.
- Một record trước encode và một bản ghi sau lưu, có nguồn PDF.
- Kiểm tra persistence và kết quả chạy lại cùng cấu hình.

Khi bị ngắt: chạy lại dùng cache đã ghi và upsert cùng ID; không coi lần bị ngắt là hoàn tất. Manifest/cache đủ cho phiên bản tuần tự, chưa cần hệ thống job/checkpoint phức tạp. Nếu cache write/upsert lỗi thì dừng và báo vị trí; không tiếp tục báo thành công.

Đầu ra: script chạy thật, báo cáo JSON và mẫu in dễ nhìn. Tiêu chí đạt: count khớp số chunk dự kiến, mở lại được, chạy lần hai không nhân đôi và dùng lại cache. Điểm dừng: bạn xem pipeline đầu-cuối trước khi thêm retrieval hoặc mở rộng dữ liệu.

### Bước 5 — Retrieval kiểm tra và so sánh ba cách chunk

Đây là kiểm tra tìm đúng nguồn, chưa sinh câu trả lời bằng LLM.

Việc làm:

1. Tạo hàm `search(query, top_k=5, filters=None)` dùng cùng embedding profile với index.
2. In top-k gồm rank, điểm, đoạn text, chapter/section, danh sách trang PDF và chunk ID.
3. Tạo khoảng 15–20 câu hỏi có đáp án nằm trong các chapter đã index: định nghĩa, phân biệt khái niệm, câu hỏi diễn đạt lại; nếu cần thì có cặp Anh/Việt.
4. Gán nguồn đúng bằng record ID và khoảng text/section đã kiểm tra trong Parser output. Không lấy một chunk ID cố định làm gold chung cho cả ba chiến lược, vì cách cắt khác nhau.
5. Với mỗi chiến lược, xác định chunk liên quan khi nó chứa đủ đoạn bằng chứng đã gán; không tính một mảnh overlap quá ngắn là đáp án đủ.
6. Đo Hit@5 (có ít nhất một kết quả liên quan trong top 5), MRR@5 và thời gian query. Chỉ gọi Recall@5 nếu đã định nghĩa đầy đủ tập relevant và cách đếm.
7. Chạy ba chiến lược trên cùng record, cấu hình kích thước, model và bộ câu hỏi. Mỗi chiến lược dùng collection riêng; có thể dùng lại vector cache cho text giống nhau.
8. Báo cáo lỗi retrieval theo nhóm: nguồn chưa nằm trong corpus, parser lỗi, chunk bị cắt mất ý, khác ngôn ngữ hoặc model xếp hạng sai.

Ví dụ câu hỏi chỉ được đưa vào gold sau khi tìm đúng đoạn nguồn: “What is the difference between a database and a DBMS?” hoặc “Lược đồ cơ sở dữ liệu khác với trạng thái cơ sở dữ liệu như thế nào?”. Không dùng câu hỏi về chapter chưa index để chấm model trên pilot Chapter 1–3.

Có thể thử query ngoài nội dung để quan sát, nhưng dense top-k vẫn có thể trả kết quả không liên quan. Chưa có cơ chế bảo đảm “không tìm thấy” hoặc trả lời đúng. Không đặt một ngưỡng điểm tùy ý rồi coi đó là độ tin cậy.

Đầu ra: bộ query + nguồn gold + bảng kết quả và mẫu retrieval. Tiêu chí đạt kỹ thuật: query đúng profile, metadata nguồn đúng, không lẫn chiến lược. Chất lượng: báo số đo thật, bạn duyệt vài kết quả trước khi chốt baseline; chưa hứa một mức phần trăm khi chưa có bộ gold.

### Bước 6 — Mở rộng từ mẫu sang toàn bộ sách

Việc làm theo thứ tự:

1. Parse sách đầy đủ bằng một lượt có context chapter/section liên tục; ghi toàn bộ record Parser, gồm footnote.
2. In báo cáo coverage theo trang/chapter/type, record thiếu chapter, thứ tự chapter bất thường, title mâu thuẫn và trang không có text. Rà mẫu đầu/cuối mỗi chapter, không chỉ kiểm tra file chạy không lỗi.
3. Phân biệt front matter/back matter hợp lệ với record thiếu chapter do Parser nhận diện sai. Không tự gộp tất cả `chapter_number=null` thành “preface”.
4. Bạn chốt policy: loại phần ngoài phạm vi nhưng giữ trong output Parser; bổ sung context đúng; hoặc mở rộng contract cho nhóm ngoài chapter/appendix. Nếu đổi contract phải cập nhật notebook/class/test liên quan trước khi ingest.
5. Chunk và preflight token toàn bộ dữ liệu trong phạm vi đã duyệt. Xác nhận số chunk và báo phần bỏ qua cùng lý do.
6. Nếu API, dự toán token cần encode mới và yêu cầu duyệt mức chi trước khi chạy lớn. Nếu local, chạy một batch đại diện để đo tốc độ rồi ước lượng; không dự đoán từ số trang PDF một cách máy móc.
7. Chạy embedding/index theo batch, dùng cơ chế cache/upsert đã test.
8. Mở lại index, xác nhận count và thử câu hỏi ở đầu/giữa/cuối sách.

Đầu ra: manifest phạm vi, Parser output đầy đủ, index hoàn tất và báo cáo coverage. “Đã ingest toàn sách” chỉ dùng khi đúng phạm vi đã cam kết; nếu loại mục lục/lời nói đầu/phụ lục thì phải nêu rõ. Pipeline này ingest **text mà Parser hỗ trợ**, không đồng nghĩa đã hiểu toàn bộ hình ảnh hoặc tái dựng bảng PDF.

### Bước 7 — Giao diện nhỏ cho ingestion và quan sát

Đề xuất Streamlit nếu chỉ cần giao diện Python local để thử nhanh; đây là lựa chọn của dự án, không bắt buộc. [Tài liệu Streamlit](https://docs.streamlit.io/get-started/fundamentals/main-concepts).

Phạm vi giao diện đầu tiên:

1. Chọn PDF và phạm vi nội dung/policy đã hỗ trợ.
2. Hiện model/storage đã chọn; chọn chiến lược chunk và size/overlap nếu muốn thử cấu hình khác.
3. Nút chạy ingestion, tiến độ theo bước và thông báo lỗi cụ thể.
4. Thống kê record/chunk/vector, phần bị bỏ qua, thời gian và trạng thái lưu.
5. Xem vài record Parser, chunk và metadata sau lưu.
6. Ô nhập query, top-k và nguồn PDF; chưa có chat hoặc sinh đáp án.
7. Chọn lại sách/index hoàn tất ở lần mở tiếp theo.

Giao diện gọi đúng các hàm pipeline đã dùng trong CLI. Với Qdrant local không chạy CLI mở cùng storage song song; dùng client được tái sử dụng trong process của UI. Nếu cần nhiều client/người dùng hoặc background ingestion đồng thời thì chuyển sang server ở một bước riêng.

Đầu ra: giao diện quan sát nhỏ. Tiêu chí đạt: chạy được cùng input/config như CLI, không tạo bản sao ngoài ý muốn và hiển thị đúng nguồn/lỗi. Không đưa API key vào màn hình kết quả hoặc log.

## 6. Cấu trúc file dự kiến

Các file dưới đây là dự kiến, chưa được tạo trong phiên lập kế hoạch:

```text
docs/
  embedding_plan.md                  # Kế hoạch này
  embedding_contract.md              # Contract ngắn sau khi duyệt
src/
  ingestion/
    textbook_embedder.py             # Base + một implementation đã chọn
  storage/
    vector_store.py                 # Một wrapper cho backend đã chọn
  retrieval/
    textbook_retriever.py            # Embed query + tìm top-k
scripts/
  run_embedding_pipeline.py          # Pipeline mẫu, in input/output
tests/
  test_textbook_embedder.py          # Fake model/client; test không trả phí
  test_vector_store.py               # Persistence và upsert bằng vector giả
  test_embedding_pipeline.py         # Test ghép đầu-cuối nhỏ
data/
  interim/                          # Parser output, manifest/input trung gian
  vector_db/                        # Dữ liệu persistent nếu dùng local
output/
  chunks/                           # Output Chunker hiện có
  embeddings/                       # Cache theo profile và manifest/report
  retrieval/                        # Query/gold/result khi đánh giá
```

Chỉ thêm dependency của phương án đã chọn. Bổ sung `.gitignore` cho index/cache/key nếu cần, giữ test fixture nhỏ không chứa toàn sách. Chưa cần framework RAG mới: tiếp tục dùng `Document` hiện có và client chính thức cho model/storage.

Không thay code Parser/Chunker đã đạt chỉ để gắn vector. Nếu mở rộng policy ngoài chapter hoặc cách dựng text, làm thành thay đổi có contract/test và mốc đối chiếu riêng.

## 7. Tiêu chí gọi là hoàn tất

### Mốc I — Embedding và lưu trữ mẫu hoàn tất

- Contract đã được duyệt.
- Mọi chunk trong phạm vi đã chọn có vector đúng profile hoặc được báo lỗi rõ, không silently truncate.
- Nội dung/metadata nguồn round-trip không mất thông tin.
- Index lưu trên đĩa, mở lại được và upsert lặp không nhân đôi.
- Có test tự động không cần API trả phí và smoke test thật tách riêng.
- Pipeline in input/output và manifest rõ ràng; chạy dở không được đánh dấu hoàn tất.

### Mốc II — Ingestion sách sẵn sàng dùng

- Phạm vi toàn sách/policy loại đã chốt và có coverage report.
- Index sách hoàn tất, count khớp, metadata truy vết đúng.
- Query thử ở nhiều phần sách tìm được nguồn được kiểm tra; có baseline retrieval và biết các lỗi còn lại.
- Chi phí/tài nguyên được đo, có thể chạy lại nhờ cache và IDs ổn định.

### Mốc III — Có giao diện quan sát

- UI gọi lại pipeline đã test và hiển thị đúng tiến độ, lỗi, mẫu dữ liệu và retrieval.
- Đây là giao diện ingestion/search, chưa phải hệ RAG trả lời hoàn chỉnh.

## 8. Phiếu quyết định

Bạn có thể trả lời theo mẫu ngắn bên dưới; các đề xuất chưa được coi là đã duyệt.

```text
Model: M1 / M2 / M3 / M4
Storage: V1 local / Qdrant server / V2 / V3 / V4
Chunk pilot: structure-aware trước / cả ba ngay
Query: Anh + Việt / chỉ Anh
Footnote: giữ False / bật True
RAM và GPU nếu chọn local: ...
Nếu API: đồng ý gửi text và giới hạn chi pilot ... USD
Mở rộng toàn sách: xem coverage rồi quyết định phần thiếu chapter
UI: làm sau pipeline / muốn xem sớm
```

Thứ tự làm việc đề xuất để bạn quan sát: **chốt lựa chọn → duyệt contract/preflight → duyệt Embedder → duyệt Storage → xem pipeline mẫu → kiểm tra retrieval/ba chiến lược → duyệt coverage và ingest cả sách → giao diện**.
