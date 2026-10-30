# Generation Materials — lý thuyết và các phương pháp sinh đáp án trong RAG

Generation nhận câu hỏi và bằng chứng để tạo câu trả lời. Trong RAG, chất lượng phụ thuộc cả việc tìm đúng nguồn, đưa nguồn vào context và sử dụng nguồn đúng khi sinh đáp án. Tài liệu trình bày nền tảng, các phương pháp, công thức và đánh đổi, tiếp nối [Embedding Materials](EmbeddingMaterials.md) và [Retrieval Materials](RetrievalMaterials.md).

```text
Câu hỏi → Retrieval → các chunk có nguồn → Context Assembly
       → prompt → model sinh đáp án → kiểm tra cấu trúc và nguồn → kết quả
```

## 1. Có bao nhiêu phương pháp Generation?

**Không có một số lượng cố định.** Cần phân biệt các quyết định ở những cấp khác nhau:

| Trục phân biệt | Các lựa chọn tiêu biểu | Trả lời câu hỏi nào? |
| --- | --- | --- |
| Cách tạo đáp án | Extractive, abstractive/generative | Trích đoạn hay viết lại/tổng hợp? |
| Tổ chức bằng chứng | Ghép context, xử lý từng phần, tổng hợp nhiều bước | Đưa nhiều nguồn vào quá trình trả lời thế nào? |
| Prompting | Zero-shot, few-shot, hướng dẫn theo nguồn | Hướng dẫn model thực hiện nhiệm vụ thế nào? |
| Decoding | Greedy, sampling, beam search | Chọn token tiếp theo thế nào? |
| Định dạng đầu ra | Văn bản tự do, JSON, schema ràng buộc | Phần mềm nhận kết quả dưới dạng nào? |
| Điều phối pipeline | Một lượt, truy xuất và kiểm tra lặp | Cần gọi model/retriever bao nhiêu bước? |
| Thích nghi model | Dùng model có sẵn, fine-tuning, LoRA | Có cập nhật tham số model không? |

Các lựa chọn này có thể kết hợp. Ví dụ: một model instruction-tuned nhận các chunk đã ghép, sinh một lượt bằng greedy decoding và trả JSON có citation. Ollama là runtime chạy model; JSON là định dạng; cả hai không phải tên một phương pháp RAG riêng.

## 2. Nền tảng: model sinh văn bản như thế nào?

### Sinh token theo điều kiện

Gọi $q$ là câu hỏi, $C$ là context bằng chứng, $I$ là hướng dẫn và $y=(y_1,\ldots,y_T)$ là đáp án. Với mô hình sinh tự hồi quy:

$$
p_\theta(y\mid q,C,I)
=\prod_{t=1}^{T}p_\theta(y_t\mid y_{<t},q,C,I).
$$

Ở mỗi bước, model dự đoán phân phối của token tiếp theo dựa trên đầu vào và phần đáp án đã sinh. Decoding chọn một token từ phân phối đó. Xác suất cao nghĩa là phù hợp với phân phối model học được, không đồng nghĩa phát biểu đúng hoặc được tài liệu chứng minh.

Đưa tài liệu vào prompt thường chỉ thay đổi đầu vào lúc inference; nó không tự cập nhật trọng số model. Bài báo [RAG gốc](https://arxiv.org/abs/2005.11401) kết hợp bộ nhớ trong tham số với tài liệu truy xuất bên ngoài; pipeline dùng prompt hiện nay không nhất thiết hiện thực đúng kiến trúc huấn luyện của bài báo đó.

### Model, tokenizer, chat template và runtime

| Thành phần | Vai trò | Ví dụ trong dự án |
| --- | --- | --- |
| Model | Tính phân phối token và sinh văn bản | Qwen chạy qua Ollama |
| Tokenizer | Chuyển văn bản thành token và ngược lại | Tokenizer tương thích với Qwen |
| Chat template | Biểu diễn vai trò system/user/assistant bằng định dạng model sử dụng | `apply_chat_template()` |
| Runtime | Nạp model, thực thi inference và phục vụ yêu cầu | Ollama |
| Code ứng dụng | Chuẩn bị context, gọi runtime, kiểm tra kết quả | `textbook_generation.py` |

Danh sách message vẫn phải được chuyển thành chuỗi token. Template khác nhau có thể dùng control token khác nhau; cùng nội dung nhưng sai template có thể ảnh hưởng đầu ra. Xem [Hugging Face — Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating).

## 3. Extractive và abstractive answering

### Extractive — chọn câu trả lời từ nguồn

Hệ thống chọn một đoạn, câu hoặc span đã có trong tài liệu. Ví dụ nguồn ghi “A database is a collection of related data”, hệ thống trả lại đúng đoạn định nghĩa đó.

**Ưu điểm:** dễ đối chiếu nguyên văn và vị trí nguồn. **Hạn chế:** chọn sai span vẫn trả lời sai; khó ghép nhiều nguồn hoặc diễn đạt bằng ngôn ngữ khác. Trích nguyên văn không bảo đảm đoạn đó trả lời đúng câu hỏi.

### Abstractive/generative — diễn đạt và tổng hợp

Model viết đáp án dựa trên một hoặc nhiều đoạn, có thể tóm tắt, giải thích hay trả lời bằng tiếng Việt từ tài liệu tiếng Anh.

**Ưu điểm:** linh hoạt với diễn đạt, ngôn ngữ và câu hỏi tổng hợp. **Hạn chế:** có thể thêm chi tiết không có nguồn, lẫn khái niệm hoặc bỏ điều kiện quan trọng.

Hai cách có thể kết hợp: trích các câu bằng chứng trước rồi yêu cầu model tổng hợp. Đoạn trung gian vẫn phải truy vết về nguồn gốc; lỗi chọn bằng chứng có thể truyền xuống đáp án.

## 4. Context Assembly — tổ chức bằng chứng

### Context khác prompt

**Context bằng chứng** là các đoạn nguồn được cung cấp. **Prompt** gồm hướng dẫn, câu hỏi, các đoạn nguồn và định dạng hội thoại. **Context window** là sức chứa token của model trong một yêu cầu; từ “context” ở đây không chỉ riêng các đoạn tài liệu.

Một nguồn có thể được trình bày như sau:

```text
[S1] PDF page: 35
A database is a collection of related data.

[S2] PDF page: 36
A DBMS enables users to create and maintain a database.
```

Nhãn `S1` giúp model trích dẫn; phần mềm giữ ánh xạ từ nhãn về chunk, trang và metadata. Nhãn không thay thế bằng chứng và không tự chứng minh mức liên quan.

### Token budget

Với model decoder-only đang dùng, ngân sách được kiểm tra trên prompt đã áp dụng chat template:

$$
N_{prompt}+B_{output}\le W,
$$

trong đó $W$ là cửa sổ cấu hình và $B_{output}$ là số token tối đa dành cho đáp án. Với $W=4096$ và $B_{output}=256$, toàn bộ prompt được dùng tối đa 3840 token, gồm cả hướng dẫn và câu hỏi.

Đếm bằng tokenizer tương thích; số ký tự hoặc số từ chỉ là ước lượng. Chat template và token đặc biệt cũng chiếm chỗ. Cấu hình cửa sổ dùng cho một lượt chạy có thể thấp hơn khả năng tối đa của model.

### Các cách chọn và rút gọn context

| Cách làm | Cơ chế | Đánh đổi |
| --- | --- | --- |
| Chọn theo hạng | Lần lượt đưa các chunk phù hợp vào budget | Đơn giản; nhiều chunk gần giống nhau có thể chiếm chỗ |
| Loại trùng | Bỏ nội dung/nguồn lặp theo tiêu chí xác định | Cần phân biệt trùng chính xác với hai đoạn chỉ gần giống |
| Mở rộng đoạn lân cận | Thêm phần trước/sau chunk đã tìm | Giữ định nghĩa và điều kiện; tăng số token |
| Trích câu liên quan | Chọn câu thay vì giữ nguyên chunk | Gọn hơn; có thể mất ngữ cảnh hoặc điều kiện |
| Tóm tắt/nén | Model hoặc bộ lọc rút gọn nội dung | Giảm token; có nguy cơ làm sai hoặc bỏ bằng chứng |

Thêm nguồn chưa chắc giúp hơn. Nghiên cứu [Lost in the Middle](https://arxiv.org/abs/2307.03172) cho thấy ở các model và nhiệm vụ được khảo sát, vị trí thông tin trong context dài ảnh hưởng kết quả. Đây là lý do cần thử số lượng, độ dài và thứ tự nguồn trên dữ liệu mục tiêu, không phải quy luật mọi model đều kém ở giữa.

**Khi dùng:** chọn theo hạng và giữ nguyên chunk là baseline dễ kiểm tra. Mở rộng/nén context là biến thực nghiệm riêng; cần đo cả phần bằng chứng còn lại sau xử lý.

## 5. Prompting — hướng dẫn model trả lời theo nguồn

### Zero-shot và few-shot

- **Zero-shot:** mô tả nhiệm vụ và quy tắc, không cung cấp ví dụ hoàn chỉnh.
- **Few-shot:** thêm vài ví dụ đầu vào–đầu ra để minh họa cách trả lời, dẫn nguồn và báo thiếu bằng chứng.

Few-shot có thể làm rõ hành vi mong muốn nhưng dùng thêm token và chịu ảnh hưởng của việc chọn ví dụ. Ví dụ nên lấy từ dev; dùng câu eval làm mẫu sẽ làm mất tính độc lập của đánh giá.

### Hướng dẫn cho grounded answering

Một prompt có thể yêu cầu: chỉ sử dụng nguồn được cung cấp; trả lời đúng ngôn ngữ; nêu đáp án thay vì lặp câu hỏi; dẫn nguồn cho phát biểu; báo thiếu bằng chứng khi không đủ thông tin.

Ví dụ:

```text
Trả lời câu hỏi bằng tiếng Việt, chỉ dựa vào các nguồn bên dưới.
Nếu nguồn không đủ để kết luận, báo thiếu bằng chứng.
Dẫn nhãn nguồn cho các phát biểu và không bổ sung chi tiết ngoài nguồn.
```

Đây là yêu cầu về hành vi, không phải phép kiểm chứng ngữ nghĩa. Nội dung nguồn cần được phân định là dữ liệu để đọc; một câu mang tính chỉ dẫn nằm trong tài liệu không nên được coi là lệnh điều khiển ứng dụng.

### Trả lời trực tiếp và xử lý nhiều bước

Một lượt trả lời trực tiếp phù hợp với định nghĩa hoặc câu hỏi có nguồn ngắn. Với câu hỏi tổng hợp, có thể tách thành câu hỏi con, tìm bằng chứng cho từng phần rồi tổng hợp đáp án. Mỗi bước thêm chi phí và cơ hội sai; nhiều bước không mặc định chính xác hơn.

Chế độ “thinking” của một model, việc chia tác vụ thành nhiều bước và khả năng trả lời đúng là ba điều khác nhau. Đánh giá bằng đáp án cùng bằng chứng, không bằng độ dài phần giải thích.

## 6. Decoding — chọn token tiếp theo

### Temperature

Gọi $z_i$ là logit của token $i$. Với nhiệt độ $\tau>0$:

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

Nhiệt độ thấp làm phân phối tập trung hơn; nhiệt độ cao làm phẳng hơn. Công thức không áp dụng bằng cách thay trực tiếp $\tau=0$; runtime thường xử lý giá trị 0 theo nhánh riêng. Nhiệt độ thấp không phải bộ kiểm tra tính đúng.

### Greedy, sampling và beam search

| Phương pháp | Cách chọn | Ưu điểm | Hạn chế |
| --- | --- | --- | --- |
| Greedy | Chọn token có xác suất cao nhất ở mỗi bước | Đơn giản, không lấy mẫu ngẫu nhiên | Quyết định cục bộ; có thể lặp hoặc đi sai hướng |
| Sampling | Lấy mẫu từ phân phối token | Tạo các cách diễn đạt khác nhau | Kết quả biến động, có thể chọn token kém phù hợp |
| Top-k sampling | Chỉ lấy mẫu trong k token cao nhất | Giới hạn tập ứng viên | k cố định không thích nghi độ tập trung của phân phối |
| Top-p / nucleus | Chọn tập nhỏ nhất có tổng xác suất ít nhất p rồi lấy mẫu | Kích thước tập thay đổi theo phân phối | Cần kiểm soát cùng temperature và các bộ lọc khác |
| Beam search | Giữ nhiều chuỗi ứng viên theo điểm tích lũy | Xem xét nhiều hướng hơn greedy | Tốn tài nguyên; điểm chuỗi cao chưa bảo đảm factuality |

Xem [Hugging Face — Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) và bài báo đề xuất [Nucleus sampling](https://arxiv.org/abs/1904.09751). Kết quả về tính đa dạng của văn bản trong bài báo không trực tiếp chứng minh chất lượng trả lời theo nguồn.

Ví dụ phân phối `[0.50, 0.30, 0.15, 0.05]`: với `top_p=0.8`, tập nucleus gồm hai token đầu; xác suất trong tập được chuẩn hóa lại trước khi lấy mẫu.

**Lưu ý tên tham số:** top-k chunk của retrieval khác top-k token của decoding. `seed` giúp kiểm soát ngẫu nhiên khi có lấy mẫu, nhưng không bảo đảm kết quả giống tuyệt đối qua model, runtime, phần cứng và phiên bản khác nhau.

## 7. Tổng hợp nhiều nguồn và context dài

### Ghép nguồn trong một lần gọi

Đưa các chunk vào một prompt và sinh một đáp án. Cách này thường được gọi là concatenation hoặc “stuff” trong thư viện ứng dụng.

**Ưu điểm:** ít bước, thuận tiện theo dõi nguồn. **Hạn chế:** bị giới hạn budget; nguồn dư có thể gây nhiễu. Phù hợp làm baseline khi bằng chứng vừa cửa sổ.

### Xử lý từng phần rồi tổng hợp

| Cách tổ chức | Luồng xử lý | Hạn chế chính |
| --- | --- | --- |
| Map–reduce | Rút thông tin từng phần → tổng hợp các kết quả trung gian | Bước đầu có thể bỏ chi tiết chỉ có ý nghĩa khi ghép nguồn |
| Refine | Tạo đáp án từ phần đầu → cập nhật qua từng phần tiếp theo | Phụ thuộc thứ tự, khó song song và có thể giữ lỗi ban đầu |
| Tree/hierarchical summary | Tổng hợp theo nhiều tầng đến khi vừa đầu vào cuối | Nhiều lượt gọi; mất thông tin qua các tầng |

Đây là các kiểu tổ chức response synthesis; tên và cách đóng gói cụ thể khác nhau giữa thư viện. [LlamaIndex — Response modes](https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/response_modes/) mô tả refine và tree summarization. Dùng kiểu tổng hợp không đồng nghĩa phải thêm thư viện đó vào dự án.

### Fusion-in-Decoder và các mô hình RAG gốc

**Fusion-in-Decoder (FiD)** encode các cặp câu hỏi–đoạn riêng rồi để decoder kết hợp các biểu diễn khi sinh đáp án. Nó là thiết kế model, khác việc nối các đoạn thành một chuỗi prompt. Xem [FiD — Leveraging Passage Retrieval](https://arxiv.org/abs/2007.01282).

**RAG-Sequence** và **RAG-Token** trong [bài báo RAG](https://arxiv.org/abs/2005.11401) khác nhau ở cách lấy tổng theo tài liệu tiềm ẩn cho cả chuỗi hay ở mức token. Không gọi một prompt chứa nhiều nguồn là RAG-Token chỉ vì đáp án được sinh từng token.

## 8. Structured output — đầu ra có cấu trúc

### Văn bản tự do, JSON và JSON Schema

Văn bản tự do thuận tiện để đọc nhưng khó trích trường ổn định. Yêu cầu JSON giúp tách dữ liệu; JSON Schema còn mô tả trường bắt buộc, kiểu và giá trị được phép. Ollama nhận schema qua trường `format`; xem [Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs).

```json
{
  "status": "answered",
  "answer": "Lược đồ mô tả cấu trúc; trạng thái là dữ liệu tại một thời điểm.",
  "citations": ["S1", "S2"]
}
```

Đây là ví dụ định dạng; nội dung chỉ được chấp nhận về ngữ nghĩa sau khi đối chiếu nguồn thực tế.

### Hai nhánh trạng thái

| Nhánh | Quy tắc ví dụ |
| --- | --- |
| `answered` | Có đáp án và ít nhất một citation thuộc danh sách nguồn |
| `insufficient_evidence` | Có lời giải thích và citation rỗng |

`const` cố định trạng thái trong từng nhánh; `enum` giới hạn nhãn nguồn; `minItems`/`maxItems` giới hạn số phần tử. `oneOf` yêu cầu đúng một nhánh hợp lệ, theo [JSON Schema — Combining schemas](https://json-schema.org/understanding-json-schema/reference/combining).

Schema hỗ trợ ràng buộc hình thức. Ứng dụng vẫn kiểm tra kết quả sau gọi model: JSON đọc được, trường đúng, citation hợp lệ, không trùng nếu policy yêu cầu và output không bị cắt. `minLength: 1` chỉ đếm ký tự, nên chưa đủ để loại chuỗi toàn khoảng trắng.

**Phân biệt:** model trả JSON đúng không chứng minh đáp án đúng; giới hạn citation vào `S1/S2` không chứng minh hai nguồn này hỗ trợ phát biểu. Schema cũng không tự phát hiện model lặp lại câu hỏi.

## 9. Citation, bám nguồn và thiếu bằng chứng

### Các mức kiểm tra citation

1. **ID tồn tại:** nhãn thuộc các nguồn đã đưa vào prompt.
2. **Truy vết được:** ánh xạ đúng tới chunk, trang và record gốc.
3. **Có hỗ trợ:** nội dung được dẫn thực sự chứng minh phát biểu.
4. **Đủ độ phủ:** các phát biểu cần nguồn đều có bằng chứng phù hợp.

Ví dụ model dẫn đúng `S1`, nhưng `S1` chỉ nói về schema trong khi đáp án khẳng định thuộc tính của transaction: ID hợp lệ, bằng chứng không phù hợp. [ALCE](https://arxiv.org/abs/2305.14627) nghiên cứu đánh giá đáp án có citation theo các chiều correctness và chất lượng dẫn nguồn, ngoài tính lưu loát.

Citation ở mức toàn đáp án dễ triển khai nhưng khó biết nguồn hỗ trợ câu nào. Citation theo câu/phát biểu rõ hơn; đổi sang cách đó cần điều chỉnh cả định dạng lẫn cách chấm.

### Báo thiếu bằng chứng

**Thiếu bằng chứng trong context** khác **không có đáp án trong toàn corpus**. Retriever có thể bỏ sót nguồn, hoặc context assembly có thể loại đoạn cần thiết vì budget.

| Trường hợp | Hành vi cần đánh giá |
| --- | --- |
| Context rỗng | Có thể trả thiếu bằng chứng ngay trong code |
| Context có nội dung nhưng không liên quan | Model cần tránh suy diễn đáp án |
| Có nguồn cho một phần câu hỏi | Trả phần được hỗ trợ và nêu giới hạn nếu policy cho phép, hoặc báo chưa đủ |
| Context chứa nguồn mâu thuẫn | Nêu mâu thuẫn và nguồn, tránh tự chọn một kết luận thiếu cơ sở |
| Có đủ bằng chứng | Cần trả lời thay vì từ chối sai |

Policy “trả lời một phần” phải được mô tả trong prompt, schema và rubric; không mặc định đã được hỗ trợ bởi schema hai trạng thái hiện tại.

Nếu code bỏ qua model khi context rỗng, ca đó kiểm tra nhánh chương trình. Muốn đánh giá khả năng từ chối của model, cần thêm ca context không rỗng nhưng thiếu bằng chứng. Điểm similarity/reranker không phải xác suất đáp án có thể trả lời đúng.

## 10. Truy xuất và kiểm tra lặp

Một pipeline có thể tạo bản nháp, kiểm tra bằng chứng, truy xuất bổ sung rồi sửa đáp án. **Ưu điểm:** có cơ hội bổ sung phần còn thiếu. **Hạn chế:** thêm độ trễ và lỗi từ bộ kiểm tra; model tự đồng ý với đáp án của mình chưa phải xác minh độc lập.

Hai hướng nghiên cứu tiêu biểu:

- **Self-RAG:** huấn luyện model dùng reflection tokens để điều khiển truy xuất và tự đánh giá. Chỉ thêm câu “hãy kiểm tra lại” vào prompt không tương đương phương pháp gốc. [Self-RAG paper](https://arxiv.org/abs/2310.11511)
- **Corrective RAG (CRAG):** đánh giá chất lượng kết quả retrieval rồi chọn hành động sửa, gồm khai thác nguồn bổ sung và lọc thông tin. Bài báo có dùng web search; điều đó thay đổi phạm vi nguồn so với hỏi–đáp chỉ trên sách. [CRAG paper](https://arxiv.org/abs/2401.15884)

**Khi dùng:** chỉ thêm vòng lặp khi đã xác định lỗi cần xử lý và đo được lợi ích so với một lượt. Cần giới hạn số lượt và giữ nguồn của thông tin mới.

## 11. Model, fine-tuning và tối ưu inference

### Dùng model có sẵn và fine-tuning

Prompting/RAG đưa thông tin vào lúc hỏi; fine-tuning cập nhật tham số từ dữ liệu huấn luyện. Fine-tuning có thể học cách trả lời, thuật ngữ hoặc định dạng, nhưng không thay thế việc kiểm tra nguồn và cập nhật corpus.

**LoRA** học các ma trận cập nhật hạng thấp trong khi giữ trọng số gốc cố định, giảm số tham số cần huấn luyện. Nó là kỹ thuật thích nghi model, không phải cách chọn chunk. Xem [Hugging Face PEFT — LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).

### Quantization, KV cache và triển khai

- **Quantization:** giảm độ chính xác biểu diễn số để giảm dung lượng/bộ nhớ; có thể ảnh hưởng chất lượng. Tốc độ còn phụ thuộc phần cứng và kernel. [Hugging Face — Quantization](https://huggingface.co/docs/transformers/main/en/quantization/overview)
- **KV cache:** giữ key/value attention của token đã xử lý để tái sử dụng lúc sinh tiếp; khác cache embedding hay cache đáp án. Context dài vẫn tăng nhu cầu bộ nhớ.
- **Local/API:** là lựa chọn triển khai với đánh đổi tài nguyên máy, truyền dữ liệu, độ trễ và chi phí. Không quyết định riêng phương pháp sinh đáp án.
- **Streaming:** trả output từng phần để người dùng đọc sớm; không tự giảm số token cần sinh hoặc bảo đảm tổng thời gian ngắn hơn.

Với `/api/generate`, các trường như `stream`, `raw`, `keep_alive` và thống kê token/thời gian thuộc giao diện runtime. Nếu đã áp dụng chat template phía ứng dụng, cần cấu hình phù hợp để tránh áp dụng template lần nữa. Xem [Ollama — Generate API](https://docs.ollama.com/api/generate).

**Khi chọn cấu hình:** đo trên cùng câu hỏi, corpus và policy chấm; ghi model/revision hoặc digest, quantization, prompt, tokenizer, cửa sổ, decoding, runtime và phần cứng. Số tham số hoặc số bit không đủ để kết luận chất lượng Anh–Việt.

## 12. Đánh giá Generation đúng cách

### Tách các chiều chất lượng

| Chiều đánh giá | Câu hỏi cần chấm |
| --- | --- |
| Correctness | Có trả lời đúng câu hỏi và đúng kiến thức không? |
| Completeness | Có đủ các ý/điều kiện cần thiết không? |
| Faithfulness / groundedness | Các phát biểu có được context hỗ trợ không? |
| Citation quality | Nguồn dẫn có hỗ trợ và bao phủ phát biểu không? |
| Relevance | Có đi thẳng vào câu hỏi không? |
| Abstention | Thiếu nguồn thì báo thiếu; có nguồn thì tránh từ chối sai? |
| Format/runtime | Định dạng hợp lệ, không lỗi kết nối hoặc bị cắt? |

Một đáp án có thể đúng theo kiến thức bên ngoài nhưng không được context hỗ trợ. Ngược lại, đáp án có thể bám sát một nguồn sai; vì vậy correctness và faithfulness là hai chiều khác nhau.

### Các cách chấm

**Chấm thủ công:** dùng rubric rõ ràng, xem câu hỏi–đáp án–nguồn cùng nhau; ghi đúng/sai, đủ/thiếu và nguyên nhân. Điểm chưa chấm phải để trống, không quy thành đạt. Citation không áp dụng cần được phân biệt với citation sai.

**So với reference:** Exact Match phù hợp một số đáp án ngắn; token F1 hoặc các thước đo trùng từ đo mức giống bề mặt. Chúng có thể phạt cách diễn đạt đúng và bỏ sót lỗi ngữ nghĩa, nên không đủ cho giải thích tự do hoặc trả lời xuyên ngôn ngữ.

**Model làm người chấm:** có thể hỗ trợ tách phát biểu, chấm bám nguồn và mức liên quan. Cần đối chiếu một phần với người chấm và lưu model/prompt đánh giá. [Ragas](https://arxiv.org/abs/2309.15217) là một hướng đánh giá nhiều thành phần của RAG; dùng framework không tự tạo ra nhãn chuẩn tuyệt đối.

Với $A$ là số câu được gán nhãn đủ bằng chứng và $U$ là số câu được gán nhãn thiếu bằng chứng, có thể báo riêng:

$$
\text{Tỷ lệ từ chối sai}=\frac{\#\text{câu đủ bằng chứng nhưng báo thiếu}}{A},\qquad
\text{Tỷ lệ cố trả lời khi thiếu nguồn}=\frac{\#\text{câu thiếu bằng chứng nhưng trả lời}}{U}.
$$

Chỉ tính khi mẫu số khác 0. Nhãn phải gắn với context thực tế và policy, không chỉ với bản thân câu hỏi. Hai tỷ lệ trạng thái này vẫn cần đi cùng kiểm tra nội dung đáp án.

### Thiết kế thực nghiệm

1. Dùng dev để sửa prompt/chọn cấu hình; chốt trước khi đánh giá eval. Các bản dịch của cùng câu ở cùng split.
2. So context chọn theo nguồn chuẩn với context retrieval trên cùng câu hỏi, model và prompt. Khác độ dài và nội dung là yếu tố cần ghi nhận khi giải thích kết quả.
3. Thêm câu tổng hợp, thiếu nguồn, ngoài corpus và có đủ nguồn. Context rỗng là phép thử riêng, không đại diện tất cả ca thiếu bằng chứng.
4. Thay từng biến như top-k, thứ tự nguồn hoặc budget; ghi số token và bằng chứng thực sự được đưa vào prompt.
5. Lưu kết quả từng ca, nguồn, output thô, lỗi và review. Không suy ra chất lượng chỉ từ một số trung bình.

### Đo thời gian

Với luồng tuần tự, thống nhất ranh giới đo:

$$
t_{total}=t_{retrieval}+t_{context}+t_{generation}+t_{validation}+t_{other}.
$$

Nếu hàm `generate()` đã đo cả chuẩn bị context và kiểm tra kết quả, không cộng các phần đó lần nữa. Tách thời gian nạp model với lượt đã làm nóng; dùng nhiều lượt khi cần kết luận độ trễ ổn định. Thời gian từ retrieval đã lưu không phải latency end-to-end.

## 13. Liên hệ với dự án hiện tại

| Thành phần | Vai trò trong thực nghiệm |
| --- | --- |
| [`build_context()`](../../src/generation/textbook_generation.py) | Chọn chunk nguyên vẹn theo budget, loại trùng chính xác, gắn nhãn và giữ nguồn |
| `SYSTEM_PROMPT` | Quy định trả lời theo bằng chứng, ngôn ngữ và định dạng |
| `OllamaGenerator` | Gọi model local, nhận output và kiểm tra cấu trúc/citation ID |
| [`notebook 07`](../../notebooks/07_rag_generation.ipynb) | Thực nghiệm context chọn sẵn và retrieval thật, cùng quy trình chấm dev/eval |

Cách tiếp cận đang dùng là **ghép nguồn vào một prompt, sinh đáp án một lượt và trả JSON có citation**, với Qwen qua Ollama. Nó chưa tương đương FiD, Self-RAG, CRAG hoặc fine-tuning.

Các kiểm tra hiện có giúp nhận biết lỗi định dạng, nguồn không hợp lệ và một số lỗi runtime. Chúng chưa tự chấm đáp án đúng, đầy đủ hoặc được nguồn hỗ trợ. Cần xem đánh giá trên pilot và độ phủ toàn sách là hai mốc riêng.

## 14. Những phân biệt cần nhớ

1. **Retriever tìm nguồn; generator sử dụng nguồn:** tìm đúng chưa bảo đảm trả lời đúng.
2. **Model khác runtime:** Qwen là model; Ollama phục vụ thực thi model.
3. **Context khác prompt và context window:** bằng chứng, toàn bộ đầu vào và sức chứa token là ba khái niệm.
4. **Có nhiều token không đồng nghĩa có nhiều bằng chứng:** nội dung lặp hoặc không liên quan vẫn dùng budget.
5. **Greedy/temperature thấp không bảo đảm sự thật:** decoding chọn token, không xác minh phát biểu.
6. **JSON hợp lệ khác đáp án hợp lệ về kiến thức:** schema kiểm tra cấu trúc.
7. **Citation ID đúng khác citation có hỗ trợ:** phải đọc nguồn tương ứng.
8. **Thiếu bằng chứng phụ thuộc context:** câu hỏi có đáp án trong sách vẫn có thể thiếu nguồn trong prompt.
9. **RAG khác fine-tuning:** thêm tài liệu lúc inference khác cập nhật tham số; hai cách có thể kết hợp.
10. **Một lượt chạy không đủ kết luận toàn hệ thống:** cần bộ câu hỏi, rubric, split và ranh giới đo rõ ràng.
