# Contract Parser → Chunker

**Trạng thái: đã chốt; đã triển khai trong Chunker.** Các quy tắc dưới đây áp dụng cho code hiện tại.

Tiến độ: đã chuẩn hóa `BaseChunker`, hoàn thiện ba chiến lược xử lý từng chapter và metadata nguồn cho từng chunk. Script `scripts/run_chunking_pipeline.py` chạy thử từ JSONL của Parser đến output Chunker, kiểm tra nhanh và in mẫu kết quả.

Tài liệu mô tả một contract chung cho `StrictFixedSizeChunker`, `GlobalRecursiveChunker` và `StructureAwareRecursiveChunker`. Mục tiêu là nhận đúng output của Parser, giữ ý nghĩa ba chiến lược trong Experiment 1 và truy vết được chunk về record nguồn. Code đang được triển khai theo từng bước để rà soát các thay đổi.

Schema và cách Parser xử lý footnote được mô tả thêm tại [footnotes.md](footnotes.md).

## 1. Phạm vi và API

API chung:

```python
chunks = chunker.chunk(records)
```

- `records`: danh sách dictionary của **một tài liệu PDF**, do `TextbookParser.parse_pages(...)` trả về hoặc được đọc từ file JSONL do Parser xuất.
- `chunks`: `list[langchain_core.documents.Document]`.
- Một lần gọi có thể nhận một hoặc nhiều chapter; Chunker xử lý từng chapter riêng và trả về một danh sách chunk.
- Thứ tự đầu vào là thứ tự Parser trả về. Chunker không sửa record gốc.
- Đọc file, chọn khoảng trang và lưu chunk là trách nhiệm của caller. `chunk(...)` nhận dữ liệu đã nạp vào bộ nhớ.
- Bước này xử lý text và metadata. Embedding, retrieval, OCR và tái dựng cấu trúc bảng thuộc các bước khác.

File mẫu hiện có là `output/parsed_chapter1.jsonl`. Tên file không quyết định chapter: tại thời điểm kiểm tra, file chứa 777 record thuộc Chapter 1, 2 và 3.

## 2. Schema đầu vào

Mỗi record giữ schema do Parser sinh ra. Các trường chapter/section nằm ở **top-level**, không nằm trong `metadata`.

| Trường | Kiểu dữ liệu | Cho phép `null` | Vai trò với Chunker |
| --- | --- | --- | --- |
| `id` | `str` | Không | Truy vết record nguồn; duy nhất trong danh sách đầu vào. |
| `pdf_page` | `int` | Không | Trang PDF, đánh số từ 0. |
| `chapter_number` | `str` | Có trong schema Parser | Tách phạm vi chapter; quy tắc khi thiếu ở mục 4. |
| `chapter_title` | `str` | Có | Metadata chapter. |
| `section_number` | `str` | Có | Xác định section hoặc đơn vị chưa có section. |
| `section_title` | `str` | Có | Metadata section và heading khi dựng văn bản. |
| `type` | `str` | Không | Lọc footnote và giữ loại content nguồn. |
| `text` | `str` | Không | Nội dung sạch dùng để chunk. |
| `raw_text` | `str` | Không | Văn bản gốc để tra cứu các offset footnote. |
| `bbox` | Dãy 4 số | Không | `[x0, y0, x1, y1]`, tọa độ record theo PDF point. |
| `metadata` | `dict` | Không | Giữ thông tin nguồn, bao gồm liên kết footnote. |

`bbox` có thể là tuple khi nhận trực tiếp từ Parser, hoặc list khi đọc lại JSONL. Các loại `type` hiện có của Parser là `normal_text`, `table_like`, `figure_caption`, `table_caption` và `footnote`.

Ví dụ record nội dung:

```json
{
  "id": "example-document:p34:b3",
  "pdf_page": 34,
  "chapter_number": "1",
  "chapter_title": "Databases and Database Users",
  "section_number": "1.1",
  "section_title": "Introduction",
  "type": "normal_text",
  "text": "A database stores related data.",
  "raw_text": "A database stores related data.\n",
  "bbox": [117.0, 373.8, 467.4, 444.3],
  "metadata": {"footnote_refs": []}
}
```

Kiểm tra đầy đủ sự hiện diện và kiểu dữ liệu của các trường trong bảng. Thiếu trường hoặc sai kiểu phải báo lỗi kèm vị trí/ID record. Trường bổ sung ngoài schema được chấp nhận và không ảnh hưởng thuật toán.

## 3. Cấu hình chung

| Tham số | Mặc định | Quy tắc |
| --- | --- | --- |
| `chunk_size` | `1000` | Số nguyên dương, tính bằng ký tự Python (`len(text)`). |
| `chunk_overlap` | `150` | Số nguyên thỏa `0 <= chunk_overlap < chunk_size`. |
| `include_footnotes` | `False` | Giữ hoặc loại record `type="footnote"`. |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Dùng cho hai chiến lược recursive, theo thứ tự ưu tiên. |

Đơn vị là **ký tự**, không phải token hoặc byte. Với recursive, separator cuối `""` là fallback cắt theo ký tự để bảo đảm giới hạn kích thước; cấu hình tùy chỉnh phải giữ fallback này.

Đầu vào rỗng hoặc không còn nội dung sau lọc trả về `[]`.

## 4. Quy tắc chuẩn bị dữ liệu

### 4.1. Lọc record và giữ văn bản nguồn

- Bỏ record có `text` rỗng hoặc chỉ chứa whitespace.
- Khi `include_footnotes=False`, bỏ record `type="footnote"`.
- Giữ các loại content còn lại, kể cả caption và `table_like`.
- Dùng `text` của Parser, không chạy lại bước làm sạch `raw_text`.
- Khi dựng corpus, giữ nguyên nội dung `text` để các khoảng nguồn vẫn ánh xạ về đúng ký tự trong record. Kiểm tra text rỗng có thể dùng `.strip()`; không dùng kết quả đó để thay text nguồn.

Khi bật `include_footnotes=True`, note được đưa vào theo vị trí trong danh sách Parser. Parser hiện đặt footnote sau phần nội dung của mỗi trang. Contract này không tự chèn note cạnh câu tham chiếu.

### 4.2. Chapter

- Mỗi chapter là một phạm vi chunking độc lập cho cả ba chiến lược.
- Không có chunk hoặc overlap đi qua ranh giới chapter.
- Chapter và record giữ thứ tự xuất hiện trong đầu vào. Record của một chapter phải tạo thành một đoạn liên tiếp; nếu chapter đã kết thúc rồi xuất hiện lại, báo lỗi thay vì tự đổi thứ tự record.
- Nếu record còn lại sau lọc có `chapter_number=null` hoặc rỗng, báo `ValueError` kèm ID. Schema Parser cho phép giá trị này, nhưng Chunker cần chapter để xác định phạm vi xử lý. Caller cần bổ sung context hoặc chọn lại dữ liệu.
- `chapter_title=null` được giữ nguyên nếu toàn chapter không có title. Nếu có một title hợp lệ thì dùng title đó; nhiều title khác nhau cho cùng chapter phải báo lỗi.

Quy tắc thiếu chapter chỉ áp dụng cho record sẽ được chunk. Một footnote bị loại mặc định không làm thất bại quá trình chỉ vì nó thiếu chapter.

### 4.3. Section và giá trị `null`

- Record có `section_number` được gán vào section tương ứng trong chapter.
- Các record **liên tiếp** cùng section tạo một đơn vị văn bản. Nếu section xuất hiện lại sau section khác, tạo một đơn vị mới để giữ thứ tự đọc. Điều này cũng xử lý trường hợp note cuối trang quay về section trước đó.
- Mỗi đơn vị có `section_unit_index`, đánh số từ 0 trong chapter, để phân biệt các lần xuất hiện của cùng section và xác định phạm vi offset.
- Record có `section_number=null` được gán nhãn kỹ thuật `<chapter_number>.intro`, theo quy ước của Experiment 1. Ví dụ: `1.intro`, `section_id="sec_1.intro"`.
- `intro` là nhãn tổng hợp cho nội dung chưa có section. Nó không chứng minh rằng Parser đã bắt đầu đọc từ đầu chapter, đặc biệt khi parse một khoảng trang ở giữa sách.
- Nếu title của đơn vị chưa có section bị thiếu, dùng `"Chapter Introduction"`. Nếu section đã có số nhưng thiếu title, dùng `"Untitled Section"`.
- Cùng một đơn vị có các title hợp lệ mâu thuẫn thì báo lỗi. Giá trị chapter/section gốc được giữ trong thông tin record nguồn.

`parent_section_id` được suy ra:

| Đơn vị | `section_id` | `parent_section_id` |
| --- | --- | --- |
| `1.intro` | `sec_1.intro` | `chapter_1` |
| `1.3` | `sec_1.3` | `chapter_1` |
| `1.3.1` | `sec_1.3.1` | `sec_1.3` |

### 4.4. Dựng văn bản để khớp thực nghiệm

Giữ đúng cách dựng văn bản của Experiment 1 cho cả ba chiến lược:

```text
[[SECTION 1.1: Introduction]]
<text của record thứ nhất>
<text của record thứ hai>
```

- Heading xuất hiện một lần ở đầu mỗi đơn vị section, được tính vào `chunk_size` và offset.
- Các record trong đơn vị nối bằng `"\n"`.
- Các đơn vị trong chapter nối bằng `"\n\n"` khi dựng corpus cho fixed-size và global recursive.
- Heading và joiner là text do Chunker thêm, không phải text của record nguồn.

Giữ heading này là lựa chọn để class tái hiện chính xác thực nghiệm. Nếu sau khi rà soát chọn cách dựng văn bản khác, notebook và class cần cùng chuyển sang quy tắc mới trước khi so sánh.

## 5. Hành vi của ba chiến lược

| Class | Phạm vi một lần split | Boundary bắt buộc | Giá trị `strategy` |
| --- | --- | --- | --- |
| `StrictFixedSizeChunker` | Corpus của một chapter | Chapter | `strict_fixed` |
| `GlobalRecursiveChunker` | Corpus của một chapter | Chapter | `global_recursive` |
| `StructureAwareRecursiveChunker` | Một đơn vị section | Chapter và đơn vị section | `structure_aware_recursive` |

### Strict fixed-size

- Cửa sổ có độ dài tối đa `chunk_size`, bước tiến là `chunk_size - chunk_overlap`.
- Có thể cắt giữa từ, câu hoặc section.
- Hai cửa sổ liên tiếp trong chapter có phần giao dài đúng `chunk_overlap`, trừ khi chỉ có một chunk thì không có cặp để đo.
- Chunk cuối có thể ngắn hơn `chunk_size`. Dừng khi đã chạm cuối corpus, không tạo thêm cửa sổ dư chỉ chứa overlap.

### Global recursive

- Splitter được chạy một lần cho mỗi chapter với cấu hình separator đã chốt.
- Section boundary chỉ là separator trong văn bản; chunk có thể chứa nhiều section.
- `chunk_overlap` là cấu hình mong muốn, overlap thực tế có thể nhỏ hơn hoặc bằng 0.

### Structure-aware recursive

- Dùng cùng cấu hình recursive nhưng xử lý từng đơn vị section riêng.
- Mỗi chunk thuộc đúng một đơn vị section; không overlap qua đơn vị kế tiếp.
- Có thể tạo chunk ngắn do đơn vị ngắn hoặc phần dư cuối đơn vị.

## 6. Schema đầu ra

Mỗi chunk là `Document(page_content=..., metadata=...)`. Tất cả chiến lược dùng chung bộ trường dưới đây để caller đọc kết quả nhất quán.

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `page_content` | `str` | Nội dung chunk, không rỗng và dài tối đa `chunk_size`. |
| `metadata.strategy` | `str` | Một trong ba giá trị ở mục 5. |
| `metadata.chunk_index` | `int` | Thứ tự chunk trong chapter, bắt đầu từ 0. |
| `metadata.chapter_number` | `str` | Chapter của chunk. |
| `metadata.chapter_title` | `str \| null` | Title chapter sau xử lý ở mục 4.2. |
| `metadata.section_ids` | `list[str]` | Các section mà khoảng chunk giao nhau, không lặp, theo thứ tự xuất hiện. |
| `metadata.section_id` | `str \| null` | ID nếu chỉ giao một đơn vị section; `null` nếu giao nhiều đơn vị. |
| `metadata.section_number` | `str \| null` | Số/nhãn chuẩn hóa của đơn vị duy nhất; `null` khi giao nhiều đơn vị. |
| `metadata.section_title` | `str \| null` | Title của đơn vị duy nhất; `null` khi giao nhiều đơn vị. |
| `metadata.parent_section_id` | `str \| null` | Parent của đơn vị duy nhất; `null` khi giao nhiều đơn vị. |
| `metadata.section_unit_index` | `int \| null` | Chỉ số đơn vị duy nhất trong chapter; `null` khi giao nhiều đơn vị. |
| `metadata.offset_scope` | `str` | `"chapter"` cho hai chiến lược đầu; `"section"` cho structure-aware. |
| `metadata.start_index` | `int` | Offset bắt đầu trong văn bản thuộc `offset_scope`. |
| `metadata.end_index` | `int` | Offset kết thúc, không bao gồm ký tự tại vị trí này. |
| `metadata.sources` | `list[dict]` | Các phần record nguồn thực sự giao với chunk. |

`start_index`/`end_index` dùng khoảng nửa mở `[start_index, end_index)` và phải thỏa:

```python
source_text[start_index:end_index] == chunk.page_content
end_index - start_index == len(chunk.page_content)
```

Ở đây `source_text` là corpus chapter hoặc text của đơn vị section đã dựng ở mục 4.4. Offset không trỏ vào PDF hay `raw_text`. Recursive splitter có thể bỏ whitespace ở hai đầu chunk; offset phải xác định đúng nội dung sau thao tác đó, kể cả khi text nguồn có đoạn lặp lại.

### 6.1. Schema một phần nguồn trong `sources`

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `record_id` | `str` | `id` gốc từ Parser. |
| `pdf_page` | `int` | Trang gốc, đánh số từ 0. |
| `type` | `str` | Loại content gốc. |
| `bbox` | Dãy 4 số | Bbox của toàn record gốc. |
| `section_number` | `str \| null` | Giá trị gốc của Parser, trước khi tạo nhãn intro. |
| `section_title` | `str \| null` | Title gốc của Parser. |
| `source_start_index` | `int` | Bắt đầu phần được dùng trong `record.text`. |
| `source_end_index` | `int` | Kết thúc phần được dùng trong `record.text`, exclusive. |
| `chunk_start_index` | `int` | Bắt đầu phần tương ứng trong `chunk.page_content`. |
| `chunk_end_index` | `int` | Kết thúc phần tương ứng trong `chunk.page_content`, exclusive. |
| `metadata` | `dict` | Bản sao metadata gốc của record. |

Hai khoảng phải ánh xạ cùng nội dung:

```python
record.text[source_start_index:source_end_index] == (
    chunk.page_content[chunk_start_index:chunk_end_index]
)
```

- Không gắn mọi record của section vào mọi chunk; chỉ gắn các record có text thực sự giao với chunk.
- Một record có thể xuất hiện trong nhiều chunk, với các khoảng nguồn khác nhau.
- `bbox` là tọa độ toàn record; không diễn giải nó thành bbox chính xác của một phần text đã cắt.
- Heading và joiner không có record nguồn. Chunk chỉ chứa heading có thể có `sources=[]` nhưng vẫn có chapter/section metadata.
- Có thể tra lại `raw_text` từ `record_id`; không cần sao chép toàn bộ `raw_text` vào mỗi chunk.

### 6.2. Liên kết footnote

- Khi bỏ footnote khỏi nội dung chunk, vẫn giữ `sources[].metadata.footnote_refs` của record nội dung.
- Các reference này thuộc **record nguồn**; không khẳng định mọi reference của record nằm trong mọi chunk cắt ra từ record đó.
- `raw_offset` vẫn là offset trong `raw_text` gốc. Không đổi nó thành offset của chunk hoặc `text` sạch.
- Để lấy nội dung note, caller tra `footnote_id` trong toàn bộ record Parser, kể cả những note không được đưa vào chunk.
- Khi `include_footnotes=True`, metadata riêng của note được giữ trong nguồn tương ứng.

## 7. Ví dụ xuyên suốt

Với record ví dụ ở mục 2, cấu hình mặc định và `StructureAwareRecursiveChunker`, một kết quả dự kiến là:

```json
{
  "page_content": "[[SECTION 1.1: Introduction]]\nA database stores related data.",
  "metadata": {
    "strategy": "structure_aware_recursive",
    "chunk_index": 0,
    "chapter_number": "1",
    "chapter_title": "Databases and Database Users",
    "section_ids": ["sec_1.1"],
    "section_id": "sec_1.1",
    "section_number": "1.1",
    "section_title": "Introduction",
    "parent_section_id": "chapter_1",
    "section_unit_index": 0,
    "offset_scope": "section",
    "start_index": 0,
    "end_index": 61,
    "sources": [{
      "record_id": "example-document:p34:b3",
      "pdf_page": 34,
      "type": "normal_text",
      "bbox": [117.0, 373.8, 467.4, 444.3],
      "section_number": "1.1",
      "section_title": "Introduction",
      "source_start_index": 0,
      "source_end_index": 31,
      "chunk_start_index": 30,
      "chunk_end_index": 61,
      "metadata": {"footnote_refs": []}
    }]
  }
}
```

Đây là dạng biểu diễn JSON của `page_content` và `metadata`, thuận tiện để xem schema; API vẫn trả về `Document`.

## 8. Báo lỗi và tiêu chí hoàn thành

Dùng `ValueError` khi cấu hình hoặc record vi phạm contract, kèm thông tin đủ để sửa: tên trường, vị trí/ID record và nguyên nhân. Không chuyển `None` hoặc dữ liệu sai kiểu thành chuỗi như `"None"` để tiếp tục chunk.

Các kiểm tra cần đạt sau khi triển khai:

1. Không sửa input; giữ thứ tự các record và các phần text của chúng theo quy tắc đã chốt.
2. Mọi chunk không rỗng, dài tối đa `chunk_size`, có offset đúng và metadata serializable bằng JSON.
3. Cả ba chiến lược không trộn chapter; structure-aware không trộn đơn vị section.
4. Fixed-size có phần giao đúng cấu hình; recursive được đo theo overlap thực tế.
5. Các khoảng trong `sources` xác định đúng record và phần text nguồn, kể cả nội dung lặp lại.
6. Nhãn intro và hierarchy khớp quy tắc; metadata nguồn vẫn giữ các giá trị `null` gốc.
7. Trên cùng đầu vào Chapter 1 và cùng cấu hình, class và Experiment 1 cho nội dung chunk, offset và metadata cấu trúc tương đương. Metadata truy vết mới là phần bổ sung.

Số chunk đã ghi nhận khi kiểm tra là `80 / 102 / 102` cho A/B/C. Đây là mốc cho dữ liệu và cấu hình hiện tại; tiêu chí chính là so sánh từng chunk giữa notebook và class cùng lần chạy.

## 9. Các quyết định đã chốt

Những lựa chọn áp dụng chung cho code và test gồm:

| Điểm đã chốt | Quy tắc |
| --- | --- |
| Phạm vi đầu vào | Một PDF, cho phép nhiều chapter; chunk độc lập theo chapter. |
| Kiểm tra schema | Yêu cầu schema đầy đủ của Parser, chấp nhận trường bổ sung. |
| Thiếu chapter | Báo lỗi với record còn lại sau lọc. |
| Thiếu section | Dùng nhãn kỹ thuật `<chapter>.intro`, giữ giá trị gốc trong `sources`. |
| Section xuất hiện lại | Giữ các đoạn liên tiếp thành các đơn vị riêng, có `section_unit_index`. |
| Heading | Giữ `[[SECTION ...]]` để khớp thực nghiệm. |
| Footnote | Mặc định loại khỏi nội dung, giữ liên kết trong metadata nguồn. |
| Metadata truy vết | Giữ record ID, trang, bbox, metadata và các khoảng ký tự tương ứng. |
| Đầu ra | `list[Document]`, chung schema metadata cho ba chiến lược. |

Khi thay đổi contract, cập nhật code, test và cách dựng dữ liệu trong notebook tương ứng. Có thể chạy pipeline thử bằng `python scripts/run_chunking_pipeline.py` từ project root; script lưu đủ output của ba chiến lược vào `output/chunks/` và in một mẫu của mỗi chiến lược.
