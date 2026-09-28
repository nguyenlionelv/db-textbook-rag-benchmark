# db-rag-eval

Dự án khảo sát PDF và xây dựng pipeline xử lý dữ liệu đơn giản cho RAG.

## Cấu trúc

```text
db-rag-eval/
├── data/
│   ├── raw/
│   └── interim/
├── docs/
├── notebooks/ 
│   ├── 01_pdf_inspection.ipynb
|   └── 02_pdf_chunking.ipynb
├── output/
├── src/
│   └── ingestion/
│       └── textbook_parser.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

Đặt PDF gốc vào `data/raw/` và dữ liệu trung gian vào `data/interim/`.
Các thư mục dữ liệu và file PDF được bỏ qua khi commit Git.

## Thiết lập Python trên Windows

Dùng Python 3.11. Từ thư mục dự án, chạy:

```powershell
cd db-rag-eval
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

Nếu dùng PowerShell, có thể kích hoạt bằng `.venv\Scripts\Activate.ps1`.

Mở `notebooks/01_pdf_inspection.ipynb` để bắt đầu khảo sát PDF.
Các file trong `src/ingestion/` là khung ban đầu cho bước trích xuất và làm sạch văn bản.

Parser sử dụng PyMuPDF; các Chunker sử dụng `langchain-core` và
`langchain-text-splitters`. Các thư viện được liệt kê trong `requirements.txt`.

## Chạy thử pipeline Chunker

Với input `output/parsed_chapter1.jsonl` đã được Parser tạo ra, chạy từ project root:

```powershell
.\.venv\Scripts\python.exe scripts\run_chunking_pipeline.py
```

Script nạp record, chạy cả ba Chunker, kiểm tra các khoảng text nguồn và in một
mẫu input/output gồm nội dung cùng metadata. Output đầy đủ được lưu thành ba file
JSONL trong `output/chunks/`. Có thể đổi input, kích thước, overlap và số mẫu in ra
bằng các hằng số ở đầu script. Contract được mô tả tại
[docs/chunker_contract.md](docs/chunker_contract.md).
