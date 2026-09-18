# db-rag-eval

Dự án khảo sát PDF và xây dựng pipeline xử lý dữ liệu đơn giản cho RAG.

## Cấu trúc

```text
db-rag-eval/
├── data/
│   ├── raw/
│   └── interim/
├── notebooks/
│   └── 01_pdf_inspection.ipynb
├── src/
│   └── ingestion/
│       ├── extract_pdf.py
│       └── clean_text.py
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

Giai đoạn đầu chỉ dùng PyMuPDF, pdfplumber, Jupyter, pandas và pytest;
chưa cần LangChain, vector database hay framework RAG khác.
