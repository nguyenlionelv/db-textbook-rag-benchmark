# db-rag-eval

## Tổng quan

Dự án nghiên cứu và hiện thực hệ thống **Retrieval-Augmented Generation (RAG)** trên tài liệu PDF. Quá trình phát triển kết hợp thực nghiệm phương pháp với xây dựng các thành phần xử lý tài liệu, biểu diễn văn bản, lưu trữ vector và truy xuất thông tin.

Định hướng pipeline:

```text
PDF → Parsing → Chunking → Embedding → Vector Storage → Retrieval → Generation
```

## Cấu trúc dự án

```text
db-rag-eval/
├── data/
│   ├── raw/                    # Tài liệu đầu vào
│   └── interim/                # Dữ liệu trung gian
├── docs/
│   ├── materials/              # Tài liệu lý thuyết
│   └── pipelines/              # Kế hoạch và đặc tả quy trình
├── notebooks/                  # Thực nghiệm và hiện thực các thành phần
├── output/                     # Kết quả xử lý và dữ liệu lưu trữ
├── scripts/                    # Script thực thi pipeline
├── src/
│   └── ingestion/              # Mã nguồn Parsing và Chunking
├── tests/                      # Kiểm thử
├── requirements.txt
└── README.md
```

## Tech Stack

| Nhóm | Công nghệ |
| --- | --- |
| Ngôn ngữ | Python |
| Xử lý PDF | PyMuPDF, pdfplumber |
| Chunking | LangChain Core, LangChain Text Splitters |
| Embedding | PyTorch, Sentence Transformers, FlagEmbedding |
| Xử lý dữ liệu | NumPy, pandas, SciPy, scikit-learn |
| Vector Storage | Qdrant Local |
| Môi trường thực nghiệm | Jupyter Notebook |
| Kiểm thử | pytest |

## Installation & Setup

Môi trường tham chiếu: **Python 3.13, Windows và PowerShell**. Từ thư mục dự án, tạo môi trường ảo và cài đặt các thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt FlagEmbedding
```

Cấu hình trên sử dụng PyTorch CPU. Các cấu hình GPU được cung cấp trong [hướng dẫn cài đặt PyTorch](https://pytorch.org/get-started/locally/).

Khởi động môi trường notebook:

```powershell
jupyter notebook
```

Chọn kernel tương ứng với `.venv` và đặt tài liệu PDF tại `data/raw/`. Đường dẫn đầu vào và cấu hình model được khai báo trong các notebook. Model pretrained cần được tải ở lần sử dụng đầu tiên; Qdrant Local lưu dữ liệu trên máy và không yêu cầu server riêng.

## Tiến độ hiện tại

| Thành phần | Phạm vi | Trạng thái |
| --- | --- | --- |
| Document Processing | PDF Inspection, Parsing, Metadata | Đã hiện thực |
| Chunking | Fixed, Recursive, Structure-aware | Đã hiện thực |
| Thực nghiệm phương pháp | TF-IDF, Static, Dense, Sparse, Multi-vector, Hybrid | Đã thực nghiệm |
| Embedding & Vector Storage | E5, Qdrant, Persistence | Đã hiện thực trong notebook |
| Retrieval | Retrieval Pipeline, Reranking | Chưa triển khai thành phần hệ thống |
| Generation | Context Assembly, LLM | Chưa triển khai |
| Evaluation | End-to-end RAG Evaluation | Chưa triển khai |
