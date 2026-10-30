# db-rag-eval

A project for building and evaluating Retrieval-Augmented Generation (RAG) on PDF textbooks. The current experiments use an English database textbook and questions in English and Vietnamese.

```text
PDF -> Parsing -> Chunking -> Embedding -> Qdrant -> Retrieval -> Generation
```

## Current status

| Component | Implementation | Status |
| --- | --- | --- |
| Parsing | Text extraction, document structure, metadata and footnotes | Implemented |
| Chunking | Fixed-size, recursive and structure-aware splitting | Implemented |
| Embedding and storage | Multilingual E5 embeddings and a persistent Qdrant Local index | Implemented and tested |
| Retrieval | BM25 and dense search, RRF fusion and BGE reranking | Evaluated on 32 queries |
| Generation | Context assembly, token budget, local Ollama calls and citation validation | Implemented |
| RAG evaluation | Selected context and retrieved context, answer review and result export | 46 dev results reviewed; Generation eval has not run |
| Full-book evaluation and UI | Corpus expansion and user interface | Pending |

The evaluated pilot index contains **233 chunks with source references from Chapters 1-3**. The results below apply to this pilot corpus.

## Project structure

```text
db-rag-eval/
├── data/
│   └── raw/              # Input PDF files
├── docs/
│   ├── materials/        # Theory and methods
│   └── pipelines/        # Pipeline plans and data contracts
├── notebooks/            # Experiments
├── output/               # Processed data and experiment results
├── scripts/              # Pipeline and runtime scripts
├── src/
│   ├── ingestion/        # Parsing, chunking, embedding and indexing
│   ├── retrieval/        # Hybrid retrieval and reranking
│   └── generation/       # Context assembly and answer generation
├── tests/                # Automated tests
├── requirements.txt
└── README.md
```

Input PDFs, model caches and Qdrant databases are excluded from Git. The repository includes pilot text data and selected experiment results.

## Setup

The reference environment uses **Python 3.13, Windows, PowerShell and CPU inference**. Run the following commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Notebook 04 also requires `FlagEmbedding` for the BGE-M3 experiments:

```powershell
python -m pip install FlagEmbedding
```

Start Jupyter and select the `.venv` kernel:

```powershell
jupyter notebook
```

To repeat PDF extraction, place the textbook in `data/raw/` and set its path in the notebook. The later pilot experiments use the saved files in `output/`.

## Build the pilot index

The indexing script reads `output/chunks/structure_aware_recursive.jsonl`, downloads the E5 model on its first run and creates the local Qdrant index:

```powershell
python scripts/run_indexing_pipeline.py --model-cache output/model_cache/e5_hub
```

After the model has been cached, the script can run offline:

```powershell
python scripts/run_indexing_pipeline.py --model-cache output/model_cache/e5_hub --offline
```

The default database path is `output/qdrant_db`. Close any notebook client using this database before running the script; Qdrant Local allows one process to open a database at a time.

## Run local Generation

### Prepare the model files

The Generation notebook uses `intfloat/multilingual-e5-small`, `BAAI/bge-reranker-v2-m3` and `qwen3:1.7b`. Build the pilot index first to prepare the E5 cache.

Download the reranker to the cache expected by the notebook:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', cache_dir='output/model_cache/reranker_hub')"
```

Save the Qwen tokenizer for local token counting:

```powershell
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e').save_pretrained('output/model_cache/qwen3_tokenizer')"
```

These downloads require an internet connection. See the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download) for cache settings.

Download the portable Windows package from the [Ollama Windows instructions](https://docs.ollama.com/windows#standalone-cli). Extract the executable and its libraries into `output/model_cache/ollama_runtime/`. The startup script expects `output/model_cache/ollama_runtime/ollama.exe`.

### Start Ollama

Run the startup script from the repository root and keep the terminal open:

```powershell
.\scripts\start_ollama.ps1
```

The server listens at `http://127.0.0.1:11434` and stores models in `output/model_cache/ollama_models/`. The script configures its child process without changing Windows environment variables.

In a second terminal, download the Generation model once:

```powershell
.\output\model_cache\ollama_runtime\ollama.exe pull qwen3:1.7b
```

The model download uses the running local server. See the [Ollama CLI reference](https://docs.ollama.com/cli) for model commands.

### Run the notebook

Open [notebook 07](notebooks/07_rag_generation.ipynb). Its default configuration uses CPU inference, a 4,096-token context window and up to 256 generated tokens.

1. Run the setup, context inspection and warm-up cells.
2. Run the dev cases with selected context and retrieved context.
3. Review the answers in `manual_review.csv`, then rerun the review and summary cells to load the scores.
4. Set `RUN_EVAL = True` only after completing the dev review and choosing the final configuration. Run the eval cases, review their answers and rerun the export cells.

Each run saves a manifest, per-query results, review scores and a summary under `output/generation/<run-id>/`. Rerunning the configuration cell creates a new run ID. A dev run can take about 20 minutes on the reference CPU setup, mainly due to reranking and text generation.

Stop Ollama with `Ctrl+C` in its terminal, or run:

```powershell
.\scripts\start_ollama.ps1 -Stop
```

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [01_pdf_inspection](notebooks/01_pdf_inspection.ipynb) | Inspect PDF text, layout and parsing output |
| [02_pdf_chunking](notebooks/02_pdf_chunking.ipynb) | Compare chunking methods |
| [03_pdf_embedding](notebooks/03_pdf_embedding.ipynb) | Create E5 embeddings and store them in Qdrant |
| [04_embedding_comparison](notebooks/04_embedding_comparison.ipynb) | Compare lexical, static, dense, sparse and multi-vector representations |
| [05_retrieval_comparison](notebooks/05_retrieval_comparison.ipynb) | Compare retrieval methods |
| [06_retrieval_pipeline](notebooks/06_retrieval_pipeline.ipynb) | Evaluate the shared retrieval modules |
| [07_rag_generation](notebooks/07_rag_generation.ipynb) | Compare context sources, generate answers and review results |

## Recorded results

### Retrieval

The retrieval benchmark contains 32 queries: 16 dev and 16 eval, with equal numbers of English and Vietnamese queries in each split. Hybrid retrieval with reranking produced the following eval results:

| Language | Queries | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: |
| English | 8 | 1.000 | 1.000 |
| Vietnamese | 8 | 0.875 | 0.875 |

See the [retrieval report](output/retrieval/hybrid_reranking/report.md) for method comparisons and timing. The metrics use the annotated evidence spans in the pilot benchmark.

### Generation

The [saved dev run](output/generation/20261030_113151_222095/manifest.json) contains 46 reviewed results: 24 with selected context and 22 with retrieved context. Selected context uses chunks chosen from known evidence; retrieved context uses live hybrid retrieval and reranking.

| Context | Language | Correct answers | Correct rate |
| --- | --- | ---: | ---: |
| Selected | English | 11/12 | 91.7% |
| Selected | Vietnamese | 7/12 | 58.3% |
| Retrieved | English | 10/11 | 90.9% |
| Retrieved | Vietnamese | 7/11 | 63.6% |

The selected-context group includes two empty-context cases that are excluded from the retrieved-context group, so the groups differ in size. Correctness, evidence support and citation support are scored separately; a correct answer can still contain unsupported claims.

See the [summary](output/generation/20261030_113151_222095/summary.csv) and [review rubric](output/generation/20261030_113151_222095/review_rubric.md). Generation eval has not run. Some existing eval topics were inspected in earlier experiments, which limits their use as an independent final benchmark.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest -q -ra
```

The latest recorded run passed **212 tests**, with **1 skipped**. Set `RUN_EMBEDDING_INTEGRATION=1` to enable the real E5 test. It uses a cached model or downloads it on the first run.

To run only the Generation tests:

```powershell
python -m pytest tests/test_textbook_generation.py -q
```

These tests use fixed responses and do not require a running Ollama server.

## Next work

- Improve Vietnamese answer quality, evidence support and refusal behavior using the dev cases.
- Choose the final Generation configuration, run eval and review the exported answers.
- Compare model and context settings with the same questions and review criteria.
- Validate full-book parsing, chunk coverage and retrieval before expanding the evaluation.
- Add a user interface after the pipeline and evaluation are stable.

## Documentation

- [Embedding methods](docs/materials/EmbeddingMaterials.md)
- [Retrieval methods](docs/materials/RetrievalMaterials.md)
- [Generation methods](docs/materials/GenerationMaterials.md)
- [Chunking data contract](docs/pipelines/chunker_contract.md)
- [Indexing plan](docs/pipelines/indexing_plan.md)
- [Retrieval plan](docs/pipelines/retrieval_plan.md)
