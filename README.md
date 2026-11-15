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
| Full-book ingestion | Chapters 1-30 and Appendices A-C, with source coverage and index checks | Complete: 17,199 records and 6,040 indexed chunks |
| Retrieval | BM25 and dense search, RRF fusion and BGE reranking | Evaluated on 32 pilot queries and 100 full-book queries |
| Generation | Context assembly, token budget, local Ollama and OpenAI-compatible API calls, citation validation | Implemented |
| Pilot Generation evaluation | Selected context and saved retrieval context, answer review and result export | 46 dev and 32 eval results reviewed |
| RAG pipeline | Live retrieval connected to Local or API generation | Implemented |
| RAG end-to-end evaluation | Answer quality and latency with live retrieval and generation | Pending |
| Retrieval improvement | Retrieval quality and response time, particularly for Vietnamese queries | Pending |
| Application | React and FastAPI interface with model/language selection and source excerpts | Implemented and tested locally |
| Public deployment | Hosted demo for external users | Pending |

The pilot index contains **233 chunks with source references from Chapters 1-3**. The full-book index contains **6,040 chunks from 30 chapters and three appendices**. Their retrieval results are reported separately below.

## Project structure

```text
db-rag-eval/
├── app/                  # React frontend, FastAPI backend and Docker configuration
├── data/
│   ├── raw/              # Input PDF files
│   └── benchmarks/       # Bilingual questions and source annotations
├── docs/
│   ├── materials/        # Theory and methods
│   └── pipelines/        # Pipeline plans and data contracts
├── notebooks/            # Experiments
├── output/               # Processed data and experiment results
├── scripts/              # Pipeline and runtime scripts
├── src/
│   ├── ingestion/        # Parsing, chunking, embedding and indexing
│   ├── retrieval/        # Hybrid retrieval and reranking
│   ├── generation/       # Context assembly and answer generation
│   └── rag_pipeline.py   # Live retrieval and generation
├── tests/                # Automated tests
├── requirements.txt
└── README.md
```

Input PDFs, model caches, Qdrant databases and local environment files are excluded from Git. The repository includes processed textbook records, chunks, benchmark questions and selected experiment results.

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

Notebook 07 uses `qwen3:4b-instruct-2507-q4_K_M` with selected chunks and the [saved retrieval contexts](output/retrieval/generation_contexts/manifest.json). It reads the saved top-5 results without loading E5, the reranker or Qdrant.

From the repository root, use the `.venv` environment to download the pinned tokenizer and save it under `output/model_cache/qwen3_4b_instruct_tokenizer/`:

```powershell
python scripts/prepare_generation_tokenizer.py
```

The tokenizer download requires an internet connection. See the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download) for cache settings.

Download the portable Windows package from the [Ollama Windows instructions](https://docs.ollama.com/windows#standalone-cli). Extract the executable and its libraries into `output/model_cache/ollama_runtime/`. The startup script expects `output/model_cache/ollama_runtime/ollama.exe`.

### Start Ollama

Run the startup script from the repository root and keep the terminal open:

```powershell
.\scripts\start_ollama.ps1
```

The server listens at `http://127.0.0.1:11434` and stores models in `output/model_cache/ollama_models/`. The script configures its child process without changing Windows environment variables.

In a second terminal, download the Generation model once:

```powershell
.\output\model_cache\ollama_runtime\ollama.exe pull qwen3:4b-instruct-2507-q4_K_M
```

The model download uses the running local server. See the [Ollama CLI reference](https://docs.ollama.com/cli) for model commands.

### Run the notebook

Open [notebook 07](notebooks/07_rag_generation.ipynb). Its default configuration uses CPU inference, a 4,096-token context window and up to 256 generated tokens.

1. Run the setup, context inspection and warm-up cells.
2. Run the dev cases with selected context and retrieved context.
3. Review the answers in `manual_review.csv`, then rerun the review and summary cells to load the scores.
4. Set `RUN_EVAL = True` only after completing the dev review and choosing the final configuration. Run the eval cases, review their answers and rerun the export cells.

Each run saves a manifest, per-query results, review scores and a summary under `output/generation/<run-id>/`. Rerunning the configuration cell creates a new run ID. Generation timing depends on the context and answer length. The saved eval run averages about 15–18 seconds per answer with selected context and 36–37 seconds with saved retrieval context on CPU; these times exclude live retrieval.

Stop Ollama with `Ctrl+C` in its terminal, or run:

```powershell
.\scripts\start_ollama.ps1 -Stop
```

## Run the application

The application provides model and answer-language selection, question answering, and source excerpts with PDF page references. It supports Local Ollama and OpenAI-compatible API generation.

### Docker

Requirements:

- Docker Desktop running with Linux containers. Host Python, Node.js and an active `.venv` are not required.
- The full-book database at `output/qdrant_full_book/20261114_full_book_v2/` and retrieval model caches at `output/model_cache/e5_hub/` and `output/model_cache/reranker_hub/`.
- The cached `qwen3:4b-instruct-2507-q4_K_M` model in `output/model_cache/ollama_models/` and its tokenizer in `output/model_cache/qwen3_4b_instruct_tokenizer/`. The supplied Compose configuration starts Ollama alongside the app.

The database and model caches are not included in Git. Stop any native backend or notebook using the same Qdrant database before starting Docker.

From the repository root, build and start the application:

```powershell
docker compose up --build -d
```

Open `http://localhost:8000`. Docker builds the frontend, starts FastAPI and Ollama, and mounts the existing index and model caches. It does not run ingestion again. The first build downloads container images and dependencies; the first question may take longer while models load.

Local generation is enabled by default. For API generation, create `app/.env` from [app/.env.example](app/.env.example) and fill in `RAG_API_MODEL`, `RAG_API_URL` and `LLM_API_KEY`. Compose reads this file when the app starts. Select Local or API from the model control in the interface.

After changing app code or environment settings, rebuild and restart the app service:

```powershell
docker compose up -d --build app
```

View backend logs:

```powershell
docker compose logs -f app
```

Stop the application:

```powershell
docker compose down
```

The index and cached models remain on the host after the containers stop.

### CLI

The same API configuration is available to the CLI. Stop the Docker app first because Qdrant Local allows only one process to open its database. From the repository root with `.venv` active, run:

```powershell
python scripts/run_rag_pipeline.py --backend api --question "What is a database?" --language en
```

See the [application guide](app/README.md) for native development, API settings, tests and deployment requirements. Local application checks do not constitute a completed end-to-end RAG evaluation.

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

#### Pilot benchmark: 32 queries

The retrieval benchmark contains 32 queries: 16 dev and 16 eval, with equal numbers of English and Vietnamese queries in each split. Hybrid retrieval with reranking produced the following eval results:

| Language | Queries | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: |
| English | 8 | 1.000 | 1.000 |
| Vietnamese | 8 | 0.875 | 0.875 |

See the [retrieval report](output/retrieval/hybrid_reranking/report.md) for method comparisons and timing. The metrics use the annotated evidence spans in the pilot benchmark.

#### Full-book benchmark: 100 queries

The [frozen benchmark](output/retrieval/full_book_100/20261114_frozen_v1/queries.jsonl) contains **100 queries: 50 Vietnamese and 50 English**, arranged as 50 translated pairs. There are 60 dev queries and 40 eval queries; paired translations and topic families remain in the same split.

Questions cover Database Systems and DBMS topics in Chapters 1-2. Retrieval searches all **6,040 indexed chunks from Chapters 1-30 and Appendices A-C**, without a chapter filter. The recorded configuration uses multilingual E5 small, BM25/RRF with 20 candidates, and BGE reranking to return five chunks.

Results across all 100 queries, including dev and eval:

| Language | Method | Queries | Hit@5 | MRR@5 | All evidence@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| English | Hybrid | 50 | 0.860 | 0.767 | 0.760 |
| English | Hybrid + reranker | 50 | 0.920 | 0.877 | 0.840 |
| Vietnamese | Hybrid | 50 | 0.360 | 0.225 | 0.320 |
| Vietnamese | Hybrid + reranker | 50 | 0.600 | 0.567 | 0.540 |

On the 40-query eval split, Hybrid + reranker produced:

| Language | Queries | Hit@5 | MRR@5 | All evidence@5 |
| --- | ---: | ---: | ---: | ---: |
| English | 20 | 0.950 | 0.925 | 0.800 |
| Vietnamese | 20 | 0.500 | 0.500 | 0.400 |

Hit@5 and MRR@5 measure matches to any annotated evidence span. All evidence@5 requires every annotated evidence group to appear in the top five. The annotations are not exhaustive relevance labels, so a missed label can still correspond to a useful or equivalent passage. The translated pairs are not independent questions, and the annotations have not received independent human review. Topics overlap earlier project experiments, so eval is not a fully unseen final benchmark.

The recorded CPU run averaged 13.30 seconds per English query and 14.83 seconds per Vietnamese query after reranking. These times include live retrieval and reranking, exclude setup and warm-up, and come from one pass rather than a latency distribution.

See the [report](output/retrieval/full_book_100/20261114_frozen_v1/report.md), [dev/eval summary](output/retrieval/full_book_100/20261114_frozen_v1/summary.csv), [overall results](output/retrieval/full_book_100/20261114_frozen_v1/overall.csv) and [configuration](output/retrieval/full_book_100/20261114_frozen_v1/frozen_config.json). The corpus and questions differ from the pilot benchmark, so the two sets of scores are not directly comparable. This run evaluates retrieval only; generated-answer quality and retrieval performance improvements remain pending.

### Generation

The [saved Generation run](output/generation/20261030_160927_838100/manifest.json) uses `qwen3:4b-instruct-2507-q4_K_M` and contains **78 reviewed results**: 46 dev results and 32 eval results. The eval set contains 16 questions, each evaluated under both context conditions.

Selected context uses chunks chosen from known evidence. Retrieved context uses verified top-5 snapshots from hybrid retrieval with reranking. Generation timing covers context preparation, the local model call and output validation; retrieval and end-to-end timing are not measured in this run.

| Split | Context | Language | Correct answers | Correct rate |
| --- | --- | --- | ---: | ---: |
| Dev | Selected | English | 11/12 | 91.7% |
| Dev | Selected | Vietnamese | 9/12 | 75.0% |
| Dev | Retrieved snapshot | English | 10/11 | 90.9% |
| Dev | Retrieved snapshot | Vietnamese | 10/11 | 90.9% |
| Eval | Selected | English | 8/8 | 100.0% |
| Eval | Selected | Vietnamese | 7/8 | 87.5% |
| Eval | Retrieved snapshot | English | 8/8 | 100.0% |
| Eval | Retrieved snapshot | Vietnamese | 7/8 | 87.5% |

The dev selected-context group includes two empty-context cases excluded from the retrieved-context group. The eval groups each contain eight English and eight Vietnamese questions. Correctness, evidence support and citation support are scored separately; a correct answer can still contain unsupported claims.

The eval errors concern added transaction properties with selected context and a data-abstraction question with incomplete retrieved evidence. Some eval topics were observed in earlier experiments, so this set is not a fully independent final benchmark. The labels are based on source-based review by the coding agent.

See the [report](output/generation/20261030_160927_838100/report.md), [summary](output/generation/20261030_160927_838100/summary.csv), [per-answer review](output/generation/20261030_160927_838100/manual_review.csv) and [review rubric](output/generation/20261030_160927_838100/review_rubric.md).

## Tests

With `.venv` active, install the application test dependencies and run the test suite from the repository root:

```powershell
python -m pip install -r app/requirements-dev.txt
python -m pytest -q -ra
```

Standard tests use fixed responses and test fixtures. Set `RUN_EMBEDDING_INTEGRATION=1` to enable the real E5 test. It uses a cached model or downloads it on the first run. Browser test instructions are in the [application guide](app/README.md).

To run only the Generation tests:

```powershell
python -m pytest tests/test_textbook_generation.py tests/test_api_generation.py -q
```

These tests use fixed responses and do not require a running Ollama server or external LLM API calls.

## Next work

- Evaluate end-to-end RAG with live retrieval and generation, covering answer correctness, evidence support, citations and latency. **Pending.**
- Improve retrieval quality and response time, particularly for Vietnamese queries, using the dev split. **Pending.**
- Prepare and verify the public demo deployment. **Pending.**

## Documentation

- [Embedding methods](docs/materials/EmbeddingMaterials.md)
- [Retrieval methods](docs/materials/RetrievalMaterials.md)
- [Generation methods](docs/materials/GenerationMaterials.md)
- [Chunking data contract](docs/pipelines/chunker_contract.md)
- [Indexing plan](docs/pipelines/indexing_plan.md)
- [Retrieval plan](docs/pipelines/retrieval_plan.md)
