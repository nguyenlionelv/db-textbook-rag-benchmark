# DB Notes

A small textbook question-answering application with source excerpts and PDF page references.
The React frontend uses the existing Python RAG pipeline through FastAPI.

## Requirements

- Docker Desktop with Linux containers and Docker Compose for the one-command setup.
- Python 3.13 and Node.js 22.19 or a compatible newer version only for native development.
- The repository's saved full-book Qdrant index and matching cached E5 and BGE models for live retrieval.
- Ollama with `qwen3:4b-instruct-2507-q4_K_M` and its matching cached tokenizer for Local generation.
- Provider configuration and an API key for API generation. OpenAI uses its input-token counting endpoint; other providers require a matching tokenizer or message counter.

## Run with Docker

Start Docker Desktop with Linux containers. From the repository root, run:

```powershell
docker compose up --build -d
```

Open `http://localhost:8000`. The image builds the frontend, Compose starts Ollama with the cached model,
and the app converts the saved index paths inside the container. No host Ollama process, manual frontend build,
environment file, or manifest editing is required for the default Local configuration.

The existing Qdrant index and model caches under `output` must be present. The first build downloads images and Python/Node dependencies.
Stop a native backend using the same Qdrant directory before starting Compose.
Model cache and index data remain on the host when containers stop. Optional API settings can be placed in `app/.env`.

```powershell
docker compose logs -f app
docker compose down
```

The app shares Ollama's container network namespace to preserve localhost-only Local generation.
Only the web application port is published to the host; Docker Desktop host networking is not required.
Compose uses the CPU-only `alpine/ollama` image, pinned by digest, with Ollama 0.40.15.
The first Local request can take several minutes to load the model on CPU. Compose allows 300 seconds for the model request;
the browser allows 600 seconds for retrieval and generation together.

## Install for native development

Run from the repository root in PowerShell. Use the existing `.venv`, or create it with `py -3.13 -m venv .venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
npm --prefix app/frontend ci
if (-not (Test-Path app/.env)) { Copy-Item app/.env.example app/.env }
```

## Run locally

Start Ollama in a separate terminal from the repository root. The supplied script uses the project's cached runtime and model files.

```powershell
./scripts/start_ollama.ps1
```

Start the backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --env-file app/.env
```

Start the frontend in another terminal:

```powershell
npm --prefix app/frontend run dev
```

Open `http://127.0.0.1:5173`. Vite forwards `/api` requests to port 8000.
The backend loads resources on one worker thread. Health checks remain available during initialization and inference.
If a backend is unavailable at startup, correct its configuration and restart the API.

The sidebar selects the model and answer language. Answers include citation buttons that open source excerpts and PDF page references.
The application keeps the current conversation in browser memory. Each question is processed independently.

## Configuration

Server settings use the `RAG_` prefix. See [.env.example](.env.example).
Relative paths are resolved from the repository root.

| Setting | Purpose |
|---|---|
| `RAG_MODE` | `live` enables inference; `examples` disables live inference |
| `RAG_ENABLE_LOCAL` | Enable Local model initialization |
| `RAG_INDEX_MANIFEST` | Saved-index manifest with the correct `qdrant_path` |
| `RAG_E5_CACHE`, `RAG_RERANKER_CACHE` | Cached retrieval model directories |
| `RAG_LOCAL_MODEL`, `RAG_LOCAL_TOKENIZER` | Matching Ollama model and tokenizer |
| `RAG_API_MODEL`, `RAG_API_URL` | Optional OpenAI-compatible model and API root |
| `RAG_API_TOKEN_COUNTER` | `module:function` counting full API messages with model-specific overhead |
| `RAG_API_TOKENIZER` | Alternative local tokenizer with the provider's matching chat template |
| `RAG_API_KEY_ENV` | Secret environment variable name; default `LLM_API_KEY` |
| `RAG_API_TOKEN_PARAMETER` | `max_completion_tokens` or `max_tokens` |
| `RAG_API_REASONING_EFFORT` | Optional provider reasoning setting, such as `none` |
| `RAG_NUM_CTX`, `RAG_MAX_NEW_TOKENS` | Context and completion token budgets |
| `RAG_ALLOWED_ORIGINS` | Comma-separated frontend origins |
| `RAG_REQUESTS_PER_HOUR` | Global accepted-request limit per process; default 60 |

For OpenAI, set `RAG_API_URL=https://api.openai.com/v1`, choose `RAG_API_MODEL`, and set `LLM_API_KEY`.
For `gpt-6-luna`, `RAG_API_REASONING_EFFORT=none` disables reasoning for direct answers.
Leave both tokenizer settings empty to count input through OpenAI's `/responses/input_tokens` endpoint.
That endpoint uses the Responses input format; Chat Completions usage is recorded separately because formatting overhead may differ.

Other providers require exactly one of `RAG_API_TOKEN_COUNTER` and `RAG_API_TOKENIZER`.
The counter must return an integer and match the provider model. The Local tokenizer is not an API default.
API mode is listed when its configuration is valid. Provider connectivity and credentials are checked when a question is sent.
Keep provider keys and URLs on the server. Frontend environment variables are public.

## Build and deployment

```powershell
npm --prefix app/frontend run build
```

When `app/frontend/dist` exists at backend startup, FastAPI also serves the built frontend at `/`.
For Vercel, use `app/frontend` as the root directory, `npm run build` as the build command, and `dist` as the output directory.
Set `VITE_API_BASE_URL` to the backend's HTTPS origin and include the frontend origin in `RAG_ALLOWED_ORIGINS`.

Run one API process. The existing local Qdrant directory cannot be opened by multiple independent clients.
On Linux, transfer a consistent copy of the index and cached models to persistent disk. Prepare a deployment manifest with `qdrant_path` pointing to the transferred directory, then set `RAG_INDEX_MANIFEST`.
The runtime reads the index and does not ingest documents. Verify model compatibility, RAM, and latency on the target host before enabling live traffic.

The multi-stage Dockerfile includes the built frontend and backend. Use the root Compose command above to supply the index,
model caches, and Local Ollama automatically. `app/compose.yaml` contains the service configuration.
Cloud deployment and Linux/ARM behavior require separate verification on the target machine.

## Tests

Run backend tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r app/requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest app/tests -q --basetemp output/pytest_tmp/app
```

Run browser tests from `app/frontend`. Install Chromium once with `npx playwright install chromium`.

```powershell
npm run test:e2e
```

The standard browser suite mocks API responses. Set `LIVE_RAG=1` and run the test matching `live Local` with a live backend and Ollama to verify a real model call.
Do not run the standalone retrieval evaluator against the same Qdrant directory while the API owns it.

The browser tests use [saved answer fixtures](frontend/public/examples.json) without model calls. The interface does not include a recorded-example view.
Citation IDs identify selected sources; their validity does not establish factual correctness.
