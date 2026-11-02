# nexus-rag

nexus-rag is a FastAPI-based chatbot application that uses LangChain and OpenAI to generate responses while keeping chat history in PostgreSQL for each authenticated user. It includes a full **RAG (Retrieval-Augmented Generation) pipeline** — upload documents to a session, and the chatbot will ground its answers using the most relevant chunks from those documents.

## Features

- FastAPI backend with chat and streaming endpoints
- JWT-based signup and login
- Persistent user, session, and message history in PostgreSQL
- LangChain prompt chaining with an OpenAI chat model
- Configurable recent-message context window via environment variable
- **RAG pipeline**: upload PDF, DOCX, TXT, MD, or CSV files per session
- **Milvus hybrid search**: semantic vectors + BM25 keyword ranking fused with RRF
- **Session-scoped retrieval**: only searches documents uploaded to the current session — no cross-session leakage
- **Queue-based document ingestion**: upload returns instantly, processing happens asynchronously via ARQ workers
- **Scalable worker architecture**: independent worker processes with retry logic and failure handling
- **Idempotent ingestion**: deterministic chunk IDs and delete-then-insert writes, so a retried job never duplicates chunks
- **Duplicate upload detection**: identical files (SHA-256) in the same session are not embedded twice
- Docker support for containerized deployment

## Architecture

```
                   Client
                     │
                     ▼
              FastAPI Upload API
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   MinIO Storage           PostgreSQL Metadata
 (original files)         (status, object_name)
        │                         │
        │                         │ Enqueue job
        │                         ▼
        │                   Redis (ARQ)
        │                         │
        │                ┌────────┴─────────┐
        │                │  ARQ Workers     │
        │                │  (scalable)      │
        │                └────────┬─────────┘
        │                         │
        └─────────────────────────┘
              Download & Process
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   Milvus Vectors          PostgreSQL Chunks
```

**Document Ingestion Flow:**
1. Client uploads file → API saves to MinIO and creates metadata row with status='queued'
2. Job enqueued in Redis with document_id
3. Worker downloads file from MinIO
4. Worker calls existing parse → chunk → embed → store pipeline
5. Status updates: queued → processing → ready/failed

**Failure handling:**
- Permanent errors (unsupported or corrupt file, no extractable text) mark the document `failed` at once, with a `failure_reason`.
- Transient errors (OpenAI, Postgres, Milvus, MinIO) set the status back to `queued` with a "Retrying…" reason and retry with exponential backoff (2s, 4s, …) up to `ARQ_MAX_TRIES` attempts. After the last attempt the document is `failed`.
- Chunk IDs are `{document_id}:{chunk_index}`, and each run deletes the document's old chunks before writing, so retries are safe.

## Tech Stack

- Python 3.11
- FastAPI
- LangChain
- LangChain OpenAI
- asyncpg
- PostgreSQL
- Milvus (vector store)
- MinIO (object storage)
- Redis (task queue)
- ARQ (async task workers)
- React (Vite + TypeScript + Tailwind)
- Docker

## Prerequisites

Before running the project, make sure you have:

- Python 3.11+
- Docker (PostgreSQL, Redis, Milvus, and MinIO all run from `docker-compose.yaml`)
- An OpenAI API key for the backend model
- A strong JWT secret for signing user tokens

## Environment Variables

Copy `.env.example` to `.env` and fill in the secrets — it documents every
variable with the same defaults described below.

```env
OPENAI_API_KEY=your_openai_api_key
JWT_SECRET=your_long_random_jwt_secret
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
PASSWORD_HASH_ITERATIONS=210000

# Origin the React frontend is served from — required for browser CORS.
FRONTEND_ORIGIN=http://localhost:5173

LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=6000

# Optional: chat model fallback (comma-separated provider:model list, tried
# in order). Defaults to a single "openai:$LLM_MODEL" entry.
# LLM_FALLBACK_CHAIN=openai:gpt-4o-mini,openai:gpt-4o

EMBEDDING_MODEL=text-embedding-3-small
# Optional: embedding model fallback, same "provider:model" format.
# Every configured model MUST produce vectors of the same dimension as
# EMBEDDING_DIM below — mixing dimensions corrupts retrieval.
# EMBEDDING_FALLBACK_CHAIN=openai:text-embedding-3-small,openai:text-embedding-3-large
EMBEDDING_DIM=1536

PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=nexus_rag
PG_USER=postgres
PG_PASSWORD=postgres
DB_POOL_SIZE=10
MAX_CHAT_TOKENS=2500

# Milvus (RAG vector store)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=doc_chunks
MILVUS_TOP_K=5
MILVUS_HYBRID_CANDIDATE_K=20
MILVUS_RRF_K=60
MILVUS_TOKEN=root:Milvus
RETRIEVAL_MAX_COSINE_DISTANCE=0.6

# Cross-encoder reranking (local model, loaded on first retrieval)
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
RERANKER_CANDIDATE_K=30
RERANKER_TOP_K=5
RERANKER_MAX_LENGTH=512

# Largest accepted upload, in bytes (default 25 MB) — larger files get 413
MAX_UPLOAD_BYTES=26214400

# MinIO (object storage for uploaded documents)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents
MINIO_SECURE=false

# Redis (ARQ task queue broker)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ARQ worker settings
ARQ_MAX_JOBS=5
ARQ_JOB_TIMEOUT=600
ARQ_MAX_TRIES=3

LOG_LEVEL=INFO
```

## Local Development

1. **Start infrastructure services** (requires Docker):

```bash
docker compose up -d postgres redis milvus minio etcd
```

This starts:
- PostgreSQL on port `5432` (database `nexus_rag`, user/password `postgres` unless overridden in `.env`)
- Redis (ARQ task queue broker) on port `6379`
- etcd (Milvus dependency)
- MinIO (object storage) on ports `9000` (API) and `9001` (console)
- Milvus Standalone on port `19530`
- Attu (Milvus admin UI) at `http://localhost:8001`

### Inspecting chunks with Attu

Attu is a local Milvus administration UI. It is independent of the FastAPI and
React applications, so chunk inspection does not expose an additional
chatbot API endpoint.

1. Start or update the service:

   ```bash
   docker compose up -d attu
   ```

2. Open [http://localhost:8001](http://localhost:8001).
3. Connect using the Docker-network address `milvus:19530` — do **not** use
   `localhost:19530`, which would refer to the Attu container itself.
4. Authenticate with the value configured in `MILVUS_TOKEN`. The development
   Compose configuration uses the standard `root` username and `Milvus`
   password unless you override it.
5. Open the `doc_chunks` collection (or the collection named by
   `MILVUS_COLLECTION_NAME`) and browse/query its entities. Each indexed chunk
   contains `text`, `filename`, `page`, `chunk_index`, `document_id`,
   `session_id`, and `user_id`, along with dense and sparse vectors.

Attu is intended for local development and administration. Do not expose port
`8001` publicly without putting it behind appropriate network access controls.

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the backend:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

5. Start the ARQ worker (in a separate terminal):

```bash
python -m src.workers.worker
```

6. Start the React frontend (in another terminal):

```bash
cd frontend
npm install
npm run dev
```

The API will be available at `http://localhost:8000`, the frontend at `http://localhost:5173`, and MinIO console at `http://localhost:9001`. See `frontend/README.md` for frontend-specific details (env vars, build, Docker).

## Tests

The unit tests use only local mocks and do not require PostgreSQL, Milvus, or
an OpenAI API call. Run them from the project root:

```bash
python -m unittest discover -s tests -v
```

Or run the same suite in the application container image:

```bash
docker compose --profile test run --rm tests
```

## Authentication

- `POST /auth/signup` — creates a user and returns an access token.
- `POST /auth/login` — verifies credentials and returns an access token.
- Authenticated requests must include `Authorization: Bearer <token>`.

### Rate limiting

Per-IP limits, backed by Redis (`REDIS_HOST`/`PORT`/`DB`, the same instance used
by the ARQ queue) so they hold across multiple worker processes:

| Endpoint                    | Limit      |
| ---------------------------- | ---------- |
| `POST /auth/login`           | 10/minute  |
| `POST /auth/signup`          | 5/minute   |
| `POST /chat`, `/chat/stream` | 20/minute  |
| `POST /documents/upload`     | 10/minute  |

A rate-limited request receives `429 Too Many Requests`.

## RAG Usage

1. Log in from the sidebar.
2. In the chat input, attach one or more files (PDF, DOCX, TXT, MD, CSV).
3. The frontend shows per-file ingestion status (⏳ queued → 🔄 processing → ✅ ready).
4. Once ready, ask questions — answers will be grounded in your document content.
5. Only documents uploaded to the current session are searched — a session with no documents gets no RAG context, regardless of what you've uploaded elsewhere.

### Milvus migration

The hybrid schema is incompatible with the old dense-only collection. Before deploying this version, drop the existing `doc_chunks` collection in Milvus, restart the API, and upload documents again. If existing Postgres document rows are retained, clear them first as well so the UI does not show stale documents as ready. This version does not rebuild Milvus automatically.

The citation schema adds a nullable `page` field to both `document_chunks` and Milvus. Because the configured Milvus 2.5 server has an immutable collection schema, an existing `doc_chunks` collection must also be recreated and its documents re-uploaded. The API fails fast with a migration message instead of silently indexing uncitable PDF chunks.

## API Endpoints

### Auth

```bash
# Sign up
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "strong-password"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "strong-password"}'
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"session_id": "demo-session", "user_query": "Hello!"}'

# Streaming
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"session_id": "demo-session", "user_query": "Tell me a short joke"}'
```

### Documents

```bash
# Upload files (session_id is a required form field)
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "session_id=demo-session" \
  -F "files=@report.pdf" \
  -F "files=@notes.txt"

# Check document status
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/documents/{document_id}/status"

# List documents for a session
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/documents?session_id=demo-session"

# Delete a document
curl -X DELETE http://localhost:8000/documents/{document_id} \
  -H "Authorization: Bearer $TOKEN"
```

### Session History

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/getSessionHistory?session_id=demo-session"
```

## Docker

Start the full stack (PostgreSQL + Redis + Milvus + etcd + MinIO + FastAPI + ARQ worker + React frontend) with:

```bash
docker compose up -d --build
```

The API is available at `http://localhost:8000` and the frontend at
`http://localhost:5173`. The frontend image bakes `VITE_API_BASE_URL` in at
build time (see `docker-compose.yaml`'s `frontend.build.args`) since Vite
inlines `VITE_*` vars into the static bundle — rebuild the `frontend` service
if that URL changes.

To containerize just the FastAPI app on its own:

```bash
docker build -t chatbot .
docker run --env-file .env -p 8000:8000 chatbot
```

**Scaling workers:**

The worker service can be scaled independently to handle higher upload volume:

```bash
# Run 3 worker instances
docker compose up -d --scale worker=3

# Check worker logs
docker compose logs -f worker
```

## Notes

- The context sent to the model is trimmed to the most recent configured number of messages. Update `MAX_CHAT_HISTORY_MESSAGES` in `.env` to change the window.
- `MILVUS_TOP_K` controls how many document chunks are retrieved per query (default 5).
- `RETRIEVAL_MAX_COSINE_DISTANCE` (default 0.6) drops dense-search candidates whose cosine distance to the query exceeds it, before ranking/reranking — a relevance floor so weakly-matching chunks aren't treated as authoritative context. 0.6 is a starting point for `text-embedding-3-small`; tune it against your own documents (retrieval scores are visible in the `SOURCES-TRACE`/`Running hybrid retrieval` debug logs).
- When `RERANKER_ENABLED=true`, the app retrieves up to `RERANKER_CANDIDATE_K`
  hybrid-search candidates, uses a local cross-encoder to select the best
  `RERANKER_TOP_K`, and sends only those chunks to the LLM. The model is
  downloaded when it is first used, so the first RAG request may take longer.
- The streaming endpoint falls back to the final end-of-stream answer when a model emits empty chunks.
- Uploading a file whose bytes already exist (and did not fail) in the same session returns the existing document with `"duplicate": true` instead of processing it again.
- Deleting a document removes its Milvus vectors first. If that fails, the API returns `502` and keeps the document so the delete can be retried — deleted documents never keep answering questions.
- Document ingestion is asynchronous — the upload endpoint returns `202 Accepted` immediately. Poll `GET /documents/{document_id}/status` to watch status change from `queued` → `processing` → `ready`/`failed`.



