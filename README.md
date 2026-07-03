# Hybrid RAG Chatbot (SOPSearch AI)

An intelligent, production-ready Virtual Assistant designed to perform highly accurate Standard Operating Procedure (SOP) searches. This system utilizes a **Hybrid Retrieval-Augmented Generation (RAG)** approach, combining dense semantic search, sparse lexical search (BM25), and cross-encoder reranking to ensure precise and hallucination-free AI responses.

For the current repository layout, runtime modules, script folders, and generated-artifact policy, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## 🌟 Key Features
- **Hybrid Search Engine:** Combines `BAAI/bge-small-en-v1.5` embeddings (Dense) with BM25 (Sparse) in Qdrant.
- **Intelligent Reranking:** Re-ranks initial hits using `cross-encoder/ms-marco-MiniLM-L-6-v2` for near-perfect context matching.
- **Secure Authentication:** JWT-based user login and registration backed by PostgreSQL (asyncpg).
- **Modern UI/UX:** React + Vite frontend featuring glassmorphism, responsive chat interfaces, dark mode styling, and an interactive profile settings panel.
- **Generative AI:** Google Gemini 2.5 Flash produces concise answers with strictly enforced source citations.

---

## 🧠 The Hybrid RAG Flow

1. **Ingestion (Document Processing)**
   - External SOP documents/policies are fetched via API or local files.
   - The text is chunked recursively using `LangChain`.
   - Each chunk is embedded densely (`BGE-small`) and sparsely (`BM25`) and stored in the **Qdrant Vector Database**.

2. **Retrieval (Hybrid Search)**
   - When a user submits a query, it is vectorized.
   - Qdrant performs a hybrid search, retrieving the top `K` most relevant chunks across both vector space and keyword matching.

3. **Reranking (Cross-Encoder)**
   - The initial `K` chunks are paired individually with the user's query.
   - The **MS-MARCO Cross-Encoder** computes an absolute relevance score for each pair, sorting the chunks to surface the most accurate context to the top.

4. **Generation (Gemini 2.5)**
   - The top reranked chunks are formatted and passed directly into the **Gemini 2.5 Flash** context window.
   - The model generates an answer strictly based on the provided context, appending explicit metadata source citations at the end of the text.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | React, Vite, Vanilla CSS |
| **Backend API** | FastAPI, Uvicorn, Python 3 |
| **Database (Vector)** | Qdrant |
| **Database (Relational)**| PostgreSQL, SQLAlchemy (Async), Alembic |
| **Auth/Security** | JWT, python-jose, bcrypt |
| **Embeddings** | HuggingFace (`BAAI/bge-small-en-v1.5`) |
| **Reranker** | SentenceTransformers (`ms-marco-MiniLM-L-6-v2`) |
| **LLM Engine** | Google Gemini (via LangChain) |

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.12+ (managed via `uv`)
- Node.js (v18+)
- PostgreSQL installed and running locally
- Qdrant Database instance (local Docker or Cloud)

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
# Gemini Config
GEMINI_API_KEY=your_gemini_api_key

# Qdrant Config
QDRANT_HOST=localhost
QDRANT_PORT=6333

# PostgreSQL Config
POSTGRES_USER=your postgres user
POSTGRES_PASSWORD=your pass
POSTGRES_HOST=localhost
POSTGRES_PORT=your port
POSTGRES_DB=qdrant

# Authentication
JWT_SECRET_KEY=your_secure_random_hash
JWT_REFRESH_SECRET_KEY=your_secure_refresh_hash
```

### 3. Backend Setup (reproducible with `uv.lock`)
1. Install [uv](https://docs.astral.sh/uv/) and Python 3.12 (see `.python-version`).
2. From the **project root**, install exact locked dependencies:
   ```bash
   uv sync
   ```
   Optional extras:
   ```bash
   uv sync --extra nlp      # root nlp_pipeline.py (spacy, langdetect, textstat)
   uv sync --extra scripts  # backend/scripts test helpers (requests, reportlab)
   ```
3. Copy env template and edit secrets:
   ```bash
   copy backend\.env.example backend\.env
   ```
4. Run database migrations:
   ```bash
   cd database
   uv run --project .. alembic upgrade head
   cd ..
   ```
5. Start the FastAPI server (run from `backend/`):
   ```bash
   cd backend
   uv run --directory .. uvicorn app.main:app --host 127.0.0.1 --port 8001
   ```
6. Start the embedding worker (second terminal, from project root):
   ```bash
   uv run python backend/run_embedding_worker.py
   ```

**Windows OCR (optional):** install [Tesseract](https://github.com/tesseract-ocr/tesseract) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases), then set `TESSERACT_CMD` and `POPPLER_PATH` in `backend/.env`.

### 4. Frontend Setup
1. Use Node.js 18+ (22 LTS recommended). From `frontend/`:
   ```bash
   cd frontend
   npm ci
   ```
2. Start the Vite dev server:
   ```bash
   npm run dev
   ```
