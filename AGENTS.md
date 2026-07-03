# Repository Guidelines

## Project Structure & Module Organization

The FastAPI application lives in `backend/app/`; chatbot orchestration is under `backend/chatbot/`, while `backend/retrieval/`, `backend/embeddings/`, and `backend/ingestion/` contain the RAG pipeline. The React/Vite client is in `frontend/src/`, organized into pages, components, API clients, hooks, and editor utilities. Database models and Alembic migrations are in `database/`. Keep operational helpers grouped under `scripts/runtime/`, `scripts/import/`, `scripts/maintenance/`, `scripts/verification/`, or `scripts/deployment/`. Architecture notes belong in `docs/`. Do not commit generated logs, caches, `frontend/dist/`, or dependency directories.

## Build, Test, and Development Commands

- `uv sync --extra nlp --extra scripts` installs the locked Python environment and optional tooling.
- `cd database && uv run --project .. alembic upgrade head` applies database migrations.
- `uv run --directory backend uvicorn app.main:app --host 127.0.0.1 --port 8001` starts the API.
- `uv run python backend/run_embedding_worker.py` starts asynchronous embedding work.
- `cd frontend && npm ci && npm run dev` installs and starts the UI.
- `cd frontend && npm run build` creates a production bundle; `npm run lint` runs ESLint.
- `node scripts/verification/test_editor_target_resolver.mjs` runs the editor-target regression checks.

Docker users may run `docker compose up --build` for the integrated stack.

## Coding Style & Naming Conventions

Use four spaces in Python and two spaces in JavaScript/JSX. Prefer type hints and small service functions in Python. Use `snake_case` for Python functions/modules, `PascalCase` for React components and classes, and `camelCase` for JavaScript functions and variables. Keep editor mutations deterministic: AI may suggest a target, but live TipTap IDs and backend validation must authorize the exact range. Run ESLint and relevant regression scripts before submission.

## Testing Guidelines

Name Python tests `test_*.py` and JavaScript checks `test_*.mjs`. Put automated backend tests in `backend/tests/`; keep scenario and integration checks in `scripts/verification/`. Every bug fix should include a focused regression reproducing the failure. Tests involving PostgreSQL, Qdrant, or the local LLM must document required services and environment variables.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit prefixes such as `feat:` and `fix:`. Keep each commit scoped and imperative, for example `fix: reject stale editor target ids`. Pull requests should explain behavior changes, list verification commands, link the issue, call out migrations or configuration changes, and include screenshots for visible UI changes.

## Security & Configuration

Copy `.env.example` locally; never commit credentials, tokens, customer SOPs, or production exports. Preserve tenant filtering, authentication checks, audit metadata, and append-only history when changing stateful endpoints.
