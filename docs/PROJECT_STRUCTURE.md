# Project Structure

This repository is organized so production code, development utilities, and generated artifacts are easy to separate.

## Runtime Application

- `backend/app/` - FastAPI application, routes, database models, services, and API schemas.
- `backend/chatbot/` - Sidebar chatbot, SOP action prompts, RAG chain, context intelligence, and assistant behavior.
- `backend/retrieval/`, `backend/embeddings/`, `backend/ingestion/` - RAG ingestion, embeddings, and retrieval infrastructure.
- `frontend/src/` - React editor, sidebar, SOP upload/open flows, editor targeting, and API clients.
- `database/` - Database migration/configuration assets.
- `nlp_pipeline.py` - SOP NLP/profile extraction pipeline.

## Backend Agent Workflow

- `backend/app/agent_routes.py` - Backend-only agent endpoints under `/api/agents`.
- `backend/app/services/agent_orchestrator.py` - Consultant workflow orchestration, DeepAgents integration, multi-SOP template learning, generation preview, and style-only cross-profile rewriting.

The editor target resolver remains deterministic on the frontend. Agents can understand and draft, but they should not decide editor positions or directly write database changes without backend validation.

## Scripts

- `scripts/runtime/` - Runtime helpers used during local operation.
- `scripts/maintenance/` - Database and operational maintenance scripts.
- `scripts/import/` - SOP import/upload helpers.
- `scripts/verification/` - Manual verification, regression, and debugging scripts.
- `scripts/deployment/` - Deployment helpers.

## Tests

- `backend/tests/` - Backend regression and workflow tests.
- `scripts/verification/` - Manual or scenario-based checks that are not part of the normal app runtime.

Generated test reports, debug text files, logs, caches, and build output should not be committed.

## Generated / Ignored Artifacts

The following are treated as generated local artifacts:

- `frontend/dist/`
- `.pytest_cache/`
- `.uv-cache/`
- `__pycache__/`
- `backend_*.log`
- `backend_*.err`
- `backend_*.err.log`
- `verify_out/`
- `verify_out.txt`

## Dependency Folders

- `.venv/`, `frontend/node_modules/`, and `.hf-cache/` are local dependency/model caches. They are not production source code, but they are useful for local development and offline model execution, so they should not be deleted during normal cleanup.
