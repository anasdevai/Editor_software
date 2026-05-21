"""Compatibility shim for `uvicorn app.main:app` from the repository root.

The actual FastAPI application lives at `backend/app/main.py`, but several
local scripts and docs still refer to `app.main:app`. This shim makes that
entrypoint work from the project root by ensuring `backend/` is on `sys.path`
before importing the real application object.
"""

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

backend_path = str(BACKEND_DIR)
root_path = str(ROOT_DIR)

if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from backend.app.main import app  # noqa: E402

