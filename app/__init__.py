"""Compatibility package for running the backend from the repo root.

The real package lives in ``backend/app``.  Extending ``__path__`` lets imports
such as ``app.models`` and ``app.database`` resolve even when Python first
loads this lightweight root-level compatibility package.
"""

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
BACKEND_APP_DIR = BACKEND_DIR / "app"

for path in (str(BACKEND_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

backend_app_path = str(BACKEND_APP_DIR)
if backend_app_path not in __path__:
    __path__.append(backend_app_path)
