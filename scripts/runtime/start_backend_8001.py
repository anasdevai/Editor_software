from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
log = ROOT / "backend_8001_live.log"
err = ROOT / "backend_8001_live.err"

with log.open("wb") as stdout, err.open("wb") as stderr:
    env = os.environ.copy()
    site_packages = ROOT / ".venv" / "Lib" / "site-packages"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "backend"), str(site_packages), env.get("PYTHONPATH", "")]
    )
    env["VIRTUAL_ENV"] = str(ROOT / ".venv")
    proc = subprocess.Popen(
        [
            getattr(sys, "_base_executable", None) or sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--workers",
            "1",
        ],
        cwd=str(BACKEND),
        env=env,
        stdout=stdout,
        stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

print(proc.pid)
