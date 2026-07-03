import io
import os
import sys
import traceback
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.ai_routes import _run_dynamic_ai_action, _resolve_explicit_style_override  # noqa: E402
from app.schemas import AIActionRequest  # noqa: E402


payload = AIActionRequest(
    action="rewrite",
    text="1. Zweck\nDies ist ein Testtext.",
    sop_title="SOP-IT-002",
    section_name="Zweck",
    section_type="Selected Text",
    edit_scope="section_only",
    sop_entity_id="4ae034db-6348-4583-b9ba-dfd401cda565",
    triggered_by="debug_style_override_runtime",
    instruction="Rewrite SOP-IT-002 in german_sop2 style",
    learn_to_profile=True,
)

try:
    print("STYLE OVERRIDE:", _resolve_explicit_style_override(payload.instruction))
    result = _run_dynamic_ai_action(payload, "rewrite")
    print("OK")
    print(result.model_dump())
except Exception as exc:
    print("ERROR")
    print(type(exc).__name__, str(exc))
    traceback.print_exc()
