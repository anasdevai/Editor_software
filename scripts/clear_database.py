"""
clear_database.py
-----------------
Truncates ALL tables in the editor_db PostgreSQL database.
Run from the project root:
    python scripts/clear_database.py
"""

import sys
import os
from pathlib import Path

# ── path setup so we can import backend app modules ──────────────────────────
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv(backend_dir / ".env", override=True)

from sqlalchemy import text, create_engine

DATABASE_URL = os.getenv("DATABASE_URL_LOCAL") or os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("[ERROR] DATABASE_URL not found. Check backend/.env")
    sys.exit(1)

engine = create_engine(DATABASE_URL, echo=False)

# Tables ordered so that child tables (FK dependents) are truncated first.
TABLES = [
    # link / junction tables first
    "sop_deviation_links",
    "deviation_capa_links",
    "capa_audit_links",
    "audit_decision_links",
    "decision_sop_links",

    # leaf / child tables
    "chat_messages",
    "ai_suggestions",
    "ai_link_suggestions",
    "ai_action_logs",
    "embedding_jobs",
    "knowledge_chunks",
    "source_references",
    "profile_detections",
    "sop_detected_parameters",
    "profile_suggestions",
    "profile_history_events",
    "profile_audit_logs",
    "profile_versions",

    # mid-level tables
    "sop_versions",
    "chat_sessions",
    "client_profiles",

    # top-level tables
    "sops",
    "deviations",
    "capas",
    "audit_findings",
    "decisions",
    "lifecycle_configs",

    # users last (other tables SET NULL on user FK)
    "users",
]

print("[WARNING] This will DELETE ALL DATA from the database.")
print("   Database:", DATABASE_URL.split('@')[-1])
confirm = input("Type 'yes' to confirm: ").strip().lower()

if confirm != "yes":
    print("Aborted.")
    sys.exit(0)

print("\n[INFO] Clearing all tables ...\n")

with engine.begin() as conn:
    # Disable FK checks temporarily so we can truncate in any order
    conn.execute(text("SET session_replication_role = 'replica';"))
    for table in TABLES:
        try:
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
            print(f"   [OK]  {table}")
        except Exception as e:
            print(f"   [SKIP] {table} -- {e}")
    # Re-enable FK checks
    conn.execute(text("SET session_replication_role = 'origin';"))

print("\n[DONE] All tables cleared. Database is now empty.\n")
