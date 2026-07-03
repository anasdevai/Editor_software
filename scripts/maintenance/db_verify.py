"""
Direct DB verification script for production QA report.
Run from project root: .venv\Scripts\python.exe scripts\db_verify.py
"""
import os, sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR  = PROJECT_ROOT / "backend"
for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.database import SessionLocal
from app.models import (SOP, SOPVersion, AIActionLog, ChatSession, ChatMessage,
                        ClientProfile, ProfileVersion, ProfileHistoryEvent, ProfileSuggestion,
                        SOPDetectedParameters)

db = SessionLocal()

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# 1. SOPs
section("1. SOPs TABLE")
sops = db.query(SOP).all()
print(f"  Row count: {len(sops)}")
for s in sops:
    print(f"  {s.sop_number} | id={s.id} | active={s.is_active} | cur_ver={s.current_version_id}")

# 2. SOP Versions for SOP-IT-003
section("2. SOPVersions for SOP-IT-003")
sop3 = db.query(SOP).filter(SOP.sop_number == "SOP-IT-003").first()
if sop3:
    vers = db.query(SOPVersion).filter(SOPVersion.sop_id == sop3.id).order_by(SOPVersion.created_at).all()
    print(f"  Total versions: {len(vers)}")
    for v in vers:
        meta = v.metadata_json or {}
        trail = meta.get("auditTrail", [])
        print(f"  v{v.version_number} | id={v.id} | audit_entries={len(trail)}")
else:
    print("  SOP-IT-003 not found")

# 3. AI Action Logs
section("3. AIActionLogs (latest 10)")
logs = db.query(AIActionLog).order_by(AIActionLog.created_at.desc()).limit(10).all()
all_count = db.query(AIActionLog).count()
print(f"  Total rows: {all_count}  |  Showing: {len(logs)}")
for l in logs:
    orig = (l.original_text or "")[:60]
    sugg = (l.suggested_text or "")[:60]
    print(f"  action={l.action} | id={l.id} | orig='{orig}...' | sugg='{sugg}...'")

# 4. Chat Sessions & Messages
section("4. ChatSessions & ChatMessages")
sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).limit(5).all()
total_sessions = db.query(ChatSession).count()
total_msgs = db.query(ChatMessage).count()
print(f"  Total sessions: {total_sessions} | Total messages: {total_msgs}")
for s in sessions:
    msgs = db.query(ChatMessage).filter(ChatMessage.session_id == s.id).count()
    print(f"  session={s.id} | messages={msgs}")

# 5. Client Profiles
section("5. ClientProfiles")
profiles = db.query(ClientProfile).all()
print(f"  Total profiles: {len(profiles)}")
for p in profiles:
    print(f"  name={p.name} | id={p.id} | sops_analyzed={p.total_sops_analyzed}")
    if p.active_profile_md:
        print(f"  profile_md_length={len(p.active_profile_md)} chars")
        print(f"  --- profile_md preview (first 400 chars) ---")
        print(p.active_profile_md[:400])
    rules = (p.active_profile_json or {}).get("rewrite_rules", [])
    print(f"  active rewrite_rules count: {len(rules)}")
    for r in rules:
        print(f"    - {r}")

# 6. Profile Versions
section("6. ProfileVersions (latest 5)")
if profiles:
    pvs = (db.query(ProfileVersion)
           .filter(ProfileVersion.profile_id == profiles[0].id)
           .order_by(ProfileVersion.version_number.desc())
           .limit(5).all())
    total_pvs = db.query(ProfileVersion).filter(ProfileVersion.profile_id == profiles[0].id).count()
    print(f"  Total profile versions: {total_pvs}  |  Showing latest 5:")
    for pv in pvs:
        reason = (pv.change_reason or "")[:80]
        print(f"  v{pv.version_number} | id={pv.id} | reason='{reason}'")

# 7. Profile Suggestions summary
section("7. ProfileSuggestions")
total_suggs = db.query(ProfileSuggestion).count()
accepted = db.query(ProfileSuggestion).filter(ProfileSuggestion.status == "accepted").count()
rejected = db.query(ProfileSuggestion).filter(ProfileSuggestion.status == "rejected").count()
pending  = db.query(ProfileSuggestion).filter(ProfileSuggestion.status == "pending").count()
print(f"  Total: {total_suggs} | accepted={accepted} | rejected={rejected} | pending={pending}")

# 8. Profile History Events
section("8. ProfileHistoryEvents (latest 5)")
if profiles:
    evs = (db.query(ProfileHistoryEvent)
           .filter(ProfileHistoryEvent.client_profile_id == profiles[0].id)
           .order_by(ProfileHistoryEvent.created_at.desc())
           .limit(5).all())
    total_evs = db.query(ProfileHistoryEvent).filter(ProfileHistoryEvent.client_profile_id == profiles[0].id).count()
    print(f"  Total events: {total_evs}  |  Showing latest 5:")
    for ev in evs:
        print(f"  type={ev.event_type} | {ev.event_summary[:90]}")

# 9. SOPDetectedParameters
section("9. SOPDetectedParameters")
all_dp = db.query(SOPDetectedParameters).all()
print(f"  Total rows: {len(all_dp)}")
for dp in all_dp:
    style = dp.writing_style or {}
    roles = dp.roles_raci or {}
    terms = dp.terminology or {}
    print(f"  sop_id={dp.sop_id} | client={dp.client_name} | file={dp.source_filename}")
    print(f"    writing_style keys: {list(style.keys())}")
    print(f"    roles_raci keys: {list(roles.keys())}")
    print(f"    terminology keys: {list(terms.keys())}")

db.close()
print(f"\n{'='*60}")
print("  DB VERIFICATION COMPLETE")
print(f"{'='*60}")
