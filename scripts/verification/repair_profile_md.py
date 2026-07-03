"""
repair_profile_md.py
--------------------
Two-in-one repair:
  1. Re-splits any merged profiles — creates one ClientProfile per SOP
     from existing SOPDetectedParameters rows.
  2. Regenerates active_profile_md for any profile that has
     active_profile_json but a blank active_profile_md.

Run from the project root:
    python scripts/repair_profile_md.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import nlp_pipeline  # noqa: E402

from backend.app.database import SessionLocal   # noqa: E402
from backend.app.models import ClientProfile, SOPDetectedParameters, SOP  # noqa: E402


def repair():
    db = SessionLocal()
    created = 0
    repaired = 0

    try:
        # ── Step 1: ensure every SOP has its own ClientProfile ─────────────
        all_params = (
            db.query(SOPDetectedParameters)
            .order_by(SOPDetectedParameters.created_at.asc())
            .all()
        )

        # Group by sop_id — keep only the latest param per SOP
        latest_per_sop = {}
        for p in all_params:
            latest_per_sop[str(p.sop_id)] = p

        print(f"Found {len(latest_per_sop)} distinct SOP(s) in SOPDetectedParameters.")

        for sop_uuid, param in latest_per_sop.items():
            sop = db.query(SOP).filter(SOP.id == param.sop_id).first()
            tenant_id = sop.tenant_id if sop else param.sop_id  # fallback

            # Build profile name from SOP metadata
            if sop and sop.sop_number:
                profile_name = f"{sop.sop_number} - {sop.title or param.source_filename or str(param.sop_id)}"
            elif sop and sop.title:
                profile_name = sop.title
            else:
                profile_name = param.source_filename or f"SOP {param.sop_id}"

            client_name = param.client_name or (sop.title if sop else "Client") or "Client"

            # Check if this SOP already has a properly linked profile
            if param.client_profile_id:
                linked = db.query(ClientProfile).filter(ClientProfile.id == param.client_profile_id).first()
                if linked:
                    # Check no other SOP owns this profile
                    other_sop_param = (
                        db.query(SOPDetectedParameters)
                        .filter(
                            SOPDetectedParameters.client_profile_id == linked.id,
                            SOPDetectedParameters.sop_id != param.sop_id,
                        )
                        .first()
                    )
                    if not other_sop_param:
                        print(f"  OK  : {profile_name!r} already has its own profile")
                        continue
                    else:
                        print(f"  FIX : {profile_name!r} shares a profile with another SOP — creating separate one")
                        # Fall through to create a new profile
            else:
                print(f"  NEW : {profile_name!r} has no profile yet — creating")

            # Get the analysis json from the param to build the profile
            analysis_json = param.analysis_json or {}
            built_profile_json = analysis_json.get("client_profile") or {}

            # Build profile_md
            built_profile_md = analysis_json.get("profile_md") or ""
            if not built_profile_md and built_profile_json:
                try:
                    built_profile_md = nlp_pipeline.generate_profile_md(built_profile_json)
                except Exception as e:
                    print(f"    Warning: generate_profile_md failed: {e}")
                    built_profile_md = f"# {client_name} SOP Profile\n\nProfile for: {profile_name}\n"

            if not built_profile_md:
                built_profile_md = f"# {client_name} SOP Profile\n\nProfile for: {profile_name}\n"

            # Create a fresh ClientProfile for this SOP
            new_profile = ClientProfile(
                tenant_id=tenant_id,
                name=profile_name,
                company_name=client_name,
                total_sops_analyzed=1,
                active_profile_json=built_profile_json or analysis_json,
                active_profile_md=built_profile_md,
            )
            db.add(new_profile)
            db.flush()

            # Link this param (and all params for this sop_id) to the new profile
            db.query(SOPDetectedParameters).filter(
                SOPDetectedParameters.sop_id == param.sop_id
            ).update({"client_profile_id": new_profile.id})

            print(f"    Created profile id={new_profile.id}, md={len(built_profile_md)} chars")
            created += 1

        # ── Step 2: regenerate profile_md for any profile still missing it ──
        profiles = db.query(ClientProfile).all()
        for profile in profiles:
            if not profile.active_profile_md and profile.active_profile_json:
                try:
                    md = nlp_pipeline.generate_profile_md(profile.active_profile_json)
                    profile.active_profile_md = md
                    repaired += 1
                    print(f"  Repaired md: {profile.name!r} — {len(md)} chars")
                except Exception as e:
                    print(f"  ERROR regenerating md for {profile.name!r}: {e}")

        db.commit()
        print(f"\nDone. Created {created} new profile(s). Repaired {repaired} profile(s).")

    except Exception as ex:
        db.rollback()
        print(f"ERROR: {ex}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    repair()
