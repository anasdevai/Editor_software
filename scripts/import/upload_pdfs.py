import os
import sys
import json
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models import SOP, SOPVersion, KnowledgeChunk, SourceReference, AIActionLog

client = TestClient(app)

pdfs = ["german_sop.pdf", "german_sop2.pdf", "german_sop3.pdf"]
sop_dir = PROJECT_ROOT / "SOP"

results = []

for pdf_name in pdfs:
    pdf_path = sop_dir / pdf_name
    result = {
        "file": pdf_name,
        "upload_status": "fail",
        "sop_id": None,
        "sop_number": None,
        "title": None,
        "version_id": None,
        "chunks_created": 0,
        "source_references_created": 0,
        "issues": []
    }
    
    if not pdf_path.exists():
        result["issues"].append(f"{pdf_name} not found")
        results.append(result)
        continue
    
    print(f"Uploading {pdf_name}...", file=sys.stderr)
    # 1. Extract text
    with open(pdf_path, "rb") as f:
        res = client.post("/api/extract-text", files={"file": (pdf_name, f, "application/pdf")})
        
    if res.status_code != 200:
        result["issues"].append(f"extract-text failed: {res.text}")
        results.append(result)
        continue
        
    extract_data = res.json()
    metadata = extract_data.get("sop_metadata_ui", {})
    doc_json = extract_data.get("structured_document", {"type": "doc", "content": []})
    
    # Check if docId is missing and inject a dummy one
    doc_id = metadata.get("sopMetadata", {}).get("documentId")
    if not doc_id:
        doc_id = f"SOP-{int(time.time())}"
        if "sopMetadata" not in metadata:
            metadata["sopMetadata"] = {}
        metadata["sopMetadata"]["documentId"] = doc_id
        
    title = metadata.get("sopMetadata", {}).get("title") or pdf_name
    
    print(f"Creating doc {doc_id} / {title}...", file=sys.stderr)
    # 2. Create document
    payload = {
        "title": title,
        "doc_json": doc_json,
        "metadata_json": metadata
    }
    
    res_create = client.post("/api/editor/docs", json=payload)
    if res_create.status_code != 200:
        result["issues"].append(f"create document failed: {res_create.text}")
        results.append(result)
        continue
        
    doc_data = res_create.json()
    result["upload_status"] = "success"
    result["sop_id"] = doc_data["id"]
    result["sop_number"] = doc_data["metadata_json"]["sopMetadata"]["documentId"]
    result["title"] = doc_data["title"]
    result["version_id"] = doc_data["current_version_id"]
    
    db = SessionLocal()
    try:
        chunks = db.query(KnowledgeChunk).filter(KnowledgeChunk.sop_id == result["sop_id"]).count()
        refs = db.query(SourceReference).filter(SourceReference.sop_id == result["sop_id"]).count()
        result["chunks_created"] = chunks
        result["source_references_created"] = refs
    except Exception as e:
        result["issues"].append(f"DB check failed: {e}")
    finally:
        db.close()
        
    results.append(result)

print(json.dumps(results, indent=2))
