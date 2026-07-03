import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# --- Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR  = PROJECT_ROOT / "backend"
load_dotenv(BACKEND_DIR / ".env", override=True)

def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        print("ERROR: QDRANT_URL is not configured in .env file.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Connecting to Qdrant at: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # We will ensure payload indexes are created on the unified collection
    # as well as the legacy/individual SOP collection to be robust.
    collections_to_index = [
        os.getenv("SEMANTIC_QDRANT_COLLECTION", "qa_semantic_chunks"),
        os.getenv("COLLECTION_SOPS", "docs_sops")
    ]
    
    fields_to_index = [
        "department",
        "metadata.department",
        "ref_number",
        "metadata.ref_number",
        "entity_type",
        "entity_id",
        "doc_type",
        "metadata.doc_type"
    ]
    
    for coll in collections_to_index:
        if not client.collection_exists(coll):
            print(f"Collection '{coll}' does not exist, skipping.")
            continue
            
        print(f"\nEnsuring payload indexes for collection: '{coll}'")
        for field in fields_to_index:
            try:
                print(f" - Creating index for field '{field}' of type KEYWORD...")
                client.create_payload_index(
                    collection_name=coll,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
                print(f"   [SUCCESS] Created or already exists.")
            except Exception as e:
                print(f"   [WARNING] Failed to create index for '{field}': {e}")
                
    print("\nQdrant payload index verification and creation complete!")

if __name__ == "__main__":
    main()
