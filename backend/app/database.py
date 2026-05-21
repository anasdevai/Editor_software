from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

app_dir = Path(__file__).resolve().parent
backend_dir = app_dir.parent
project_dir = backend_dir.parent

# Prefer the backend-local env file, but still support a repo-root .env.
for env_path in (backend_dir / ".env", project_dir / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_LOCAL = os.getenv("DATABASE_URL_LOCAL")
DATABASE_URL_DOCKER = os.getenv("DATABASE_URL_DOCKER")

# Determine environment: Docker vs Local
IS_DOCKER = os.path.exists('/.dockerenv') or os.getenv("IS_DOCKER", "false").lower() == "true"

if IS_DOCKER:
    DATABASE_URL = DATABASE_URL_DOCKER or DATABASE_URL
else:
    DATABASE_URL = DATABASE_URL_LOCAL or DATABASE_URL

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found")

engine = create_engine(DATABASE_URL, echo=False, pool_size=20, max_overflow=20, pool_timeout=30)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
