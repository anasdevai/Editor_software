FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for layer caching.
COPY backend/requirements.txt .

# Single RUN: runtime OS deps + build deps → pip → purge compilers (smaller layer, less disk).
# Do NOT install libtesseract-dev (pulls huge static libs like libtesseract.a). pytesseract only needs tesseract-ocr.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    tesseract-ocr \
    poppler-utils \
    curl \
    && pip install --no-cache-dir -r requirements.txt alembic asyncpg \
    && apt-get purge -y build-essential libpq-dev \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

COPY backend /app
COPY database /app/database

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
