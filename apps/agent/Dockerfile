# ── Build stage ────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for psycopg2 and yfinance
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set PYTHONPATH so `from agent.xxx import ...` resolves correctly
ENV PYTHONPATH=/app

# ── Runtime ─────────────────────────────────────────────────────
FROM base AS runtime

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

# Railway injects PORT automatically; fall back to 8001 for local runs
EXPOSE ${PORT:-8001}

# Production: FastAPI only (Chainlit is for local dev only)
# Railway sets $PORT; local dev defaults to 8001
CMD ["sh", "-c", "uvicorn agent.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
