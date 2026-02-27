# ── Single-stage production image ──────────────────────────────
# Python 3.12 matches pyproject.toml requires-python = ">=3.12"
FROM python:3.12-slim

WORKDIR /app

# System deps:
#   gcc + libpq-dev  → psycopg2-binary build headers
#   curl             → Railway health check probing during startup
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first — this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (requirements-dev.txt intentionally excluded from prod)
COPY . .

# PYTHONPATH lets `from agent.xxx import ...` resolve correctly from /app
ENV PYTHONPATH=/app

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

# Railway injects $PORT automatically; fall back to 8001 for local docker runs
EXPOSE ${PORT:-8001}

# Production entrypoint: FastAPI server
CMD ["sh", "-c", "uvicorn agent.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
