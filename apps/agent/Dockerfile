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

# ── Runtime ─────────────────────────────────────────────────────
FROM base AS runtime

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000 8001

# Default: start both Chainlit and FastAPI
CMD ["sh", "-c", \
     "chainlit run agent/ui/chainlit_app.py --host 0.0.0.0 --port 8000 & \
      uvicorn agent.api.main:app --host 0.0.0.0 --port 8001"]
