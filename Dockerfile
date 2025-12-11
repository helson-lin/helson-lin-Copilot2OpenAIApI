# GitHub Copilot OpenAI Adapter - Dockerfile
# Multi-stage build for smaller image size

# ============== Build Stage ==============
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ============== Production Stage ==============
FROM python:3.11-slim as production

# Labels
LABEL org.opencontainers.image.title="GitHub Copilot OpenAI Adapter"
LABEL org.opencontainers.image.description="Convert GitHub Copilot API to OpenAI-compatible format"
LABEL org.opencontainers.image.source="https://github.com/your-username/githubcp"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY server.py .

# Create directory for token storage
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables with defaults
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    HTTP_MAX_CONNECTIONS=100 \
    HTTP_MAX_KEEPALIVE=20 \
    HTTP_TIMEOUT=120 \
    MAX_CLIENTS_CACHE=1000 \
    RATE_LIMIT_ENABLED=true \
    RATE_LIMIT_REQUESTS=60 \
    RATE_LIMIT_WINDOW=60 \
    MODEL_CACHE_TTL=3600 \
    TOKEN_STORAGE_PATH=/app/data/.github_tokens.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()"

# Run with gunicorn in production
CMD ["sh", "-c", "if [ \"$WORKERS\" -gt 1 ]; then \
    gunicorn server:app -w $WORKERS -k uvicorn.workers.UvicornWorker --bind $HOST:$PORT; \
    else \
    uvicorn server:app --host $HOST --port $PORT; \
    fi"]
