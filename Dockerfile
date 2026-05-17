# ──────────────────────────────────────────────────────────────────
# Pymerag API — Multi-stage Docker build
# ──────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════
# Stage 1: Builder — compile and cache dependencies
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency specification first for layer caching
COPY pyproject.toml ./

# Create a minimal app package so uv can resolve the project
RUN mkdir -p app && touch app/__init__.py

# Install all production dependencies into the system Python
RUN uv pip install --system --no-cache .


# ══════════════════════════════════════════════════════════════════
# Stage 2: Runtime — lean production image
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim

WORKDIR /app

# Copy pre-built site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Create data directory for volume mounting
RUN mkdir -p /app/data/uploads && chmod 755 /app/data

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/admin/health').raise_for_status()" || exit 1

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
