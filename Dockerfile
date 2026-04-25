# NegotiArena — OpenEnv-compatible HuggingFace Space
# Deploys the FastAPI server for RL environment interaction

FROM python:3.11-slim

LABEL maintainer="NegotiArena Team"
LABEL description="Multi-agent negotiation environment for scalable oversight training"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first (Docker cache layer)
COPY pyproject.toml ./
COPY requirements.txt ./

# Install core dependencies (no GPU/training deps in server image)
RUN uv pip install --system --no-cache \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.6.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0 \
    httpx>=0.27.0 \
    rich>=13.7.0 \
    python-dotenv>=1.0.0

# Copy source
COPY negotiarena_env.py ./
COPY server/ ./server/
COPY training/prompts.py ./training/
COPY openenv.yaml ./
COPY README.md ./

# Create empty __init__ files
RUN touch server/__init__.py training/__init__.py

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 7860 (HF Spaces standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]