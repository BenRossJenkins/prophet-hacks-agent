# syntax=docker/dockerfile:1.7
# Python base image — slim is enough; agent has no native deps beyond cryptography.
FROM python:3.11-slim AS runtime

# uvicorn workers and httpx benefit from this for cleaner stdout buffering.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time deps for cryptography / lxml / yfinance native wheels. We use
# binary wheels where available, but the `cryptography` rebuild path needs
# build-essential as a fallback on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching). Copying just pyproject.toml
# means we don't bust the deps layer on every code change.
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install \
        "ai-prophet-core>=0.1.3" \
        "ai-prophet>=0.1.0" \
        "anthropic>=0.40" \
        "openai>=1.40" \
        "google-genai>=1.0" \
        "google-cloud-storage>=2.18" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "pydantic>=2.8" \
        "python-dotenv>=1.0" \
        "requests>=2.32" \
        "yfinance>=0.2"

# Application code
COPY agent/ ./agent/

# Cloud Run injects PORT (default 8080). We bind 0.0.0.0 + that port.
ENV PORT=8080
EXPOSE 8080

# Single uvicorn worker — the LLM ensemble already uses ThreadPoolExecutor
# internally, so adding worker processes just multiplies memory without
# improving the latency of any single request.
CMD ["sh", "-c", "uvicorn agent.predict:app --host 0.0.0.0 --port ${PORT}"]
