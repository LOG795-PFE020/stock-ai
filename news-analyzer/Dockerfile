# Multi-stage build for better optimization
FROM python:3.10-slim AS base

# Set environment variables to be used across all stages
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stage 1: Build dependencies
FROM base AS dependencies

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /build

# Install PyTorch for CPU (compatible with ARM64)
RUN pip install --no-cache-dir torch==2.0.1

# Copy and install base requirements first (less likely to change)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Copy and install main requirements
# Remove asyncio package which is causing problems (it's part of Python std lib)
# Also remove uuid package which conflicts with built-in uuid module
# Remove torch as we've already installed it
COPY requirements.txt .
RUN sed -i '/asyncio==/d' requirements.txt && \
    sed -i '/uuid==/d' requirements.txt && \
    sed -i '/torch>=/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean pip cache to save space
    rm -rf /root/.cache/pip

# Stage 2: Download models (can be cached separately)
FROM dependencies AS model-cache

# Create the cache directory
RUN mkdir -p /root/.cache/huggingface

# Pre-download and cache the model data
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; \
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3); \
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')" && \
    # Clean up after downloading to save space
    find /root/.cache/huggingface -name "*.h5" -type f -delete

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Final image
FROM base AS final

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a non-root user for running the application
RUN useradd -m -U app_user && \
    chown -R app_user:app_user /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R app_user:app_user /app/models /app/data /app/logs && \
    chmod -R 755 /app/models /app/data /app/logs

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local /usr/local

# Copy pre-downloaded models and data
COPY --from=model-cache /root/.cache/huggingface /home/app_user/.cache/huggingface
COPY --from=model-cache /root/nltk_data /home/app_user/nltk_data
RUN chown -R app_user:app_user /home/app_user/.cache /home/app_user/nltk_data

# Copy application code (only what's needed)
COPY stock_news_analyzer.py deepcrawler.py group_crawler.py np3k_group_crawler.py news_publisher.py ./
RUN chown -R app_user:app_user /app

# Set environment variables
ENV PYTHONPATH=/app \
    # RabbitMQ configuration
    RABBITMQ_HOST=rabbitmq \
    RABBITMQ_PORT=5672 \
    RABBITMQ_USER=guest \
    RABBITMQ_PASS=guest \
    # API configuration
    API_HOST=0.0.0.0 \
    API_PORT=8092 \
    # PyTorch MPS configuration (for Apple Silicon)
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    # Redis configuration
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    YF_RATE_LIMIT=2 \
    YF_RATE_PERIOD=60 \
    # Ensure Hugging Face cache is accessible
    TRANSFORMERS_CACHE=/home/app_user/.cache/huggingface \
    HF_HOME=/home/app_user/.cache/huggingface \
    NLTK_DATA=/home/app_user/nltk_data

# Make port 8092 available
EXPOSE 8092

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8092/api/health || exit 1

# Start Redis and the application
CMD ["sh", "-c", "service redis-server start && cd /app && su app_user -c 'python stock_news_analyzer.py'"]