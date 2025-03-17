# Multi-stage build for better optimization
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stage 1: Build dependencies
FROM base AS dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install PyTorch for CPU
RUN pip install torch==2.0.1

# Copy and install base requirements
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

# Copy and install main requirements
COPY requirements.txt .
RUN sed -i '/asyncio==/d' requirements.txt && \
    sed -i '/uuid==/d' requirements.txt && \
    sed -i '/torch>=/d' requirements.txt && \
    pip install -r requirements.txt

# Stage 2: Download models
FROM dependencies AS model-cache

RUN mkdir -p /root/.cache/huggingface
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; \
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3); \
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')" && \
    find /root/.cache/huggingface -name "*.h5" -type f -delete
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Final image
FROM base AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -U app_user && \
    chown -R app_user:app_user /app

RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R app_user:app_user /app/models /app/data /app/logs && \
    chmod -R 755 /app/models /app/data /app/logs

COPY --from=dependencies /usr/local /usr/local
COPY --from=model-cache /root/.cache/huggingface /home/app_user/.cache/huggingface
COPY --from=model-cache /root/nltk_data /home/app_user/nltk_data
RUN chown -R app_user:app_user /home/app_user/.cache /home/app_user/nltk_data

COPY stock_news_analyzer.py deepcrawler.py group_crawler.py np3k_group_crawler.py news_publisher.py ./
RUN chown -R app_user:app_user /app

ENV PYTHONPATH=/app \
    RABBITMQ_HOST=rabbitmq \
    RABBITMQ_PORT=5672 \
    RABBITMQ_USER=guest \
    RABBITMQ_PASS=guest \
    API_HOST=0.0.0.0 \
    API_PORT=8092 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    YF_RATE_LIMIT=2 \
    YF_RATE_PERIOD=60 \
    TRANSFORMERS_CACHE=/home/app_user/.cache/huggingface \
    HF_HOME=/home/app_user/.cache/huggingface \
    NLTK_DATA=/home/app_user/nltk_data

EXPOSE 8092

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8092/api/health || exit 1

CMD ["sh", "-c", "service redis-server start && cd /app && su app_user -c 'python stock_news_analyzer.py'"]