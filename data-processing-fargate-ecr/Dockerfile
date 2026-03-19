FROM python:3.11-slim

ARG PREFETCH_HF_MODELS=1
ARG FINBERT_MODEL_NAME=ProsusAI/finbert
ARG TWITTER_ROBERTA_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FINBERT_MODEL_NAME=${FINBERT_MODEL_NAME} \
    TWITTER_ROBERTA_MODEL_NAME=${TWITTER_ROBERTA_MODEL_NAME} \
    FINBERT_MODEL_DIR=/opt/models/finbert \
    TWITTER_ROBERTA_MODEL_DIR=/opt/models/twitter_roberta_sentiment \
    MODEL_CACHE_DIR=/opt/model-cache \
    HF_HOME=/opt/model-cache/huggingface \
    TRANSFORMERS_CACHE=/opt/model-cache/huggingface/transformers \
    HF_HUB_CACHE=/opt/model-cache/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    LAMBDA_TASK_ROOT=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY data-processing-ecr/requirements.txt ./requirements.txt

RUN grep -v '^torch$' requirements.txt > requirements-no-torch.txt \
    && pip install --upgrade pip \
    && pip install --only-binary=:all: -r requirements-no-torch.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && rm -f requirements-no-torch.txt

RUN mkdir -p "${FINBERT_MODEL_DIR}" "${TWITTER_ROBERTA_MODEL_DIR}" /opt/data

# Allow Hugging Face downloads during the image build so the model assets are baked in.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; pairs = [(os.environ['FINBERT_MODEL_NAME'], os.environ['FINBERT_MODEL_DIR']), (os.environ['TWITTER_ROBERTA_MODEL_NAME'], os.environ['TWITTER_ROBERTA_MODEL_DIR'])]; exec('for model_name, target_dir in pairs:\n    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)\n    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)\n    tok.save_pretrained(target_dir)\n    mdl.save_pretrained(target_dir, safe_serialization=True)')"

COPY data-processing-ecr/data/merged.parquet /opt/data/merged.parquet
COPY data-processing-ecr/preprocessing_lambda.py /app/preprocessing_lambda.py
COPY data-processing-fargate-ecr/processor_task.py /app/processor_task.py

RUN python -m compileall -q /app/preprocessing_lambda.py /app/processor_task.py \
    && chmod -R a+rX /app /opt/models /opt/data /opt/model-cache

# Force offline loading at runtime so ECS uses the baked model assets.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "/app/processor_task.py"]
