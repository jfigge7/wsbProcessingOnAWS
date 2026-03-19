# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

ARG PREFETCH_HF_MODELS=1
ARG FINBERT_MODEL_NAME=ProsusAI/finbert
ARG TWITTER_ROBERTA_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest

ENV FINBERT_MODEL_NAME=${FINBERT_MODEL_NAME} \
    TWITTER_ROBERTA_MODEL_NAME=${TWITTER_ROBERTA_MODEL_NAME} \
    FINBERT_MODEL_DIR=/opt/models/finbert \
    TWITTER_ROBERTA_MODEL_DIR=/opt/models/twitter_roberta_sentiment \
    MODEL_CACHE_DIR=/tmp/model-cache \
    HF_HOME=/tmp/model-cache/huggingface \
    TRANSFORMERS_CACHE=/tmp/model-cache/huggingface/transformers \
    HF_HUB_CACHE=/tmp/model-cache/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1

# Copy requirements
COPY requirements.txt .

# Install non-torch dependencies, then install the CPU-only torch wheel.
RUN grep -v '^torch$' requirements.txt > requirements-no-torch.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --only-binary=:all: -r requirements-no-torch.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && rm -f requirements-no-torch.txt

# Bake the text-classification assets into the image when enabled.
RUN mkdir -p "${FINBERT_MODEL_DIR}" "${TWITTER_ROBERTA_MODEL_DIR}" \
    && if [ "${PREFETCH_HF_MODELS}" = "1" ]; then \
         python -c "import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; pairs = [(os.environ['FINBERT_MODEL_NAME'], os.environ['FINBERT_MODEL_DIR']), (os.environ['TWITTER_ROBERTA_MODEL_NAME'], os.environ['TWITTER_ROBERTA_MODEL_DIR'])]; exec(\"for model_name, target_dir in pairs:\\n    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)\\n    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)\\n    tok.save_pretrained(target_dir)\\n    mdl.save_pretrained(target_dir, safe_serialization=True)\")"; \
       fi \
    && rm -rf "${MODEL_CACHE_DIR}"

# Copy parquet dataset into image
COPY data/merged.parquet /opt/data/merged.parquet

# Copy function code
COPY preprocessing_lambda.py ${LAMBDA_TASK_ROOT}

# Precompile the handler module so cold starts do less work inside the Lambda sandbox.
RUN python -m compileall -q ${LAMBDA_TASK_ROOT}/preprocessing_lambda.py

# Set the CMD to your handler
CMD ["preprocessing_lambda.lambda_handler"]
