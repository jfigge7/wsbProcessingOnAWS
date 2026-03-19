# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

ENV HF_HUB_DISABLE_TELEMETRY=1

# Copy requirements
COPY requirements.txt .

# Install non-torch dependencies, then install the CPU-only torch wheel.
RUN grep -v '^torch$' requirements.txt > requirements-no-torch.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --only-binary=:all: -r requirements-no-torch.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && rm -f requirements-no-torch.txt

# Copy function code and any baked model artifacts.
COPY model.py ${LAMBDA_TASK_ROOT}
COPY models /opt/models
ENV LOCAL_MODEL_ARTIFACT=/opt/models/reddit-risk-premium/model_latest.pt

# Precompile the handler module so cold starts do less work inside the Lambda sandbox.
RUN python -m compileall -q ${LAMBDA_TASK_ROOT}/model.py

# Default image handler (can be overridden per Lambda function ImageConfig)
CMD ["model.predict_handler"]
