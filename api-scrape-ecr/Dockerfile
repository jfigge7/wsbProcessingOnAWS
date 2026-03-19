FROM public.ecr.aws/lambda/python:3.14

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy function code
COPY scraper.py ${LAMBDA_TASK_ROOT}/scraper.py

# (inside your Dockerfile)
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/local/bin/aws-lambda-rie
RUN chmod +x /usr/local/bin/aws-lambda-rie

# Set the Lambda handler
CMD ["scraper.handler"]
