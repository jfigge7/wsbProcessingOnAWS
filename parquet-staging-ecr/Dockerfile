FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY staging_lambda.py ${LAMBDA_TASK_ROOT}

CMD ["staging_lambda.lambda_handler"]