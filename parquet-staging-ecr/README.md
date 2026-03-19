# Parquet Staging Lambda

This container replaces the inline `reddit-parquet-staging-lambda` so the staging step can use `pyarrow` to merge multi-file trading days into a single `comments.parquet` and `submissions.parquet` before writing the day manifest.

## Build

Example build and push flow once Docker is available:

```bash
aws ecr create-repository --repository-name reddit-parquet-staging-ecr
aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 647371008162.dkr.ecr.us-west-1.amazonaws.com
docker build -t reddit-parquet-staging-ecr:latest parquet-staging-ecr
docker tag reddit-parquet-staging-ecr:latest 647371008162.dkr.ecr.us-west-1.amazonaws.com/reddit-parquet-staging-ecr:latest
docker push 647371008162.dkr.ecr.us-west-1.amazonaws.com/reddit-parquet-staging-ecr:latest
aws cloudformation deploy --stack-name reddit-parquet-staging-workflow --template-file infra/reddit-parquet-staging-workflow.yaml --capabilities CAPABILITY_IAM
```

If you use a different repository or tag, override the `StagingLambdaImageUri` CloudFormation parameter.
