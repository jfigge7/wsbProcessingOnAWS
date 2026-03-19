# Reddit Data Project Containers

This workspace contains the local code for a multi-stage AWS pipeline that stages Reddit parquet data, processes WallStreetBets comments into model-ready observations, and trains/predicts a risk-premium model.

## What This Repository Contains

The codebase is organized around the major pipeline stages:

- `parquet-staging-ecr/`
  - Containerized staging Lambda.
  - Merges per-day parquet fragments, drops orphan days after a grace period, writes day manifests, and emits batch-ready `monthly_ready.json` files for downstream processing.
- `data-processing-ecr/`
  - Original processing implementation used by the processor runtime.
  - Converts staged Reddit comments/submissions into observation parquet files.
- `data-processing-fargate-ecr/`
  - ECS/Fargate wrapper for the processing stage.
  - Reuses the processing code without the Lambda runtime so long-running processing can happen on Fargate.
- `model-ecr/`
  - Model training and prediction Lambda code.
  - Builds targets from price history, trains the model, writes metrics/model artifacts, and produces predictions/signals.
- `api-scrape-ecr/`
  - Scraper container code.
- `infra/`
  - CloudFormation templates for the staging and processing/model workflows.
- `scripts/`
  - Utility scripts for local data prep and maintenance.
- `test-ecr/`
  - Small test container app.

## Current Pipeline Shape

At a high level, the AWS workflow looks like this:

1. Parquet files are uploaded to the staging bucket.
2. The staging container Lambda normalizes trading-day data and writes manifests into the raw bucket.
3. `monthly_ready.json` trigger files enter the processing workflow.
4. Step Functions launches processing tasks on ECS/Fargate.
5. Processed observation parquet files are enqueued for model work.
6. The model queue worker batches processed files and invokes:
   - `reddit-model-container-lambda` for training
   - `reddit-predict-model-container-lambda` for prediction
7. The latest model artifact and model outputs are written to the model output bucket.

## Key Local Entry Points

- Staging logic: [staging_lambda.py](/workspaces/Reddit%20Data%20Project/Containers/parquet-staging-ecr/staging_lambda.py)
- Processing logic: [preprocessing_lambda.py](/workspaces/Reddit%20Data%20Project/Containers/data-processing-ecr/preprocessing_lambda.py)
- Fargate processor entrypoint: [processor_task.py](/workspaces/Reddit%20Data%20Project/Containers/data-processing-fargate-ecr/processor_task.py)
- Model training/prediction: [model.py](/workspaces/Reddit%20Data%20Project/Containers/model-ecr/model.py)
- Staging infra: [reddit-parquet-staging-workflow.yaml](/workspaces/Reddit%20Data%20Project/Containers/infra/reddit-parquet-staging-workflow.yaml)
- Processing/model infra: [reddit-processing-approval-workflow.yaml](/workspaces/Reddit%20Data%20Project/Containers/infra/reddit-processing-approval-workflow.yaml)

## Notes About the Current Design

- Processing currently runs on ECS/Fargate, not Lambda.
- The model worker currently batches processed files before invoking training/prediction.
- The latest trained model is stored in S3 and should be preferred over any baked model artifact.
- Several subfolders already include their own README files for image-specific build steps.

## Existing Image-Specific READMEs

- [parquet-staging-ecr/README.md](/workspaces/Reddit%20Data%20Project/Containers/parquet-staging-ecr/README.md)
- [data-processing-fargate-ecr/README.md](/workspaces/Reddit%20Data%20Project/Containers/data-processing-fargate-ecr/README.md)

## Local Development

This repository is being worked on inside a dev container. Common tasks in this workspace are:

- editing Lambda or ECS container code
- validating CloudFormation templates
- rebuilding and pushing container images to ECR
- inspecting live AWS workflow behavior from the terminal

## Caveats

- This repository reflects the local codebase, not necessarily every currently deployed AWS revision.
- Some runtime behavior is controlled by Lambda environment variables and live AWS resource configuration.
- Large parts of the pipeline are stateful because they depend on S3 buckets, queues, Step Functions executions, and saved model artifacts.
