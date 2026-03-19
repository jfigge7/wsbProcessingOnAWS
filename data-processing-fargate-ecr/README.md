Build this image with the repository root as the Docker build context, because the Dockerfile copies shared files from `data-processing-ecr/`.

The easiest option is the helper script in this folder:

```bash
./data-processing-fargate-ecr/build_image.sh reddit-processor-fargate:latest
```

You can also run Docker directly from the repository root:

```bash
docker build -f data-processing-fargate-ecr/Dockerfile -t reddit-processor-fargate:latest .
```

If you run `docker build` from inside `data-processing-fargate-ecr/` with `.` as the context, the build will fail because `data-processing-ecr/...` is outside that context.

Push flow example:

```bash
aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 647371008162.dkr.ecr.us-west-1.amazonaws.com
docker tag reddit-processor-fargate:latest 647371008162.dkr.ecr.us-west-1.amazonaws.com/data-processing-fargate-ecr:latest
docker push 647371008162.dkr.ecr.us-west-1.amazonaws.com/data-processing-fargate-ecr:latest
```

The task entrypoint reads the Step Functions task token plus the S3 trigger bucket/key from container environment variables, then calls the existing processor code and preserves the current `send_task_success` / `send_task_failure` callback behavior.

This image now bakes in:

- Python dependencies and core runtime libraries (`libgomp1`, `libstdc++6`, `ca-certificates`)
- both Hugging Face sentiment models and tokenizers under `/opt/models/...`
- the static parquet reference asset at `/opt/data/merged.parquet`
- downloads the Hugging Face model assets during `docker build`, then forces offline loading at runtime
