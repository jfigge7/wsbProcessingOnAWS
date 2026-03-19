import json
import os
import sys
import uuid
from types import SimpleNamespace

import preprocessing_lambda


def _build_event_from_env():
    raw_event = os.environ.get("PROCESSOR_EVENT_JSON")
    if raw_event:
        event = json.loads(raw_event)
    else:
        bucket = os.environ["PROCESSOR_TRIGGER_BUCKET"]
        key = os.environ["PROCESSOR_TRIGGER_KEY"]
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": bucket},
                        "object": {"key": key},
                    }
                }
            ]
        }

        output_bucket = os.environ.get("PROCESSOR_OUTPUT_BUCKET")
        output_prefix = os.environ.get("PROCESSOR_OUTPUT_PREFIX")
        output_file_prefix = os.environ.get("PROCESSOR_OUTPUT_FILE_PREFIX")
        rows_per_file = os.environ.get("PROCESSOR_OUTPUT_ROWS_PER_FILE")
        sort_before_write = os.environ.get("PROCESSOR_SORT_BEFORE_WRITE")

        output = {}
        if output_bucket:
            output["bucket"] = output_bucket
        if output_prefix:
            output["prefix"] = output_prefix
        if output_file_prefix:
            output["file_prefix"] = output_file_prefix
        if rows_per_file:
            output["rows_per_file"] = int(rows_per_file)
        if sort_before_write:
            output["sort_before_write"] = sort_before_write.strip().lower() in ("1", "true", "yes", "y")
        if output:
            event["output"] = output

    task_token = os.environ.get("STEP_FUNCTION_TASK_TOKEN")
    if task_token:
        event["taskToken"] = task_token
    return event


def main():
    event = _build_event_from_env()
    context = SimpleNamespace(
        aws_request_id=os.environ.get("AWS_REQUEST_ID", str(uuid.uuid4()))
    )
    result = preprocessing_lambda.lambda_handler(event, context)
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
