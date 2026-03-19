import json
import os
import datetime as dt
import boto3

s3 = boto3.client("s3")

DEST_BUCKET = os.environ.get("DEST_BUCKET")  # optional for testing

def lambda_handler(event, context):
    print("Lambda invoked!")
    print("Event received:", json.dumps(event))

    now = dt.datetime.utcnow().isoformat() + "Z"

    response = {
        "message": "Container Lambda is working!",
        "timestamp": now,
        "request_id": context.aws_request_id,
    }

    return response
