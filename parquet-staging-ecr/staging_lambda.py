import json
import logging
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import unquote_plus

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
RAW_BUCKET = os.environ["RAW_BUCKET"]
RAW_PREFIX = os.environ.get("RAW_PREFIX", "imported-parquet/")
DELETE_SOURCE_AFTER_STAGE = os.environ.get("DELETE_SOURCE_AFTER_STAGE", "true").lower() in ("1", "true", "yes", "y")
TARGET_COMMENTS_PER_BATCH = max(1, int(os.environ.get("TARGET_COMMENTS_PER_BATCH", "50000")))
ORPHAN_GRACE_SECONDS = max(0, int(os.environ.get("ORPHAN_GRACE_SECONDS", "900")))
MAX_TRADING_DAYS_PER_BATCH = max(1, int(os.environ.get("MAX_TRADING_DAYS_PER_BATCH", "5")))
BATCHES_PREFIX = "batches/"
KEY_RE = re.compile(
    r"^(?:(?P<root>.+?)/)?(?P<dataset>comments|submissions)/year=(?P<year>\d{4})/month=(?P<month>\d{1,2})/day=(?P<day>\d{1,2})/[^/]+\.parquet$"
)
RAW_DAY_MANIFEST_RE = re.compile(
    r"^(.*/)?year=(?P<year>\d{4})/month=(?P<month>\d{2})/day=(?P<day>\d{2})/manifest_(?P=year)(?P=month)(?P=day)\.json$"
)


def _list_objects(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj.get("Key")
            if key:
                objects.append({"key": key, "last_modified": obj.get("LastModified")})
    objects.sort(key=lambda item: item["key"])
    return objects


def _list_keys(bucket, prefix):
    return [item["key"] for item in _list_objects(bucket, prefix)]


def _head_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = (exc.response.get("Error") or {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _delete_keys(bucket, keys):
    keys = [key for key in keys if key]
    if not keys:
        return 0
    deleted = 0
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
        )
        deleted += len(chunk)
    return deleted


def _delete_prefix(bucket, prefix):
    return _delete_keys(bucket, _list_keys(bucket, prefix))


def _read_json(bucket, key):
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def _write_json(bucket, key, payload):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def _read_table(bucket, key):
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = (exc.response.get("Error") or {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise FileNotFoundError(f"s3://{bucket}/{key}") from exc
        raise
    body = resp["Body"].read()
    return pq.read_table(BytesIO(body))


def _concat_tables(tables):
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote=True)


def _merge_dataset(bucket, source_keys, target_key):
    tables = [_read_table(bucket, source_key) for source_key in source_keys]
    merged = _concat_tables(tables)
    out = BytesIO()
    pq.write_table(merged, out, compression="snappy")
    s3.put_object(Bucket=RAW_BUCKET, Key=target_key, Body=out.getvalue(), ContentType="application/octet-stream")
    return int(merged.num_rows)


def _manifest_keys(manifest, singular_field, plural_field):
    keys = []
    plural_value = manifest.get(plural_field) or []
    if isinstance(plural_value, list):
        keys.extend(str(key) for key in plural_value if key)
    singular_value = manifest.get(singular_field)
    if singular_value:
        singular_value = str(singular_value)
        if singular_value not in keys:
            keys.insert(0, singular_value)
    return keys


def _month_source_prefixes(root, year, month):
    base_prefix = f"{root}/" if root else ""
    month_num = int(month)
    return [
        f"{base_prefix}comments/year={year}/month={month_num}/",
        f"{base_prefix}submissions/year={year}/month={month_num}/",
    ]


def _month_has_pending_source_parquets(bucket, root, year, month):
    if not DELETE_SOURCE_AFTER_STAGE:
        return False
    for prefix in _month_source_prefixes(root, year, month):
        for key in _list_keys(bucket, prefix):
            if key.endswith(".parquet"):
                return True
    return False


def _month_source_inventory(bucket, root, year, month):
    comments_prefix, submissions_prefix = _month_source_prefixes(root, year, month)
    inventory = {}
    for dataset, prefix in (("comments", comments_prefix), ("submissions", submissions_prefix)):
        for obj in _list_objects(bucket, prefix):
            key = obj["key"]
            if not key.endswith(".parquet"):
                continue
            match = KEY_RE.match(key)
            if not match:
                continue
            day = f"{int(match.group('day')):02d}"
            day_info = inventory.setdefault(day, {"comments": [], "submissions": []})
            day_info[dataset].append(obj)
    for day_info in inventory.values():
        day_info["comments"].sort(key=lambda item: item["key"])
        day_info["submissions"].sort(key=lambda item: item["key"])
    return inventory


def _stage_day(bucket, year, month, day, day_info):
    source_comments_keys = [item["key"] for item in day_info.get("comments") or []]
    source_submissions_keys = [item["key"] for item in day_info.get("submissions") or []]
    if not source_comments_keys or not source_submissions_keys:
        return None

    raw_day_prefix = f"{RAW_PREFIX.rstrip('/')}/year={year}/month={month}/day={day}/"
    raw_comments_key = raw_day_prefix + "comments.parquet"
    raw_submissions_key = raw_day_prefix + "submissions.parquet"
    manifest_key = raw_day_prefix + f"manifest_{year}{month}{day}.json"
    already_staged = _head_exists(RAW_BUCKET, manifest_key)

    if not already_staged:
        try:
            comment_count = _merge_dataset(bucket, source_comments_keys, raw_comments_key)
            _merge_dataset(bucket, source_submissions_keys, raw_submissions_key)
        except FileNotFoundError:
            if _head_exists(RAW_BUCKET, manifest_key):
                logger.info("Day %s-%s-%s was already staged by another invocation; skipping duplicate work.", year, month, day)
                already_staged = True
            else:
                raise
        if not already_staged:
            manifest = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "bucket": RAW_BUCKET,
                "comments_key": raw_comments_key,
                "submissions_key": raw_submissions_key,
                "source_bucket": bucket,
                "source_comments_key": source_comments_keys[0],
                "source_submissions_key": source_submissions_keys[0],
                "source_comments_keys": source_comments_keys,
                "source_submissions_keys": source_submissions_keys,
                "comment_count": int(comment_count),
                "trading_day": f"{year}-{month}-{day}",
            }
            _write_json(RAW_BUCKET, manifest_key, manifest)

    if DELETE_SOURCE_AFTER_STAGE:
        _delete_keys(bucket, source_comments_keys + source_submissions_keys)

    return {
        "day": day,
        "already_staged": already_staged,
        "manifest_key": manifest_key,
        "raw_comments_key": raw_comments_key,
        "raw_submissions_key": raw_submissions_key,
    }


def _drop_mature_orphan_days(bucket, root, year, month):
    if not DELETE_SOURCE_AFTER_STAGE:
        return {"dropped_days": [], "dropped_keys": 0, "remaining_orphan_days": []}

    now = datetime.now(timezone.utc)
    inventory = _month_source_inventory(bucket, root, year, month)
    dropped_days = []
    dropped_keys = 0
    remaining_orphan_days = []

    for orphan_day, day_info in sorted(inventory.items()):
        has_comments = bool(day_info.get("comments"))
        has_submissions = bool(day_info.get("submissions"))
        if has_comments and has_submissions:
            continue
        if not (has_comments or has_submissions):
            continue

        objects = list(day_info.get("comments") or []) + list(day_info.get("submissions") or [])
        latest_modified = None
        for obj in objects:
            candidate = obj.get("last_modified")
            if candidate and (latest_modified is None or candidate > latest_modified):
                latest_modified = candidate
        if latest_modified is not None and latest_modified.tzinfo is None:
            latest_modified = latest_modified.replace(tzinfo=timezone.utc)
        age_seconds = (now - latest_modified).total_seconds() if latest_modified is not None else ORPHAN_GRACE_SECONDS
        if age_seconds < ORPHAN_GRACE_SECONDS:
            remaining_orphan_days.append(orphan_day)
            continue

        dropped_days.append(orphan_day)
        dropped_keys += _delete_keys(bucket, [obj["key"] for obj in objects])

    return {
        "dropped_days": dropped_days,
        "dropped_keys": dropped_keys,
        "remaining_orphan_days": remaining_orphan_days,
    }


def _load_manifest_infos(month_prefix):
    infos = []
    for key in _list_keys(RAW_BUCKET, month_prefix):
        if not key.endswith(".json"):
            continue
        if f"/{BATCHES_PREFIX}" in key or "/monthly_ready.json" in key:
            continue
        match = RAW_DAY_MANIFEST_RE.match(key)
        if not match:
            continue
        manifest = _read_json(RAW_BUCKET, key)
        comments_keys = _manifest_keys(manifest, "comments_key", "comments_keys")
        submissions_keys = _manifest_keys(manifest, "submissions_key", "submissions_keys")
        if not comments_keys or not submissions_keys:
            continue
        if not all(_head_exists(RAW_BUCKET, manifest_key) for manifest_key in comments_keys + submissions_keys):
            continue
        comment_count = manifest.get("comment_count")
        if comment_count is None:
            comment_count = 0
        infos.append(
            {
                "day": match.group("day"),
                "manifest_key": key,
                "manifest": manifest,
                "comment_count": max(0, int(comment_count or 0)),
            }
        )
    infos.sort(key=lambda item: (item["day"], item["manifest_key"]))
    return infos


def _partition_manifest_infos(manifest_infos, target_comments):
    batches = []
    current = []
    current_comments = 0
    for info in manifest_infos:
        info_comments = max(0, int(info.get("comment_count") or 0))
        if current and (
            current_comments + info_comments > target_comments
            or len(current) >= MAX_TRADING_DAYS_PER_BATCH
        ):
            batches.append(current)
            current = []
            current_comments = 0
        current.append(info)
        current_comments += info_comments
    if current:
        batches.append(current)
    return batches


def _materialize_month_batches(raw_month_prefix, manifest_infos):
    batch_root_prefix = f"{raw_month_prefix}{BATCHES_PREFIX}"
    deleted_batch_objects = _delete_prefix(RAW_BUCKET, batch_root_prefix)
    batch_triggers = []
    batches = _partition_manifest_infos(manifest_infos, TARGET_COMMENTS_PER_BATCH)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for idx, batch in enumerate(batches, start=1):
        batch_prefix = f"{batch_root_prefix}batch={idx:04d}/"
        latest_manifest = None
        batch_comment_count = 0
        for info in batch:
            latest_manifest = info["manifest_key"]
            batch_comment_count += int(info.get("comment_count") or 0)
            manifest_name = info["manifest_key"].rsplit("/", 1)[-1]
            _write_json(RAW_BUCKET, f"{batch_prefix}{manifest_name}", info["manifest"])

        ready_key = f"{batch_prefix}monthly_ready.json"
        _write_json(
            RAW_BUCKET,
            ready_key,
            {
                "timestamp": timestamp,
                "bucket": RAW_BUCKET,
                "prefix": batch_prefix,
                "latest_manifest": latest_manifest,
                "batch_index": idx,
                "batch_size_days": len(batch),
                "batch_comment_count": batch_comment_count,
            },
        )
        batch_triggers.append(f"s3://{RAW_BUCKET}/{ready_key}")

    return {
        "batch_count": len(batches),
        "batch_triggers": batch_triggers,
        "deleted_batch_objects": deleted_batch_objects,
    }


def _stage_one(bucket, key):
    match = KEY_RE.match(key)
    if not match:
        return {"status": "ignored", "key": key}

    root = (match.group("root") or "").strip("/")
    year = f"{int(match.group('year')):04d}"
    month = f"{int(match.group('month')):02d}"
    day = f"{int(match.group('day')):02d}"
    raw_month_prefix = f"{RAW_PREFIX.rstrip('/')}/year={year}/month={month}/"

    staged_days = []
    inventory = _month_source_inventory(bucket, root, year, month)
    for inventory_day in sorted(inventory):
        day_info = inventory[inventory_day]
        if day_info.get("comments") and day_info.get("submissions"):
            stage_result = _stage_day(bucket, year, month, inventory_day, day_info)
            if stage_result:
                staged_days.append(stage_result)

    orphan_result = _drop_mature_orphan_days(bucket, root, year, month)
    current_stage = next((item for item in staged_days if item["day"] == day), None)
    current_dropped = day in set(orphan_result.get("dropped_days") or [])
    pending_source = _month_has_pending_source_parquets(bucket, root, year, month)

    if pending_source:
        if current_stage:
            status = "already_staged_waiting_for_month_completion" if current_stage["already_staged"] else "staged_waiting_for_month_completion"
        elif current_dropped:
            status = "dropped_unpaired_day_waiting_for_month_completion"
        else:
            status = "waiting_for_month_completion"
        return {
            "status": status,
            "deleted_source": DELETE_SOURCE_AFTER_STAGE,
            "month_complete": False,
            "staged_days": [item["day"] for item in staged_days],
            "dropped_days": orphan_result.get("dropped_days") or [],
            "remaining_orphan_days": orphan_result.get("remaining_orphan_days") or [],
        }

    manifest_infos = _load_manifest_infos(raw_month_prefix)
    batch_result = _materialize_month_batches(raw_month_prefix, manifest_infos)

    if current_stage:
        return {
            "status": "already_staged_month_batched" if current_stage["already_staged"] else "month_batched",
            "manifest": f"s3://{RAW_BUCKET}/{current_stage['manifest_key']}",
            "comments_key": f"s3://{RAW_BUCKET}/{current_stage['raw_comments_key']}",
            "submissions_key": f"s3://{RAW_BUCKET}/{current_stage['raw_submissions_key']}",
            "deleted_source": DELETE_SOURCE_AFTER_STAGE,
            "month_complete": True,
            "batch_count": batch_result["batch_count"],
            "batch_triggers": batch_result["batch_triggers"],
            "deleted_batch_objects": batch_result["deleted_batch_objects"],
            "staged_days": [item["day"] for item in staged_days],
            "dropped_days": orphan_result.get("dropped_days") or [],
        }

    if current_dropped:
        return {
            "status": "dropped_unpaired_day_month_batched",
            "deleted_source": DELETE_SOURCE_AFTER_STAGE,
            "month_complete": True,
            "batch_count": batch_result["batch_count"],
            "batch_triggers": batch_result["batch_triggers"],
            "deleted_batch_objects": batch_result["deleted_batch_objects"],
            "staged_days": [item["day"] for item in staged_days],
            "dropped_days": orphan_result.get("dropped_days") or [],
        }

    return {
        "status": "month_batched" if staged_days else "no_stageable_day_month_batched",
        "deleted_source": DELETE_SOURCE_AFTER_STAGE,
        "month_complete": True,
        "batch_count": batch_result["batch_count"],
        "batch_triggers": batch_result["batch_triggers"],
        "deleted_batch_objects": batch_result["deleted_batch_objects"],
        "staged_days": [item["day"] for item in staged_days],
        "dropped_days": orphan_result.get("dropped_days") or [],
    }


def lambda_handler(event, context):
    results = []
    for record in event.get("Records") or []:
        bucket = ((record.get("s3") or {}).get("bucket") or {}).get("name")
        key = unquote_plus((((record.get("s3") or {}).get("object") or {}).get("key") or ""))
        if not bucket or not key:
            continue
        results.append(_stage_one(bucket, key))
    return {"results": results}
