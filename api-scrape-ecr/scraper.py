import os
import time
import asyncio
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import boto3
import asyncpraw
import pandas as pd
from asyncprawcore import Requestor

try:
    from asyncpraw.models import MoreComments
except Exception:
    MoreComments = None


# -----------------------
# Rate limiter
# -----------------------
class AsyncSlidingWindowRateLimiter:
    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = int(max_calls)
        self.period = float(period)
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self.period
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                sleep_for = (self._calls[0] + self.period) - now
            await asyncio.sleep(max(0.0, sleep_for))


class HttpRequestCounter:
    def __init__(self) -> None:
        self._n = 0
        self._lock = asyncio.Lock()

    async def inc(self) -> None:
        async with self._lock:
            self._n += 1

    async def get(self) -> int:
        async with self._lock:
            return self._n


class RateLimitedRequestor(Requestor):
    def __init__(
        self,
        *args: Any,
        rate_limiter: AsyncSlidingWindowRateLimiter,
        request_counter: HttpRequestCounter,
        **kwargs: Any,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._request_counter = request_counter
        super().__init__(*args, **kwargs)

    async def request(self, *args: Any, **kwargs: Any):
        await self._rate_limiter.acquire()
        await self._request_counter.inc()
        return await super().request(*args, **kwargs)


# -----------------------
# Helpers
# -----------------------
def _author_name(author_obj: Any) -> Optional[str]:
    return str(author_obj) if author_obj is not None else None


def _must_getenv(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

async def _expand_comments_best_effort(
    comments_forest,
    request_counter: HttpRequestCounter,
    request_budget_global: int,
    request_budget_submission: int,
    submission_start_requests: int,
    batch_size: int = 32,
) -> Tuple[bool, str]:
    """
    Returns (fully_expanded, stop_reason)

    stop_reason in:
      - "completed"
      - "global_budget"
      - "per_submission_budget"
      - "exception"
    """

    def _has_morecomments() -> bool:
        # list() is local flattening, no HTTP
        flat = comments_forest.list()
        if MoreComments is None:
            return any(getattr(c, "__class__", None).__name__ == "MoreComments" for c in flat)
        return any(isinstance(c, MoreComments) for c in flat)

    if not _has_morecomments():
        return True, "completed"

    while True:
        current = await request_counter.get()
        used_global = current
        used_sub = current - submission_start_requests

        if used_global >= request_budget_global:
            return False, "global_budget"
        if used_sub >= request_budget_submission:
            return False, "per_submission_budget"

        try:
            await comments_forest.replace_more(limit=batch_size)
        except Exception:
            return False, "exception"

        if not _has_morecomments():
            return True, "completed"
        
def _strip_fullname(fullname: str) -> Tuple[str, str]:
    """
    Returns (kind, id) from a fullname like 't1_abc' or 't3_xyz'.
    If it doesn't look like a fullname, returns ('', fullname).
    """
    if not fullname:
        return "", ""
    if "_" in fullname:
        kind, _id = fullname.split("_", 1)
        return kind, _id
    return "", fullname


def filter_comments_with_computable_depth(
    comments_rows: List[Dict[str, Any]],
    submission_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Keeps only comments whose entire parent chain exists up to the submission.
    Adds 'depth' to each kept comment.

    Returns: (filtered_rows, stats)
    stats: kept, dropped_missing_parent, dropped_cycle_or_invalid
    """
    submission_fullname = f"t3_{submission_id}"

    by_id: Dict[str, Dict[str, Any]] = {row["id"]: row for row in comments_rows}
    depth_cache: Dict[str, int] = {}

    dropped_missing_parent = 0
    dropped_cycle_or_invalid = 0

    # Use iterative stack-based resolution to avoid recursion limits and to detect cycles.
    def compute_depth(comment_id: str) -> Optional[int]:
        if comment_id in depth_cache:
            return depth_cache[comment_id]
        if comment_id not in by_id:
            return None

        visiting: set[str] = set()
        stack: List[str] = [comment_id]

        while stack:
            cid = stack[-1]
            if cid in depth_cache:
                stack.pop()
                continue

            if cid in visiting:
                # We came back to a node currently being processed => cycle
                return None

            visiting.add(cid)
            row = by_id.get(cid)
            if not row:
                return None

            parent_fullname = row.get("parent_id") or ""
            parent_kind, parent_id = _strip_fullname(parent_fullname)

            if parent_fullname == submission_fullname or parent_kind == "t3":
                # Direct reply to post (or treat any t3 as top-level for safety)
                depth_cache[cid] = 1
                visiting.remove(cid)
                stack.pop()
                continue

            if parent_kind != "t1" or not parent_id:
                # Unknown / malformed parent_id
                return None

            # Parent must exist in our scraped set
            if parent_id not in by_id:
                return None

            # Ensure parent depth is resolved first
            if parent_id not in depth_cache:
                stack.append(parent_id)
                continue

            # Parent depth known => child depth
            depth_cache[cid] = depth_cache[parent_id] + 1
            visiting.remove(cid)
            stack.pop()

        return depth_cache.get(comment_id)

    filtered: List[Dict[str, Any]] = []
    for row in comments_rows:
        d = compute_depth(row["id"])
        if d is None:
            # Decide why: missing parent vs other invalid
            parent_fullname = row.get("parent_id") or ""
            pk, pid = _strip_fullname(parent_fullname)
            if pk == "t1" and pid and pid not in by_id:
                dropped_missing_parent += 1
            else:
                dropped_cycle_or_invalid += 1
            continue

        # Keep it and attach depth for downstream processing
        row2 = dict(row)
        row2["depth"] = int(d)
        filtered.append(row2)

    stats = {
        "kept": len(filtered),
        "dropped_missing_parent": dropped_missing_parent,
        "dropped_cycle_or_invalid": dropped_cycle_or_invalid,
        "total_before_filter": len(comments_rows),
    }
    return filtered, stats

# -----------------------
# Submission selection
# -----------------------
async def fetch_top_and_discussion_submissions(
    reddit: asyncpraw.Reddit,
    subreddit_name: str,
    cutoff_utc: float,
    top_n: int = 5,
    min_score: int = 100,
    daily_flair: str = "Daily Discussion",
    hot_scan_limit: int = 25,
) -> Tuple[List[Dict[str, Any]], List[asyncpraw.models.Submission]]:
    subreddit = await reddit.subreddit(subreddit_name)
    selected_by_id: Dict[str, asyncpraw.models.Submission] = {}

    # Top N from the last day
    async for s in subreddit.top(time_filter="day", limit=top_n):
        created = float(getattr(s, "created_utc", 0) or 0)
        if created < cutoff_utc:
            continue
        if int(getattr(s, "score", 0) or 0) < min_score:
            continue
        selected_by_id[s.id] = s

    # Stickied discussion threads from HOT
    title_needles = (
        "what are your moves",
        "daily discussion thread",
        "weekend discussion thread",
    )

    async for s in subreddit.hot(limit=hot_scan_limit):
        created = float(getattr(s, "created_utc", 0) or 0)
        if created < cutoff_utc:
            continue
        if not bool(getattr(s, "stickied", False)):
            continue

        flair = (getattr(s, "link_flair_text", None) or "").strip()
        title = (getattr(s, "title", "") or "").strip().lower()

        is_discussion = (flair == daily_flair) or any(n in title for n in title_needles)
        if is_discussion:
            selected_by_id[s.id] = s

    selected = list(selected_by_id.values())

    subs: List[Dict[str, Any]] = []
    for s in selected:
        created = float(getattr(s, "created_utc", 0) or 0)
        subs.append(
            {
                "id": s.id,
                "author": _author_name(s.author),
                "title": s.title,
                "url": s.url,
                "selftext": s.selftext,
                "score": s.score,
                "upvote_ratio": s.upvote_ratio,
                "num_comments": s.num_comments,
                "stickied": s.stickied,
                "created_utc": created,
                "subreddit": str(s.subreddit),
                "link_flair_text": getattr(s, "link_flair_text", None),
            }
        )

    subs.sort(key=lambda x: x.get("created_utc", 0), reverse=True)
    selected.sort(key=lambda s: float(getattr(s, "created_utc", 0) or 0), reverse=True)
    return subs, selected


# -----------------------
# Comments
# -----------------------
async def fetch_comments_for_submission(
    submission: asyncpraw.models.Submission,
    request_counter: HttpRequestCounter,
    request_budget_global: int,
    request_budget_submission: int,
    replace_more_batch_size: int = 32,
) -> Tuple[List[Dict[str, Any]], bool, str, Dict[str, int]]:
    """
    Returns:
      (comments_rows, fully_expanded, stop_reason, stats)

    stats includes request counts used globally and for this submission.
    """
    start_requests = await request_counter.get()

    await submission.load()
    comments = submission.comments  # property (no await)

    fully_expanded, stop_reason = await _expand_comments_best_effort(
        comments,
        request_counter=request_counter,
        request_budget_global=request_budget_global,
        request_budget_submission=request_budget_submission,
        submission_start_requests=start_requests,
        batch_size=replace_more_batch_size,
    )

    comms: List[Dict[str, Any]] = []
    for c in comments.list():
        if MoreComments is not None and isinstance(c, MoreComments):
            continue
        if getattr(c, "__class__", None).__name__ == "MoreComments":
            continue

        comms.append(
            {
                "id": c.id,
                "author": _author_name(c.author),
                "body": c.body,
                "parent_id": c.parent_id,
                "link_id": c.link_id,
                "score": c.score,
                "created_utc": float(getattr(c, "created_utc", 0) or 0),
                "subreddit": str(c.subreddit),
                "submission_id": submission.id,
            }
        )

    comms, depth_stats = filter_comments_with_computable_depth(comms, submission_id=submission.id)

    end_requests = await request_counter.get()
    stats = {
        "start_requests": int(start_requests),
        "end_requests": int(end_requests),
        "used_requests": int(end_requests - start_requests),
        **depth_stats,
    }

    return comms, fully_expanded, stop_reason, stats

async def fetch_comments_for_submissions(
    submissions: List[asyncpraw.models.Submission],
    request_counter: HttpRequestCounter,
    request_budget_global: int,
    request_budget_submission: int,
    max_concurrency: int = 3,
    replace_more_batch_size: int = 32,
) -> Tuple[List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(s: asyncpraw.models.Submission):
        async with sem:
            return await fetch_comments_for_submission(
                s,
                request_counter=request_counter,
                request_budget_global=request_budget_global,
                request_budget_submission=request_budget_submission,
                replace_more_batch_size=replace_more_batch_size,
            )

    results = await asyncio.gather(*(_one(s) for s in submissions))

    all_comments = [c for (rows, _, _, _) in results for c in rows]
    fully_expanded_all = all(expanded for (_, expanded, _, _) in results)

    per_submission = []
    for (rows, expanded, reason, stats), sub in zip(results, submissions):
        per_submission.append(
            {
                "submission_id": sub.id,
                "fully_expanded": bool(expanded),
                "stop_reason": reason,
                "comments_returned": int(len(rows)),
                **stats,
            }
        )

    return all_comments, fully_expanded_all, per_submission

# -----------------------
# Output
# -----------------------
def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """
    NOTE: This requires a parquet engine in your container, e.g. pyarrow or fastparquet.
    If you see "Unable to find a usable engine", add pyarrow to your image.
    """
    bio = BytesIO()
    df.to_parquet(bio, index=False)
    return bio.getvalue()


async def run_job() -> Dict[str, Any]:
    REQUEST_BUDGET = int(os.environ.get("REQUEST_BUDGET", "1450"))
    MAX_REQUESTS_PER_SUBMISSION = int(os.environ.get("MAX_REQUESTS_PER_SUBMISSION", "600"))
    REPLACE_MORE_BATCH_SIZE = int(os.environ.get("REPLACE_MORE_BATCH_SIZE", "32"))

    subreddit = os.environ.get("SUBREDDIT", "wallstreetbets")
    hours = float(os.environ.get("HOURS", "24"))
    min_score = int(os.environ.get("MIN_SCORE", "100"))
    top_n = int(os.environ.get("TOP_N", "5"))
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "3"))

    bucket = os.environ["S3_BUCKET"]
    client_id = os.environ["REDDIT_CLIENT_ID"]
    client_secret = os.environ["REDDIT_CLIENT_SECRET"]
    user_agent = os.environ["REDDIT_USER_AGENT"]

    now_utc = time.time()
    cutoff_utc = now_utc - (hours * 3600)

    rate_limiter = AsyncSlidingWindowRateLimiter(max_calls=100, period=60.0)
    request_counter = HttpRequestCounter()

    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        requestor_class=RateLimitedRequestor,
        requestor_kwargs={"rate_limiter": rate_limiter, "request_counter": request_counter},
    )

    try:
        submissions_rows, selected_submission_objs = await fetch_top_and_discussion_submissions(
            reddit=reddit,
            subreddit_name=subreddit,
            cutoff_utc=cutoff_utc,
            top_n=top_n,
            min_score=min_score,
            daily_flair="Daily Discussion",
            hot_scan_limit=25,
        )

        comments_rows, fully_expanded_all, per_submission_expansion = await fetch_comments_for_submissions(
            selected_submission_objs,
            request_counter=request_counter,
            request_budget_global=REQUEST_BUDGET,
            request_budget_submission=MAX_REQUESTS_PER_SUBMISSION,
            max_concurrency=max_concurrency,
            replace_more_batch_size=REPLACE_MORE_BATCH_SIZE,
        )
    finally:
        await reddit.close()

    subs_df = pd.DataFrame(submissions_rows)
    comm_df = pd.DataFrame(comments_rows)

    http_requests = await request_counter.get()

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now_utc))
    subs_key = f"submissions_{ts}.parquet"
    comm_key = f"comments_{ts}.parquet"
    manifest_key = f"manifest_{ts}.json"

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=subs_key, Body=df_to_parquet_bytes(subs_df), ContentType="application/octet-stream")
    s3.put_object(Bucket=bucket, Key=comm_key, Body=df_to_parquet_bytes(comm_df), ContentType="application/octet-stream")

    manifest = {
        "timestamp": ts,
        "bucket": bucket,
        "subreddit": subreddit,
        "hours": hours,
        "top_n": top_n,
        "min_score": min_score,
        "selected_submission_count": len(selected_submission_objs),
        "submissions": int(len(subs_df)),
        "comments": int(len(comm_df)),
        "http_requests": int(http_requests),
        "request_budget": int(REQUEST_BUDGET),
        "max_requests_per_submission": int(MAX_REQUESTS_PER_SUBMISSION),
        "fully_expanded_all": bool(fully_expanded_all),
        "per_submission_expansion": per_submission_expansion,
        "submissions_key": subs_key,
        "comments_key": comm_key,
    }
    s3.put_object(Bucket=bucket, Key=manifest_key, Body=json.dumps(manifest).encode("utf-8"), ContentType="application/json")

    return manifest

def handler(event, context):
    # Lambda is not running an event loop already, so asyncio.run is fine here.
    return asyncio.run(run_job())