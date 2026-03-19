import os
import re
import math
import json
import logging
from io import BytesIO
from urllib.parse import unquote_plus
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Iterator

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from botocore.exceptions import ClientError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from zoneinfo import ZoneInfo  # py3.9+
except ImportError:
    ZoneInfo = None


# ======================================================================================
# Logging
# ======================================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ======================================================================================
# Configuration (Lambda: prefer env vars)
# ======================================================================================
FINBERT_MODEL_DIR = os.getenv("FINBERT_MODEL_DIR", "/opt/models/finbert")
TWITTER_ROBERTA_MODEL_DIR = os.getenv("TWITTER_ROBERTA_MODEL_DIR", "/opt/models/twitter_roberta_sentiment")
STOCKS_PARQUET = os.getenv("STOCKS_PARQUET", "/opt/data/merged.parquet")
FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
TWITTER_ROBERTA_MODEL_NAME = os.getenv("TWITTER_ROBERTA_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/model-cache")
HF_HOME = os.getenv("HF_HOME", os.path.join(MODEL_CACHE_DIR, "huggingface"))
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))

BATCH_SIZE_TWITTER = int(os.getenv("BATCH_SIZE_TWITTER", "64"))
BATCH_SIZE_FINBERT = int(os.getenv("BATCH_SIZE_FINBERT", "32"))
CHUNK_SIZE_ITEMS = int(os.getenv("CHUNK_SIZE_ITEMS", "2048"))
MAX_DEPTH = int(os.getenv("MAX_COMMENT_DEPTH", "10"))
EASTERN_TZ = os.getenv("MARKET_TZ", "America/New_York")

TITLE_WEIGHT_MULT = float(os.getenv("TITLE_WEIGHT_MULT", "1.5"))
SELFTEXT_WEIGHT_MULT = float(os.getenv("SELFTEXT_WEIGHT_MULT", "1.0"))

# Output is ALWAYS parquet
OUTPUT_ROWS_PER_FILE_DEFAULT = int(os.getenv("OBS_OUTPUT_ROWS_PER_FILE", "50000"))
OUTPUT_FILE_PREFIX_DEFAULT = os.getenv("OBS_OUTPUT_PREFIX", "observations")
SORT_BEFORE_WRITE_DEFAULT = os.getenv("SORT_BEFORE_WRITE", "False").lower() in ("1", "true", "yes", "y")

# S3 output defaults
OBS_OUTPUT_S3_BUCKET = os.getenv("OBS_OUTPUT_S3_BUCKET")  # may be overridden by event.output.bucket
OBS_OUTPUT_S3_PREFIX = os.getenv("OBS_OUTPUT_S3_PREFIX", "observations/")  # may be overridden by event.output.prefix

MIN_SUBMISSION_COMMENTS = int(os.getenv("MIN_SUBMISSION_COMMENTS", "25"))
MIN_SUBMISSION_SCORE = int(os.getenv("MIN_SUBMISSION_SCORE", "25"))

MIN_TICKER_COMMENT_ITEMS = int(os.getenv("MIN_TICKER_COMMENT_ITEMS", "5"))
MIN_TICKER_TOTAL_ITEMS = int(os.getenv("MIN_TICKER_TOTAL_ITEMS", "0"))

ORDER_DECAY_BASE = float(os.getenv("ORDER_DECAY_BASE", "0.95"))
ORDER_DECAY_EVERY_N = float(os.getenv("ORDER_DECAY_EVERY_N", "1"))

TIME_DECAY_BASE = float(os.getenv("TIME_DECAY_BASE", "0.95"))
TIME_DECAY_EVERY_HOURS = float(os.getenv("TIME_DECAY_EVERY_HOURS", "1"))

DEPTH_DECAY_BASE = float(os.getenv("DEPTH_DECAY_BASE", "0.99"))
MIN_EFFECTIVE_WEIGHT = float(os.getenv("MIN_EFFECTIVE_WEIGHT", "0.005"))

# NEW: Input / archive config (input is S3-trigger bucket with manifest_*.json)
INPUT_BUCKET = os.getenv("INPUT_BUCKET")  # optional override (otherwise use trigger bucket)
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "")  # optional: restrict listing to a prefix (e.g. "incoming/")
ARCHIVE_BUCKET = os.getenv("ARCHIVE_BUCKET")  # required for "move"
ARCHIVE_PREFIX = os.getenv("ARCHIVE_PREFIX", "archive/")  # optional

# NEW: Trading calendar config
TRADING_CALENDAR = os.getenv("TRADING_CALENDAR", "XNYS")  # e.g. XNYS, XNAS

# NEW: Manifest settings
MANIFEST_PREFIX = os.getenv("MANIFEST_PREFIX", "manifest_")  # files like manifest_YYYYMMDDTHHMMSSZ.json
MANIFEST_SUFFIX = os.getenv("MANIFEST_SUFFIX", ".json")

COMMENT_PARQUET_COLUMNS = ["id", "parent_id", "link_id", "body", "author", "score", "created_utc", "created_ts", "created", "trading_date_et"]
SUBMISSION_PARQUET_COLUMNS = ["id", "author", "title", "selftext", "score", "num_comments", "created_utc", "created_ts"]

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", TRANSFORMERS_CACHE)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", MODEL_CACHE_DIR)


# ======================================================================================
# Lazy globals
# ======================================================================================
_s3 = None
_sfn = None

_finbert_tok = None
_finbert_model = None
_twitter_tok = None
_twitter_model = None

_stocks_df = None
_symbol_regex = None
_company_regex = None
_cashtag_regex = None

_symbol_col_name = None
_company_col_name = None
_name_to_symbol: Dict[str, str] = {}

_sub_created_cache: Dict[str, Optional[float]] = {}


def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _get_sfn():
    global _sfn
    if _sfn is None:
        _sfn = boto3.client("stepfunctions")
    return _sfn


# ======================================================================================
# Helpers
# ======================================================================================
def _chunks(iterable: Iterable, n: int):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def _normalize_fullname(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _base_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if "_" in s and len(s.split("_", 1)[0]) == 2:
        return s.split("_", 1)[1]
    return s


def _is_submission_fullname(x: str) -> bool:
    return str(x).startswith("t3_")


def _clean_text(x: Any) -> str:
    if not x:
        return ""
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    return " ".join(str(x).split())


def _fast_parse_timestamp_to_epoch_utc(x: Any) -> Optional[float]:
    if x is None:
        return None

    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        try:
            return float(x)
        except Exception:
            return None

    if isinstance(x, datetime):
        dt = x
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()

    try:
        s = str(x).strip()
    except Exception:
        return None
    if not s:
        return None

    if s.isdigit():
        try:
            return float(s)
        except Exception:
            return None

    try:
        if all(ch in "0123456789.+-" for ch in s) and any(ch.isdigit() for ch in s):
            return float(s)
    except Exception:
        pass

    try:
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
        else:
            s2 = s
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    except Exception:
        pass

    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime().timestamp()
    except Exception:
        return None


def _epoch_utc_to_et_iso(epoch_seconds: float) -> Optional[str]:
    if epoch_seconds is None or ZoneInfo is None:
        return None
    try:
        tz_et = ZoneInfo(EASTERN_TZ)
        dt_utc = datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
        return dt_utc.astimezone(tz_et).isoformat()
    except Exception:
        return None


# ======================================================================================
# S3 writers (parquet-only)
# ======================================================================================
def _s3_key_join(prefix: str, key: str) -> str:
    prefix = (prefix or "").lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{key}"


def _write_batch_rows_to_s3(
    rows: List[Dict[str, Any]],
    *,
    bucket: str,
    prefix: str,
    batch_idx: int,
    file_prefix: str = OUTPUT_FILE_PREFIX_DEFAULT,
) -> str:
    """
    Writes one batch to S3 as parquet (in-memory).
    """
    if not rows:
        raise ValueError("rows is empty")

    s3 = _get_s3()
    key = _s3_key_join(prefix, f"{file_prefix}_{batch_idx:07d}.parquet")

    df = pd.DataFrame.from_records(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)

    buf = BytesIO()
    pq.write_table(table, buf, compression="zstd", use_dictionary=True)
    buf.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    return key


# ======================================================================================
# Models
# ======================================================================================
def _is_local_path(p: str) -> bool:
    return isinstance(p, str) and (p.startswith("/") or p.startswith("."))


def _ensure_model_available(model_ref: str, fallback_repo: str, cache_subdir: str) -> str:
    if not _is_local_path(model_ref):
        return model_ref
    if os.path.exists(model_ref):
        return model_ref

    target_dir = os.path.join(MODEL_CACHE_DIR, cache_subdir)
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(target_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(
        fallback_repo,
        use_fast=True,
        cache_dir=TRANSFORMERS_CACHE,
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        fallback_repo,
        cache_dir=TRANSFORMERS_CACHE,
    )
    tok.save_pretrained(target_dir)
    mdl.save_pretrained(target_dir)
    return target_dir


def _load_textcls_model(model_ref: str):
    if model_ref == FINBERT_MODEL_DIR:
        resolved_ref = _ensure_model_available(model_ref, FINBERT_MODEL_NAME, "finbert")
    elif model_ref == TWITTER_ROBERTA_MODEL_DIR:
        resolved_ref = _ensure_model_available(model_ref, TWITTER_ROBERTA_MODEL_NAME, "twitter_roberta_sentiment")
    else:
        resolved_ref = model_ref

    local_only = _is_local_path(resolved_ref)
    load_kwargs = {
        "local_files_only": local_only,
        "cache_dir": TRANSFORMERS_CACHE,
    }
    tok = AutoTokenizer.from_pretrained(resolved_ref, use_fast=True, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_ref, **load_kwargs)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tok, model


def _get_finbert_model():
    global _finbert_tok, _finbert_model
    if _finbert_model is None or _finbert_tok is None:
        _finbert_tok, _finbert_model = _load_textcls_model(FINBERT_MODEL_DIR)
    return _finbert_tok, _finbert_model


def _get_twitter_roberta_model():
    global _twitter_tok, _twitter_model
    if _twitter_model is None or _twitter_tok is None:
        _twitter_tok, _twitter_model = _load_textcls_model(TWITTER_ROBERTA_MODEL_DIR)
    return _twitter_tok, _twitter_model


@dataclass
class _ModelRunner:
    tokenizer: Any
    model: Any
    batch_size: int
    max_length: int = 256
    amp_dtype: Any = torch.float16

    def predict_proba_3way(self, texts: List[str]) -> List[Dict[str, float]]:
        if not texts:
            return []

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        id2label = getattr(self.model.config, "id2label", None) or {}

        def _norm_label(s: str) -> str:
            s = str(s).strip().lower()
            if s in ("label_0", "neg", "negative"):
                return "negative"
            if s in ("label_1", "neu", "neutral"):
                return "neutral"
            if s in ("label_2", "pos", "positive"):
                return "positive"
            return s

        idx_to_key = {}
        for i in range(int(getattr(self.model.config, "num_labels", 3))):
            idx_to_key[i] = _norm_label(id2label.get(i, f"label_{i}"))
        if not (set(idx_to_key.values()) >= {"negative", "neutral", "positive"}):
            idx_to_key = {0: "negative", 1: "neutral", 2: "positive"}

        bs = int(self.batch_size)
        while True:
            try:
                out: List[Dict[str, float]] = []
                use_amp = torch.cuda.is_available()

                with torch.inference_mode():
                    for i in range(0, len(texts), bs):
                        chunk = texts[i : i + bs]
                        enc = self.tokenizer(
                            chunk,
                            padding=True,
                            truncation=True,
                            max_length=int(self.max_length),
                            return_tensors="pt",
                        )
                        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                                logits = self.model(**enc).logits
                        else:
                            logits = self.model(**enc).logits

                        probs = torch.softmax(logits, dim=-1).detach().cpu()

                        for row in probs:
                            d = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
                            for j, p in enumerate(row.tolist()):
                                key = idx_to_key.get(j)
                                if key in d:
                                    d[key] = float(p)
                            out.append(d)

                return out

            except torch.OutOfMemoryError:
                if not torch.cuda.is_available():
                    raise
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                bs = max(1, bs // 2)
                if bs == 1:
                    raise


# ======================================================================================
# Stocks -> regexes + mappings
# ======================================================================================
AMBIGUOUS_TICKERS: Set[str] = {
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "AN","AS","AT","BE","BY","DO","FOR","FROM","GO","IF","IN","IS","IT","NO","NOT","OF","ON","OR","SO","THE",
    "TO","UP","US","WE","YOU","HE","SHE","THEY","HARD","HIGH","GOOD","WAY","ARE","CAN","ALL","AM","ANY","AWAY",
    "BEAT","USE","POST","LOW","NEW","OLD","NOW","PAY","RUN","SEE","SET","TOP","WIN","YES","BUY","SELL","CALL",
    "PUT","FUND","FUNDS","BOND","BONDS","CASH","COST","RATE","JUST","LOT","WEEK","PLAY","HERE","BULL","BEAR",
    "OUT","LIFE","HELP","HOPE","LIKE","LOVE","GLAD","WWW","NEXT","SURE","NICE","GAIN","COM","VEGA","NEAR","TINY",
    "TALK","KNOW","MADE","ELSE","GROW","BIT","TRIP","TIME","SAY","HAS","DIVE","MIND","WANT","FUN","SPAM","SITE",
    "EM","OPEN","EASY","EDIT","ITM","OTM","TWO","MOVE","MAN","GL","REAL","NOTE","LEAD","YEAR","HOUR","ODDS",
    "RARE","AGO","EVER","WELL","MINE","EDGE","FAN","MAX","MIN","HOLD","SEMI","YOLO","FOMO","DD","HODL","MOON",
    "ATH","PT","IV","IMO","IMHO","TLDR","AMA","ADD","SHORT","LONG","BULLS","BEARS","PUMP","DUMP","HEDGE","RIP",
    "RUN","APE","BAG","LOL","WEN",
}

COMPANY_ALIASES: Dict[str, str] = {
    "google": "GOOG",
    "facebook": "META",
    "meta": "META",
    "instagram": "META",
    "tsmc": "TSM",
    "lucid": "LCID",
    "lucid motors": "LCID",
    "rivian": "RIVN",
    "paypal": "PYPL",
    "venmo": "PYPL",
    "square": "SQ",
    "robinhood": "HOOD",
    "coinbase": "COIN",
    "disney": "DIS",
    "warner bros": "WBD",
    "warner brothers": "WBD",
    "warner": "WBD",
    "costco": "COST",
    "nike": "NKE",
    "chipotle": "CMG",
    "game stop": "GME",
}

def _normalize_company_key(name: str) -> str:
    return " ".join(str(name).strip().lower().split())

def _is_ambiguous_ticker(tkr: str) -> bool:
    return str(tkr).upper() in AMBIGUOUS_TICKERS

def _load_stocks():
    global _stocks_df, _symbol_regex, _company_regex, _cashtag_regex
    global _symbol_col_name, _company_col_name, _name_to_symbol

    if _stocks_df is not None:
        return _stocks_df

    try:
        df = pd.read_parquet(STOCKS_PARQUET)
    except Exception as e:
        df = pd.DataFrame(columns=["symbol", "name"])
        logger.warning("could not read stocks parquet at %s: %s", STOCKS_PARQUET, e)

    cols = {c.lower(): c for c in df.columns}
    _symbol_col_name = cols.get("symbol") or cols.get("ticker")
    _company_col_name = cols.get("name") or cols.get("company_name")

    symbols: List[str] = []
    company_names: List[str] = []

    if _symbol_col_name and _symbol_col_name in df.columns:
        symbols = df[_symbol_col_name].dropna().astype(str).str.upper().unique().tolist()

    _name_to_symbol = {}
    if _company_col_name and _symbol_col_name and _company_col_name in df.columns and _symbol_col_name in df.columns:
        tmp = df[[_company_col_name, _symbol_col_name]].dropna()
        for comp, sym in tmp.itertuples(index=False):
            ck = _normalize_company_key(comp)
            if ck:
                _name_to_symbol.setdefault(ck, str(sym).upper())

        company_names = tmp[_company_col_name].astype(str).str.strip().unique().tolist()
        company_names = [n for n in company_names if 2 <= len(n) <= 100]

    for alias_name, alias_sym in COMPANY_ALIASES.items():
        _name_to_symbol.setdefault(_normalize_company_key(alias_name), alias_sym.upper())

    alias_company_names = [name for name in COMPANY_ALIASES.keys() if 2 <= len(name) <= 100]
    company_names = list(set(company_names + alias_company_names))

    _stocks_df = df
    _cashtag_regex = re.compile(r"(?P<cashtag>\$[A-Z]{1,6}\b)")

    if symbols:
        symbols_sorted = sorted(set(symbols), key=lambda s: -len(s))
        symbols_escaped = [re.escape(s) for s in symbols_sorted]
        symbol_pattern = r"\b(?P<symbol>(" + "|".join(symbols_escaped) + r"))\b"
        _symbol_regex = re.compile(symbol_pattern, flags=re.IGNORECASE)
    else:
        _symbol_regex = re.compile(r"$^")

    if company_names:
        names_sorted = sorted(set(company_names), key=lambda n: -len(n))[:300]
        names_escaped = [re.escape(n) for n in names_sorted if n.strip()]
        company_pattern = r"\b(?P<company>(" + "|".join(names_escaped) + r"))\b"
        try:
            _company_regex = re.compile(company_pattern, flags=re.IGNORECASE)
        except re.error:
            _company_regex = None
    else:
        _company_regex = None

    return _stocks_df

def extract_tickers_from_text(text: str) -> List[str]:
    if not text:
        return []
    _load_stocks()

    tickers: Set[str] = set()
    for m in _cashtag_regex.finditer(text):
        tickers.add(m.group("cashtag").lstrip("$").upper())

    for m in _symbol_regex.finditer(text):
        surface = m.group("symbol")
        sym = surface.upper()
        if _is_ambiguous_ticker(sym) and not surface.isupper():
            continue
        tickers.add(sym)

    if _company_regex and _name_to_symbol:
        for m in _company_regex.finditer(text):
            comp_surface = m.group("company").strip()
            sym = _name_to_symbol.get(_normalize_company_key(comp_surface))
            if sym:
                tickers.add(sym.upper())

    return sorted(tickers)


# ======================================================================================
# Depth computation
# ======================================================================================
def compute_depths(comments: List[Dict[str, Any]]) -> Dict[str, int]:
    by_base: Dict[str, Tuple[str, str]] = {}
    bases: List[str] = []

    for c in comments:
        cid_base = _base_id(c.get("id"))
        if not cid_base:
            continue
        parent_id = _normalize_fullname(c.get("parent_id"))
        link_id = _normalize_fullname(c.get("link_id"))
        by_base[cid_base] = (parent_id, link_id)
        bases.append(cid_base)

    depths: Dict[str, int] = {}
    for cid_base in bases:
        if cid_base in depths:
            continue

        stack: List[str] = []
        cur = cid_base
        while True:
            if cur in depths:
                base_depth = depths[cur]
                break

            pl = by_base.get(cur)
            if pl is None:
                base_depth = 0
                break

            parent_id, link_id = pl

            if parent_id and link_id and parent_id == link_id:
                base_depth = 0
                break
            if _is_submission_fullname(parent_id):
                base_depth = 0
                break

            parent_base = _base_id(parent_id)
            if not parent_base or parent_base not in by_base:
                base_depth = 0
                break

            stack.append(cur)
            cur = parent_base

        d = base_depth
        while stack:
            child = stack.pop()
            d = min(MAX_DEPTH, d + 1)
            depths[child] = d

        depths.setdefault(cid_base, base_depth)

    return depths


# ======================================================================================
# Submission indexing
# ======================================================================================
def build_sub_index(submissions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for s in submissions:
        sid = str(s.get("id", "")).strip()
        if not sid:
            continue
        if sid.startswith("t3_"):
            idx[sid[3:]] = s
            idx[sid] = s
        else:
            idx[sid] = s
            idx[f"t3_{sid}"] = s
    return idx


def _make_submission_pseudo_rows_one(sub: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not sub:
        return []
    sid_base = _base_id(sub.get("id"))
    if not sid_base:
        return []

    author = sub.get("author") if sub.get("author") is not None else "[deleted]"
    title = _clean_text(sub.get("title") or "")
    selftext = _clean_text(sub.get("selftext") or "")

    pseudo: List[Dict[str, Any]] = []
    if title:
        pseudo.append(
            {
                "submission_id": sid_base,
                "ticker_text": title,
                "author": author,
                "comment_score": 0.0,
                "depth": 0,
                "weight": 1.0 * TITLE_WEIGHT_MULT,
                "kind": "submission_title",
            }
        )

    if selftext:
        pseudo.append(
            {
                "submission_id": sid_base,
                "ticker_text": selftext,
                "author": author,
                "comment_score": 0.0,
                "depth": 0,
                "weight": 1.0 * SELFTEXT_WEIGHT_MULT,
                "kind": "submission_selftext",
            }
        )

    return pseudo


def _submission_passes_min_activity(sub: Optional[Dict[str, Any]]) -> bool:
    if not sub:
        return False
    n_comments = sub.get("num_comments")
    score = sub.get("score")
    n_comments_i = int(n_comments) if n_comments is not None else 0
    score_i = int(score) if score is not None else 0
    return (n_comments_i >= MIN_SUBMISSION_COMMENTS and score_i >= MIN_SUBMISSION_SCORE)


def _get_submission_created_utc_epoch(sub: Optional[Dict[str, Any]]) -> Optional[float]:
    if not sub:
        return None

    sid = _base_id(sub.get("id"))
    if sid and sid in _sub_created_cache:
        return _sub_created_cache[sid]

    created_src = sub.get("created_utc")
    if created_src is None:
        created_src = sub.get("created_ts")

    epoch = _fast_parse_timestamp_to_epoch_utc(created_src)
    if sid:
        _sub_created_cache[sid] = epoch
    return epoch


# ======================================================================================
# Weighting
# ======================================================================================
def _safe_pow(base: float, exp: float) -> float:
    base = float(base)
    exp = float(exp)
    if base <= 0:
        return 0.0
    return base ** exp


def combined_comment_weight(*, order_index: int, hours_since_submission: Optional[float], depth: int) -> float:
    if ORDER_DECAY_EVERY_N > 0:
        order_exp = float(order_index) / float(ORDER_DECAY_EVERY_N)
    else:
        order_exp = float(order_index)
    order_factor = _safe_pow(ORDER_DECAY_BASE, order_exp)

    if hours_since_submission is None:
        time_factor = 1.0
    else:
        if TIME_DECAY_EVERY_HOURS > 0:
            time_exp = float(hours_since_submission) / float(TIME_DECAY_EVERY_HOURS)
        else:
            time_exp = float(hours_since_submission)
        time_factor = _safe_pow(TIME_DECAY_BASE, time_exp)

    depth_factor = _safe_pow(DEPTH_DECAY_BASE, int(depth))
    return float(order_factor) * float(time_factor) * float(depth_factor)


# ======================================================================================
# Aggregation
# ======================================================================================
def _agg_init(trading_date_et: str, ticker: str) -> Dict[str, Any]:
    return {
        "trading_date_et": trading_date_et,
        "ticker": ticker,
        "n_items": 0,
        "authors": set(),
        "comment_score_sum": 0.0,
        "comment_score_sum_comments_only": 0.0,
        "n_comment_items": 0,
        "comment_score_max": float("-inf"),
        "depth_sum": 0.0,
        "n_from_comments": 0,
        "n_from_title": 0,
        "n_from_selftext": 0,
        "weight_sum": 0.0,
        "fin_neg_wsum": 0.0,
        "fin_neu_wsum": 0.0,
        "fin_pos_wsum": 0.0,
        "tw_neg_wsum": 0.0,
        "tw_neu_wsum": 0.0,
        "tw_pos_wsum": 0.0,
    }


def _agg_update(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    trading_date_et: str,
    ticker: str,
    author: str,
    comment_score: float,
    depth: int,
    weight: float,
    kind: str,
    fin_scores: Dict[str, float],
    tw_scores: Dict[str, float],
):
    key = (trading_date_et, ticker)
    a = agg.get(key)
    if a is None:
        a = _agg_init(trading_date_et, ticker)
        agg[key] = a

    a["n_items"] += 1
    a["authors"].add(author)
    a["comment_score_sum"] += float(comment_score)

    if kind == "comment":
        a["n_comment_items"] += 1
        a["comment_score_sum_comments_only"] += float(comment_score)
        a["comment_score_max"] = max(a["comment_score_max"], float(comment_score))

    a["depth_sum"] += float(depth)

    if kind == "comment":
        a["n_from_comments"] += 1
    elif kind == "submission_title":
        a["n_from_title"] += 1
    elif kind == "submission_selftext":
        a["n_from_selftext"] += 1

    w = float(weight)
    a["weight_sum"] += w

    a["fin_neg_wsum"] += fin_scores["negative"] * w
    a["fin_neu_wsum"] += fin_scores["neutral"] * w
    a["fin_pos_wsum"] += fin_scores["positive"] * w
    a["tw_neg_wsum"] += tw_scores["negative"] * w
    a["tw_neu_wsum"] += tw_scores["neutral"] * w
    a["tw_pos_wsum"] += tw_scores["positive"] * w


def _passes_min_items_filter(a: Dict[str, Any]) -> bool:
    n_comment = int(a.get("n_comment_items", 0) or 0)
    n_total = int(a.get("n_items", 0) or 0)

    if int(MIN_TICKER_COMMENT_ITEMS) > 0 and n_comment < int(MIN_TICKER_COMMENT_ITEMS):
        return False
    if int(MIN_TICKER_TOTAL_ITEMS) > 0 and n_total < int(MIN_TICKER_TOTAL_ITEMS):
        return False
    return True


def _iter_agg_rows(agg: Dict[Tuple[str, str], Dict[str, Any]], date_feats_cache: Dict[str, Dict[str, Any]]):
    for (trading_date_et, ticker), a in agg.items():
        if not _passes_min_items_filter(a):
            continue

        n = int(a["n_items"])
        weight_sum = float(a["weight_sum"])
        denom = weight_sum if weight_sum != 0.0 else float("nan")

        n_comment_items = int(a["n_comment_items"])
        ticker_comment_score_max = float(a["comment_score_max"])
        ticker_comment_score_mean = (
            float(a["comment_score_sum_comments_only"]) / n_comment_items if n_comment_items else float("nan")
        )

        row = {
            "trading_date_et": trading_date_et,
            "ticker": ticker,
            **date_feats_cache.get(trading_date_et, {}),

            "n_items": n,
            "n_authors": int(len(a["authors"])),

            "comment_score_sum": float(a["comment_score_sum"]),
            "ticker_comment_score_max": ticker_comment_score_max,
            "ticker_comment_score_mean": ticker_comment_score_mean,

            "avg_depth": float(a["depth_sum"]) / n if n else float("nan"),
            "n_from_comments": int(a["n_from_comments"]),
            "n_from_title": int(a["n_from_title"]),
            "n_from_selftext": int(a["n_from_selftext"]),
            "n_comment_items": n_comment_items,

            "fin_neg_sum_w": float(a["fin_neg_wsum"]),
            "fin_neu_sum_w": float(a["fin_neu_wsum"]),
            "fin_pos_sum_w": float(a["fin_pos_wsum"]),
            "tw_neg_sum_w": float(a["tw_neg_wsum"]),
            "tw_neu_sum_w": float(a["tw_neu_wsum"]),
            "tw_pos_sum_w": float(a["tw_pos_wsum"]),

            "fin_neg_mean_w": float(a["fin_neg_wsum"]) / denom if weight_sum else float("nan"),
            "fin_neu_mean_w": float(a["fin_neu_wsum"]) / denom if weight_sum else float("nan"),
            "fin_pos_mean_w": float(a["fin_pos_wsum"]) / denom if weight_sum else float("nan"),
            "tw_neg_mean_w": float(a["tw_neg_wsum"]) / denom if weight_sum else float("nan"),
            "tw_neu_mean_w": float(a["tw_neu_wsum"]) / denom if weight_sum else float("nan"),
            "tw_pos_mean_w": float(a["tw_pos_wsum"]) / denom if weight_sum else float("nan"),
        }
        yield row


# ======================================================================================
# Build a trading_stream directly from in-memory comment rows
# ======================================================================================
def iter_comments_grouped_from_memory(
    comments: List[Dict[str, Any]],
) -> Iterator[Tuple[str, Optional[str], str, List[Dict[str, Any]]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for c in comments or []:
        tday = c.get("trading_date_et")
        if not tday:
            continue

        link_id = c.get("link_id")
        if not link_id:
            continue

        sid = _base_id(link_id)
        if not sid:
            continue

        if c.get("created_utc") is None:
            c_created = c.get("created_ts") or c.get("created")
            epoch = _fast_parse_timestamp_to_epoch_utc(c_created)
            if epoch is not None:
                c["created_utc"] = float(epoch)

        groups[(str(tday), str(sid))].append(c)

    for (tday, sid) in sorted(groups.keys()):
        yield (tday, None, sid, groups[(tday, sid)])


# ======================================================================================
# Model aggregation processing
# ======================================================================================
def _process_items_chunk_datasets(
    fin_runner: _ModelRunner,
    tw_runner: _ModelRunner,
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    items: List[Dict[str, Any]],
    *,
    dedupe_texts_within_chunk: bool = True,
):
    if not items:
        return

    filtered: List[Dict[str, Any]] = []
    for it in items:
        if float(it.get("weight", 1.0)) < MIN_EFFECTIVE_WEIGHT:
            continue
        tickers = extract_tickers_from_text(it["ticker_text"])
        if not tickers:
            continue
        it2 = dict(it)
        it2["_tickers"] = tickers
        filtered.append(it2)

    if not filtered:
        return

    texts = [it["ticker_text"] for it in filtered]

    if dedupe_texts_within_chunk:
        text_to_indices: Dict[str, List[int]] = {}
        unique_texts: List[str] = []
        for i, t in enumerate(texts):
            if t not in text_to_indices:
                text_to_indices[t] = []
                unique_texts.append(t)
            text_to_indices[t].append(i)

        fin_unique = fin_runner.predict_proba_3way(unique_texts)
        tw_unique = tw_runner.predict_proba_3way(unique_texts)

        fin_scores_all = [None] * len(texts)
        tw_scores_all = [None] * len(texts)

        for ut, fin_s, tw_s in zip(unique_texts, fin_unique, tw_unique):
            for i in text_to_indices[ut]:
                fin_scores_all[i] = fin_s
                tw_scores_all[i] = tw_s
    else:
        fin_scores_all = fin_runner.predict_proba_3way(texts)
        tw_scores_all = tw_runner.predict_proba_3way(texts)

    for it, fin_s, tw_s in zip(filtered, fin_scores_all, tw_scores_all):
        fin_s = fin_s or {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        tw_s = tw_s or {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

        trading_date_et = it.get("trading_date_et")
        if not trading_date_et:
            continue

        for tkr in it["_tickers"]:
            _agg_update(
                agg=agg,
                trading_date_et=str(trading_date_et),
                ticker=str(tkr).upper(),
                author=it["author"],
                comment_score=float(it["comment_score"]),
                depth=int(it["depth"]),
                weight=float(it["weight"]),
                kind=it["kind"],
                fin_scores=fin_s,
                tw_scores=tw_s,
            )


# ======================================================================================
# Core: build observations -> S3 batches (parquet only)
# ======================================================================================
class _ObservationWriter:
    def __init__(
        self,
        *,
        s3_bucket: str,
        s3_prefix: str,
        rows_per_file: int = OUTPUT_ROWS_PER_FILE_DEFAULT,
        file_prefix: str = OUTPUT_FILE_PREFIX_DEFAULT,
        sort_before_write: bool = SORT_BEFORE_WRITE_DEFAULT,
    ) -> None:
        _load_stocks()

        fin_tok, fin_model = _get_finbert_model()
        tw_tok, tw_model = _get_twitter_roberta_model()
        self.fin_runner = _ModelRunner(tokenizer=fin_tok, model=fin_model, batch_size=BATCH_SIZE_FINBERT)
        self.tw_runner = _ModelRunner(tokenizer=tw_tok, model=tw_model, batch_size=BATCH_SIZE_TWITTER)

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.rows_per_file = int(rows_per_file)
        self.file_prefix = file_prefix
        self.sort_before_write = bool(sort_before_write)

        self.written_keys: List[str] = []
        self.batch_rows: List[Dict[str, Any]] = []
        self.batch_idx = 0

        self.cur_day: Optional[str] = None
        self.agg_bucket: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.feats_bucket: Optional[Dict[str, Any]] = None
        self.day_min_created_utc_epoch: Optional[float] = None

    def _flush_bucket(self, trading_date_et: str) -> None:
        if self.feats_bucket is None:
            date_feats_cache = {}
        else:
            nsubs = int(self.feats_bucket.get("day_n_submissions", 0)) or 0
            self.feats_bucket["day_submission_score_mean"] = (
                float(self.feats_bucket.get("day_submission_score_sum", 0.0)) / nsubs if nsubs else float("nan")
            )
            self.feats_bucket["day_submission_num_comments_mean"] = (
                float(self.feats_bucket.get("day_submission_num_comments_sum", 0)) / nsubs if nsubs else float("nan")
            )

            if self.day_min_created_utc_epoch is not None:
                self.feats_bucket["window_from_et"] = _epoch_utc_to_et_iso(self.day_min_created_utc_epoch)

            date_feats_cache = {trading_date_et: self.feats_bucket}

        if self.agg_bucket:
            rows_iterable = list(_iter_agg_rows(self.agg_bucket, date_feats_cache))
            if self.sort_before_write:
                rows_iterable.sort(key=lambda r: (r.get("trading_date_et", ""), r.get("ticker", "")))

            for row in rows_iterable:
                self.batch_rows.append(row)
                if len(self.batch_rows) >= self.rows_per_file:
                    key = _write_batch_rows_to_s3(
                        self.batch_rows,
                        bucket=self.s3_bucket,
                        prefix=self.s3_prefix,
                        batch_idx=self.batch_idx,
                        file_prefix=self.file_prefix,
                    )
                    self.written_keys.append(key)
                    self.batch_rows.clear()
                    self.batch_idx += 1

        self.agg_bucket.clear()
        self.feats_bucket = None
        self.day_min_created_utc_epoch = None

    def process_trading_stream(
        self,
        trading_stream: Iterable[Tuple[str, Optional[str], str, List[Dict[str, Any]]]],
        sub_idx: Dict[str, Dict[str, Any]],
    ) -> None:
        for trading_date_et, window_from_et_hint, submission_id, sub_comments in trading_stream:
            if self.cur_day is None:
                self.cur_day = trading_date_et
            elif trading_date_et != self.cur_day:
                self._flush_bucket(self.cur_day)
                self.cur_day = trading_date_et

            sub = sub_idx.get(submission_id) or sub_idx.get(f"t3_{submission_id}")
            if not _submission_passes_min_activity(sub):
                continue

            if self.feats_bucket is None:
                self.feats_bucket = {
                    "trading_date_et": trading_date_et,
                    "window_from_et": window_from_et_hint,
                    "day_submission_score_sum": 0.0,
                    "day_submission_num_comments_sum": 0,
                    "day_n_submissions": 0,
                }

            self.feats_bucket["day_n_submissions"] += 1
            if sub and sub.get("score") is not None:
                self.feats_bucket["day_submission_score_sum"] += float(sub.get("score"))
            if sub and sub.get("num_comments") is not None:
                self.feats_bucket["day_submission_num_comments_sum"] += int(sub.get("num_comments"))

            sub_created_epoch = _get_submission_created_utc_epoch(sub)
            depths_by_base = compute_depths(sub_comments)

            eligible: List[Tuple[Optional[float], Dict[str, Any], str, int]] = []
            for c in sub_comments:
                body = _clean_text(c.get("body") or "")
                if not body:
                    continue

                created_epoch = c.get("created_utc")
                if created_epoch is None:
                    created_epoch = _fast_parse_timestamp_to_epoch_utc(c.get("created_ts"))
                if created_epoch is None:
                    created_epoch = _fast_parse_timestamp_to_epoch_utc(c.get("created"))
                if created_epoch is not None:
                    created_epoch = float(created_epoch)
                    if self.day_min_created_utc_epoch is None or created_epoch < self.day_min_created_utc_epoch:
                        self.day_min_created_utc_epoch = created_epoch

                cid_base = _base_id(c.get("id"))
                depth = depths_by_base.get(cid_base, 0)
                eligible.append((created_epoch, c, body, depth))

            eligible.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else float("inf")))

            if eligible:
                chunk_items: List[Dict[str, Any]] = []
                for order_idx, (created_epoch, c, body, depth) in enumerate(eligible):
                    hours_since = None
                    if sub_created_epoch is not None and created_epoch is not None:
                        try:
                            hours_since = (float(created_epoch) - float(sub_created_epoch)) / 3600.0
                            if hours_since < 0:
                                hours_since = 0.0
                        except Exception:
                            hours_since = None

                    w = combined_comment_weight(order_index=order_idx, hours_since_submission=hours_since, depth=depth)
                    if w < MIN_EFFECTIVE_WEIGHT:
                        continue

                    chunk_items.append(
                        {
                            "trading_date_et": trading_date_et,
                            "ticker_text": body,
                            "author": c.get("author") if c.get("author") is not None else "[deleted]",
                            "comment_score": float(c.get("score")) if c.get("score") is not None else 0.0,
                            "depth": depth,
                            "weight": w,
                            "kind": "comment",
                        }
                    )

                    if len(chunk_items) >= CHUNK_SIZE_ITEMS:
                        _process_items_chunk_datasets(self.fin_runner, self.tw_runner, self.agg_bucket, chunk_items)
                        chunk_items = []

                if chunk_items:
                    _process_items_chunk_datasets(self.fin_runner, self.tw_runner, self.agg_bucket, chunk_items)

            pseudo_items = _make_submission_pseudo_rows_one(sub)
            for p in pseudo_items:
                p["trading_date_et"] = trading_date_et

            for pseudo_chunk in _chunks(pseudo_items, CHUNK_SIZE_ITEMS):
                _process_items_chunk_datasets(self.fin_runner, self.tw_runner, self.agg_bucket, pseudo_chunk)

    def finish(self) -> List[str]:
        if self.cur_day is not None:
            self._flush_bucket(self.cur_day)
            self.cur_day = None

        if self.batch_rows:
            key = _write_batch_rows_to_s3(
                self.batch_rows,
                bucket=self.s3_bucket,
                prefix=self.s3_prefix,
                batch_idx=self.batch_idx,
                file_prefix=self.file_prefix,
            )
            self.written_keys.append(key)
            self.batch_rows.clear()
            self.batch_idx += 1

        return list(self.written_keys)


def build_observations_batched_to_s3(
    trading_stream: Iterable[Tuple[str, Optional[str], str, List[Dict[str, Any]]]],
    submissions: Optional[List[Dict[str, Any]]] = None,
    *,
    s3_bucket: str,
    s3_prefix: str,
    rows_per_file: int = OUTPUT_ROWS_PER_FILE_DEFAULT,
    file_prefix: str = OUTPUT_FILE_PREFIX_DEFAULT,
    sort_before_write: bool = SORT_BEFORE_WRITE_DEFAULT,
) -> List[str]:
    writer = _ObservationWriter(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        rows_per_file=rows_per_file,
        file_prefix=file_prefix,
        sort_before_write=sort_before_write,
    )
    writer.process_trading_stream(trading_stream, build_sub_index(submissions or []))
    return writer.finish()


# ======================================================================================
# NEW: Trading-day check + S3 manifest(JSON) -> parquet(keys) -> read -> process -> move
# ======================================================================================
def _get_today_local_date_str(now_utc: datetime, tz_name: str) -> str:
    if ZoneInfo is None:
        return now_utc.date().isoformat()
    try:
        tz = ZoneInfo(tz_name)
        return now_utc.astimezone(tz).date().isoformat()
    except Exception:
        return now_utc.date().isoformat()


def _is_trading_day_today(now_utc: Optional[datetime] = None) -> bool:
    """
    Uses pandas_market_calendars to check if *today* (in MARKET_TZ) is a trading day.
    """
    now_utc = now_utc or datetime.now(timezone.utc)

    try:
        import pandas_market_calendars as mcal
    except Exception as e:
        # Safer default: treat as trading day so the pipeline doesn't silently stop
        logger.warning("pandas_market_calendars not available; defaulting to trading-day=True. err=%s", e)
        return True

    date_str = _get_today_local_date_str(now_utc, EASTERN_TZ)
    cal = mcal.get_calendar(TRADING_CALENDAR)
    sched = cal.schedule(start_date=date_str, end_date=date_str)
    return not sched.empty


def _parse_s3_trigger_records(event: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rec in (event.get("Records") or []):
        s3info = rec.get("s3") or {}
        b = ((s3info.get("bucket") or {}).get("name") or "").strip()
        k = ((s3info.get("object") or {}).get("key") or "").strip()
        if b and k:
            out.append((b, unquote_plus(k)))
    return out


def _list_s3_objects(bucket: str, prefix: str = "") -> List[str]:
    s3 = _get_s3()
    prefix = (prefix or "").lstrip("/")
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj.get("Key")
            if k:
                keys.append(k)
    return keys


def _read_json_object(bucket: str, key: str) -> Dict[str, Any]:
    s3 = _get_s3()
    resp = s3.get_object(Bucket=bucket, Key=key)
    data = resp["Body"].read()
    try:
        txt = data.decode("utf-8")
    except Exception:
        txt = data.decode("utf-8", errors="ignore")
    return json.loads(txt)


def _read_parquet_object_to_records(
    bucket: str,
    key: str,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    s3 = _get_s3()
    resp = s3.get_object(Bucket=bucket, Key=key)
    data = resp["Body"].read()
    buf = BytesIO(data)

    requested_columns = columns
    if columns:
        parquet_file = pq.ParquetFile(buf)
        available_columns = set(parquet_file.schema_arrow.names)
        requested_columns = [name for name in columns if name in available_columns]
        buf.seek(0)

    table = pq.read_table(buf, columns=requested_columns)
    return table.to_pylist()


def _dedupe_by_id(rows: List[Dict[str, Any]], seen: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    seen_ids: Set[str] = seen if seen is not None else set()
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        rid = str(r.get("id") or "").strip()
        if rid:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
        out.append(r)
    return out


def _archive_dest_key(src_key: str) -> str:
    now_utc = datetime.now(timezone.utc)
    day = _get_today_local_date_str(now_utc, EASTERN_TZ)
    ap = (ARCHIVE_PREFIX or "").lstrip("/")
    if ap and not ap.endswith("/"):
        ap += "/"
    return f"{ap}{day}/{src_key.lstrip('/')}"


def _move_s3_object(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    s3 = _get_s3()
    try:
        s3.copy_object(
            Bucket=dst_bucket,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404"}:
            logger.warning(
                "Skipping archive move for missing source object s3://%s/%s; assuming it was already moved.",
                src_bucket,
                src_key,
            )
            return
        raise
    s3.delete_object(Bucket=src_bucket, Key=src_key)


def _move_many_to_archive(src_bucket: str, keys: List[str]) -> List[Dict[str, str]]:
    if not ARCHIVE_BUCKET:
        raise RuntimeError("ARCHIVE_BUCKET env var is required to move processed input files.")
    moved: List[Dict[str, str]] = []
    for k in keys:
        dst_key = _archive_dest_key(k)
        _move_s3_object(src_bucket, k, ARCHIVE_BUCKET, dst_key)
        moved.append({"from": f"s3://{src_bucket}/{k}", "to": f"s3://{ARCHIVE_BUCKET}/{dst_key}"})
    return moved


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _extract_task_token(event: Dict[str, Any]) -> Optional[str]:
    if not isinstance(event, dict):
        return None
    token = event.get("taskToken") or event.get("TaskToken")
    if token is None:
        return None
    token = str(token).strip()
    return token or None


def _build_callback_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    output = result.get("output") or {}
    return {
        "skipped": bool(result.get("skipped")),
        "mode": result.get("mode"),
        "input_bucket": result.get("input_bucket"),
        "input_prefix": result.get("input_prefix"),
        "n_manifests": result.get("n_manifests"),
        "output": {
            "bucket": output.get("bucket"),
            "prefix": output.get("prefix"),
            "n_files": output.get("n_files"),
            "keys": output.get("keys") or [],
        },
    }


def _forward_processed_output(event: Dict[str, Any]) -> Dict[str, Any]:
    processed = event.get("processed_output") or event.get("output") or {}
    bucket = processed.get("bucket")
    keys = [str(k) for k in (processed.get("keys") or []) if k]
    if not bucket:
        raise ValueError("processed_output must include output.bucket.")
    if not keys:
        return {
            "skipped": True,
            "mode": "forward_processed_output",
            "reason": "no_processed_files",
            "output": {
                "bucket": bucket,
                "prefix": processed.get("prefix") or OBS_OUTPUT_S3_PREFIX,
                "n_files": 0,
                "keys": [],
            },
        }
    return {
        "skipped": False,
        "mode": "forward_processed_output",
        "output": {
            "bucket": bucket,
            "prefix": processed.get("prefix") or OBS_OUTPUT_S3_PREFIX,
            "n_files": int(processed.get("n_files") or len(keys)),
            "keys": keys,
        },
    }


def _is_manifest_key(key: str) -> bool:
    k = (key or "").split("/")[-1]
    return k.startswith(MANIFEST_PREFIX) and k.endswith(MANIFEST_SUFFIX)


def _is_monthly_ready_key(key: str) -> bool:
    return (key or "").endswith("/monthly_ready.json") or (key or "") == "monthly_ready.json"


def _delete_s3_object_if_exists(bucket: str, key: str) -> bool:
    s3 = _get_s3()
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404"}:
            logger.warning("Skipping delete for missing object s3://%s/%s.", bucket, key)
            return False
        raise


def _collect_manifest_keys_to_process(
    trigger_bucket: str,
    trigger_key: str,
) -> Tuple[str, str, List[str]]:
    """
    Returns (input_bucket, input_prefix, manifest_keys_to_process).

    Behavior:
      - input_bucket = INPUT_BUCKET override or trigger_bucket
      - input_prefix = INPUT_PREFIX override or trigger object's directory
      - manifest keys = all manifest_*.json under that prefix + ensure trigger is included if it is a manifest
    """
    input_bucket = (INPUT_BUCKET or trigger_bucket).strip()

    if INPUT_PREFIX:
        input_prefix = INPUT_PREFIX.lstrip("/")
    else:
        input_prefix = trigger_key.rsplit("/", 1)[0] + "/" if "/" in trigger_key else ""

    listed = _list_s3_objects(input_bucket, input_prefix)
    manifest_keys = [k for k in listed if _is_manifest_key(k)]

    if input_bucket == trigger_bucket and _is_manifest_key(trigger_key):
        if trigger_key not in manifest_keys:
            manifest_keys.append(trigger_key)

    manifest_keys = sorted(set(manifest_keys))
    return input_bucket, input_prefix, manifest_keys


# ======================================================================================
# Lambda handler
# ======================================================================================
def _run_lambda(event, context):
    action = str((event or {}).get("action") or "").strip().lower()
    if action == "forward_processed_output":
        return _forward_processed_output(event)

    # 1) S3-trigger: read manifests (json), then read parquet keys referenced in manifests
    records = _parse_s3_trigger_records(event)
    if records:
        trigger_bucket, trigger_key = records[0]

        input_bucket, input_prefix, manifest_keys = _collect_manifest_keys_to_process(trigger_bucket, trigger_key)

        logger.info(
            "S3 trigger: bucket=%s key=%s; using input bucket=%s prefix=%s; found %d manifest(s)",
            trigger_bucket, trigger_key, input_bucket, input_prefix, len(manifest_keys)
        )

        if not manifest_keys:
            logger.warning("No manifest files found to process under s3://%s/%s", input_bucket, input_prefix)
            return {
                "skipped": False,
                "mode": "s3_trigger",
                "input_bucket": input_bucket,
                "input_prefix": input_prefix,
                "n_manifests": 0,
                "output": None,
            }

        # Track ALL S3 objects we should move after processing:
        # - manifests
        # - the parquet objects referenced by those manifests
        to_move: Set[str] = set(manifest_keys)

        # Output settings: parquet only
        out = event.get("output") or {}
        out_bucket = out.get("bucket") or OBS_OUTPUT_S3_BUCKET
        out_prefix = out.get("prefix") or OBS_OUTPUT_S3_PREFIX
        if not out_bucket:
            raise RuntimeError("Missing output bucket. Set event.output.bucket or OBS_OUTPUT_S3_BUCKET env var.")

        writer = _ObservationWriter(
            s3_bucket=out_bucket,
            s3_prefix=out_prefix,
            rows_per_file=int(out.get("rows_per_file") or OUTPUT_ROWS_PER_FILE_DEFAULT),
            file_prefix=str(out.get("file_prefix") or OUTPUT_FILE_PREFIX_DEFAULT),
            sort_before_write=bool(out.get("sort_before_write") if "sort_before_write" in out else SORT_BEFORE_WRITE_DEFAULT),
        )
        seen_submission_ids: Set[str] = set()
        seen_comment_ids: Set[str] = set()
        total_submission_rows = 0
        total_comment_rows = 0

        for idx, mk in enumerate(manifest_keys, start=1):
            manifest = _read_json_object(input_bucket, mk)

            subs_key = manifest.get("submissions_key")
            comm_key = manifest.get("comments_key")
            manifest_bucket = (manifest.get("bucket") or input_bucket).strip()

            if not subs_key or not comm_key:
                logger.warning("Manifest missing submissions_key/comments_key: s3://%s/%s", input_bucket, mk)
                continue

            if manifest_bucket == input_bucket:
                to_move.add(subs_key)
                to_move.add(comm_key)

            subs_rows = _dedupe_by_id(
                _read_parquet_object_to_records(manifest_bucket, subs_key, SUBMISSION_PARQUET_COLUMNS),
                seen_submission_ids,
            )
            comm_rows = _dedupe_by_id(
                _read_parquet_object_to_records(manifest_bucket, comm_key, COMMENT_PARQUET_COLUMNS),
                seen_comment_ids,
            )

            total_submission_rows += len(subs_rows)
            total_comment_rows += len(comm_rows)

            writer.process_trading_stream(
                iter_comments_grouped_from_memory(comm_rows),
                build_sub_index(subs_rows),
            )

            logger.info(
                "Processed manifest %d/%d (%s): %d submissions, %d comments after dedupe",
                idx,
                len(manifest_keys),
                mk,
                len(subs_rows),
                len(comm_rows),
            )

        keys_written = writer.finish()
        logger.info(
            "Finished batch prefix=%s with %d manifests, %d unique submissions, %d unique comments, %d output files",
            input_prefix,
            len(manifest_keys),
            total_submission_rows,
            total_comment_rows,
            len(keys_written),
        )

        # Move ONLY objects that live in the input bucket (per your requirement).
        # (If manifests referenced parquet in another bucket, we did not move them.)
        move_keys = sorted({k for k in to_move if isinstance(k, str) and k})
        moved = _move_many_to_archive(input_bucket, move_keys)

        deleted_monthly_trigger = False
        if trigger_bucket == input_bucket and _is_monthly_ready_key(trigger_key):
            deleted_monthly_trigger = _delete_s3_object_if_exists(trigger_bucket, trigger_key)

        return {
            "skipped": False,
            "mode": "s3_trigger",
            "input_bucket": input_bucket,
            "input_prefix": input_prefix,
            "n_manifests": len(manifest_keys),
            "n_input_objects_moved": len(move_keys),
            "moved": moved,
            "deleted_monthly_trigger": deleted_monthly_trigger,
            "output": {
                "bucket": out_bucket,
                "prefix": out_prefix,
                "n_files": len(keys_written),
                "keys": keys_written,
            },
        }

    # 3) Direct-invoke mode (backwards compatible): event already contains lists
    submissions = event.get("submissions") or []
    comments = event.get("comments") or []

    out = event.get("output") or {}
    out_bucket = out.get("bucket") or OBS_OUTPUT_S3_BUCKET
    out_prefix = out.get("prefix") or OBS_OUTPUT_S3_PREFIX
    if not out_bucket:
        raise RuntimeError("Missing output bucket. Set event.output.bucket or OBS_OUTPUT_S3_BUCKET env var.")

    trading_stream = iter_comments_grouped_from_memory(comments)

    keys = build_observations_batched_to_s3(
        trading_stream,
        submissions=submissions,
        s3_bucket=out_bucket,
        s3_prefix=out_prefix,
        rows_per_file=int(out.get("rows_per_file") or OUTPUT_ROWS_PER_FILE_DEFAULT),
        file_prefix=str(out.get("file_prefix") or OUTPUT_FILE_PREFIX_DEFAULT),
        sort_before_write=bool(out.get("sort_before_write") if "sort_before_write" in out else SORT_BEFORE_WRITE_DEFAULT),
    )

    return {
        "skipped": False,
        "mode": "direct_invoke",
        "bucket": out_bucket,
        "prefix": out_prefix,
        "n_files": len(keys),
        "keys": keys,
    }


def lambda_handler(event, context):
    task_token = _extract_task_token(event)
    try:
        result = _run_lambda(event, context)
    except Exception as exc:
        if not task_token:
            raise
        _get_sfn().send_task_failure(
            taskToken=task_token,
            error=type(exc).__name__[:256],
            cause=str(exc)[:32768],
        )
        logger.exception("Sent Step Functions task failure callback.")
        return {
            "status": "stepfunctions_task_failure_sent",
            "error": type(exc).__name__,
            "message": str(exc),
        }

    if task_token:
        callback_payload = _build_callback_payload(result)
        _get_sfn().send_task_success(
            taskToken=task_token,
            output=json.dumps(callback_payload, default=_json_default),
        )
        logger.info("Sent Step Functions task success callback.")
    return result
