import io
import json
import os
import urllib.parse
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

s3 = boto3.client("s3")
sns = boto3.client("sns")

TARGET_COL = "risk_premium_pct"
TICKER_RETURN_COL = "ticker_return_pct"
SP500_RETURN_COL = "sp500_return_pct"
PRED_COL = "predicted_risk_premium_pct"
PRED_TICKER_COL = "prediction_ticker"
PRED_DATE_COL = "prediction_trading_date"
PRED_RANK_COL = "prediction_rank_on_day"
PRED_POP_COL = "is_predicted_pop"
DATE_COL = os.environ.get("TRADING_DATE_COLUMN", "trading_date_et")
TICKER_COL = os.environ.get("TICKER_COLUMN", "ticker")
SP500_SYMBOL = os.environ.get("SP500_SYMBOL", "^GSPC")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", os.environ.get("DEST_BUCKET", ""))
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/reddit-risk-premium")
LATEST_MODEL_KEY = os.environ.get("LATEST_MODEL_KEY", f"{MODEL_PREFIX}/model_latest.pt")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", MODEL_BUCKET)
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "model-outputs")
BAKED_MODEL_ARTIFACT = os.environ.get("LOCAL_MODEL_ARTIFACT", "/opt/models/reddit-risk-premium/model_latest.pt")
ARCHIVE_BUCKET = os.environ.get("ARCHIVE_BUCKET", "")
ARCHIVE_PREFIX = os.environ.get("ARCHIVE_PREFIX", "archive/")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
SNS_ALERT_UPSIDE_THRESHOLD_PCT = float(os.environ.get("SNS_ALERT_UPSIDE_THRESHOLD_PCT", os.environ.get("SNS_ALERT_THRESHOLD_PCT", "10.0")))
SNS_ALERT_DOWNSIDE_THRESHOLD_PCT = float(os.environ.get("SNS_ALERT_DOWNSIDE_THRESHOLD_PCT", "-15.0"))
SNS_MAX_ROWS_IN_MESSAGE = int(os.environ.get("SNS_MAX_ROWS_IN_MESSAGE", "25"))
MIN_TRAIN_ROWS_FOR_ALERTS = int(os.environ.get("MIN_TRAIN_ROWS_FOR_ALERTS", "1000"))
TARGET_CLIP_ABS = float(os.environ.get("TARGET_CLIP_ABS", "100.0"))
PREDICTION_CLIP_ABS = float(os.environ.get("PREDICTION_CLIP_ABS", str(TARGET_CLIP_ABS)))

EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "100"))
LR = float(os.environ.get("TRAIN_LR", "0.001"))
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))
HIDDEN_DIM = int(os.environ.get("MODEL_HIDDEN_DIM", "64"))
TRAIN_RESUME_DEFAULT = os.environ.get("TRAIN_RESUME", "true").strip().lower() in ("1", "true", "yes", "y")


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_symbol(symbol: object) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.startswith("$"):
        raw = raw[1:]
    return raw


def _is_probable_ticker(symbol: object) -> bool:
    cleaned = _normalize_symbol(symbol)
    if cleaned == SP500_SYMBOL:
        return True
    if not cleaned:
        return False
    if len(cleaned) > 6:
        return False
    return cleaned.isalpha()


def _parse_s3_locations(event: Dict) -> List[Tuple[str, str]]:
    sources = event.get("sources") or []
    if sources:
        locations: List[Tuple[str, str]] = []
        for item in sources:
            if not isinstance(item, dict):
                continue
            bucket = item.get("bucket") or item.get("source_bucket")
            key = item.get("key") or item.get("source_key")
            if bucket and key:
                locations.append((str(bucket), str(key)))
        if locations:
            return locations

    if "source_bucket" in event and "source_key" in event:
        return [(event["source_bucket"], event["source_key"])]

    # Supports direct handoff from processing Lambda response:
    # {"output": {"bucket": "...", "keys": ["..."]}}
    out = event.get("output") or {}
    out_bucket = out.get("bucket")
    out_keys = out.get("keys") or []
    if out_bucket and out_keys:
        return [(out_bucket, str(key)) for key in out_keys]

    # Also support top-level shape from some processing modes:
    # {"bucket": "...", "keys": ["..."]}
    top_bucket = event.get("bucket")
    top_keys = event.get("keys") or []
    if top_bucket and top_keys:
        return [(top_bucket, str(key)) for key in top_keys]

    if "Records" in event and event["Records"]:
        record = event["Records"][0]
        src_bucket = record["s3"]["bucket"]["name"]
        src_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        return [(src_bucket, src_key)]

    raise ValueError("Unable to locate source S3 object in event.")


def _parse_s3_location(event: Dict) -> Tuple[str, str]:
    locations = _parse_s3_locations(event)
    return locations[-1]


def _read_df_from_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    buff = io.BytesIO(raw)

    lower_key = key.lower()
    if lower_key.endswith(".parquet"):
        return pd.read_parquet(buff)
    if lower_key.endswith(".csv"):
        return pd.read_csv(buff)
    if lower_key.endswith(".json"):
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame.from_records(payload)
        if isinstance(payload, dict):
            for k in ("data", "records", "rows", "observations", "items"):
                v = payload.get(k)
                if isinstance(v, list):
                    return pd.DataFrame.from_records(v)
            return pd.DataFrame([payload])
        raise ValueError(f"Unsupported JSON payload shape for key: {key}")

    raise ValueError(f"Unsupported input format for key: {key}")


def _write_df_to_s3_parquet(df: pd.DataFrame, bucket: str, key: str) -> str:
    out = io.BytesIO()
    df.to_parquet(out, index=False)
    out.seek(0)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=out.getvalue(),
        ContentType="application/octet-stream",
    )
    return key


def _fetch_close_to_close_returns(symbols: List[str], trading_dates: List[pd.Timestamp]) -> Dict[str, Dict[pd.Timestamp, float]]:
    if not symbols:
        return {}

    uniq_dates = sorted({pd.Timestamp(d).normalize() for d in trading_dates})
    start_dt = uniq_dates[0] - timedelta(days=15)
    end_dt = uniq_dates[-1] + timedelta(days=5)

    tickers = sorted({_normalize_symbol(symbol) for symbol in symbols if _is_probable_ticker(symbol)})
    returns_map: Dict[str, Dict[pd.Timestamp, float]] = {}

    for symbol in tickers:
        hist = yf.download(
            tickers=symbol,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if hist is None or hist.empty:
            returns_map[symbol] = {}
            continue

        closes = _extract_close_series(hist)
        if closes.empty:
            returns_map[symbol] = {}
            continue

        pct = closes.pct_change() * 100.0
        ret_lookup = {}
        for idx, val in pct.dropna().items():
            if isinstance(val, (pd.Series, pd.DataFrame)):
                continue
            ret_lookup[pd.Timestamp(idx).normalize()] = float(val)
        returns_map[symbol] = ret_lookup

    return returns_map


def _extract_close_series(hist: pd.DataFrame) -> pd.Series:
    close_candidates = ("Adj Close", "Close")

    if isinstance(hist, pd.Series):
        return hist.dropna().sort_index()

    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return pd.Series(dtype="float64")

    if isinstance(hist.columns, pd.MultiIndex):
        for close_col in close_candidates:
            if close_col in hist.columns.get_level_values(0):
                close_frame = hist.xs(close_col, axis=1, level=0)
                if isinstance(close_frame, pd.Series):
                    return close_frame.dropna().sort_index()
                if close_frame.shape[1] == 1:
                    return close_frame.iloc[:, 0].dropna().sort_index()
        return pd.Series(dtype="float64")

    for close_col in close_candidates:
        if close_col in hist.columns:
            close_data = hist[close_col]
            if isinstance(close_data, pd.DataFrame):
                if close_data.shape[1] == 0:
                    return pd.Series(dtype="float64")
                close_data = close_data.iloc[:, 0]
            return close_data.dropna().sort_index()

    if hist.shape[1] == 1:
        return hist.iloc[:, 0].dropna().sort_index()

    return pd.Series(dtype="float64")


def _clip_values(values: np.ndarray, clip_abs: float) -> np.ndarray:
    if clip_abs <= 0:
        return np.array(values, copy=True)
    return np.clip(values, -float(clip_abs), float(clip_abs))


def attach_risk_premium_target(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns or TICKER_COL not in df.columns:
        raise ValueError(f"Input data must include columns '{DATE_COL}' and '{TICKER_COL}'.")

    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.normalize()
    out[TICKER_COL] = out[TICKER_COL].map(_normalize_symbol)

    symbols = sorted(set(out[TICKER_COL].dropna().unique().tolist() + [SP500_SYMBOL]))
    returns_map = _fetch_close_to_close_returns(symbols=symbols, trading_dates=out[DATE_COL].tolist())

    sp500_returns = returns_map.get(SP500_SYMBOL, {})

    ticker_returns: List[Optional[float]] = []
    bench_returns: List[Optional[float]] = []
    premiums: List[Optional[float]] = []

    for tkr, tday in zip(out[TICKER_COL].tolist(), out[DATE_COL].tolist()):
        tday = pd.Timestamp(tday).normalize()
        t_ret = returns_map.get(tkr, {}).get(tday)
        b_ret = sp500_returns.get(tday)

        ticker_returns.append(t_ret if t_ret is not None else np.nan)
        bench_returns.append(b_ret if b_ret is not None else np.nan)

        if t_ret is None or b_ret is None:
            premiums.append(np.nan)
        else:
            premiums.append(float(t_ret - b_ret))

    out[TICKER_RETURN_COL] = ticker_returns
    out[SP500_RETURN_COL] = bench_returns
    out[TARGET_COL] = premiums
    if TARGET_CLIP_ABS > 0:
        out[TARGET_COL] = out[TARGET_COL].clip(lower=-float(TARGET_CLIP_ABS), upper=float(TARGET_CLIP_ABS))
    return out


def _select_numeric_features(df: pd.DataFrame) -> List[str]:
    excluded = {TARGET_COL, TICKER_RETURN_COL, SP500_RETURN_COL}
    num_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found for model training.")
    return num_cols


def _build_training_tensors(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    train_df = df.dropna(subset=[TARGET_COL]).copy()
    if train_df.empty:
        raise ValueError("No rows with non-null risk premium target available for training.")

    x = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values
    y = train_df[TARGET_COL].astype(np.float32).values.reshape(-1, 1)
    y = _clip_values(y, TARGET_CLIP_ABS).astype(np.float32, copy=False)

    if scaler_mean is None or scaler_std is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0.0] = 1.0
    else:
        mean = np.array(scaler_mean, dtype=np.float32)
        std = np.array(scaler_std, dtype=np.float32)
        if mean.shape[0] != x.shape[1] or std.shape[0] != x.shape[1]:
            raise ValueError("Saved scaler stats are incompatible with current feature dimension.")
        std = std.copy()
        std[std == 0.0] = 1.0
    x_scaled = np.array((x - mean) / std, copy=True)
    y = np.array(y, copy=True)

    return torch.from_numpy(x_scaled), torch.from_numpy(y), mean, std


def train_model(df: pd.DataFrame, resume_artifact: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    feature_cols = _select_numeric_features(df)
    scaler_mean = None
    scaler_std = None
    if resume_artifact is not None:
        prev_features = resume_artifact.get("feature_columns", [])
        if list(prev_features) != list(feature_cols):
            raise ValueError("Cannot resume train: feature columns changed from saved model.")
        scaler_mean = np.array(resume_artifact["feature_mean"], dtype=np.float32)
        scaler_std = np.array(resume_artifact["feature_std"], dtype=np.float32)

    x_tensor, y_tensor, mean, std = _build_training_tensors(
        df,
        feature_cols,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )

    model_hidden_dim = int(HIDDEN_DIM)
    if resume_artifact is not None:
        model_hidden_dim = int(resume_artifact["hidden_dim"])
    model = MLPRegressor(input_dim=x_tensor.shape[1], hidden_dim=model_hidden_dim)
    if resume_artifact is not None:
        model.load_state_dict(resume_artifact["state_dict"])
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        final_pred = model(x_tensor)
        final_loss = float(loss_fn(final_pred, y_tensor).item())

    rows_used = int(y_tensor.shape[0])
    prior_cumulative_rows = int((resume_artifact or {}).get("cumulative_rows_trained", 0))
    cumulative_rows_trained = prior_cumulative_rows + rows_used

    artifact = {
        "state_dict": model.state_dict(),
        "feature_columns": feature_cols,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "input_dim": int(x_tensor.shape[1]),
        "hidden_dim": int(model_hidden_dim),
        "target_column": TARGET_COL,
        "sp500_symbol": SP500_SYMBOL,
        "target_clip_abs": float(TARGET_CLIP_ABS),
        "prediction_clip_abs": float(PREDICTION_CLIP_ABS),
        "min_train_rows_for_alerts": int(MIN_TRAIN_ROWS_FOR_ALERTS),
        "cumulative_rows_trained": cumulative_rows_trained,
    }
    metrics = {
        "train_mse": final_loss,
        "rows_used": rows_used,
        "cumulative_rows_trained": cumulative_rows_trained,
        "resumed_from_existing": bool(resume_artifact is not None),
    }
    return artifact, metrics


def _save_model_artifact(artifact: Dict, model_bucket: str, model_key: str) -> None:
    buff = io.BytesIO()
    torch.save(artifact, buff)
    buff.seek(0)
    s3.put_object(Bucket=model_bucket, Key=model_key, Body=buff.getvalue(), ContentType="application/octet-stream")


def _load_model_artifact(model_bucket: str, model_key: str) -> Dict:
    obj = s3.get_object(Bucket=model_bucket, Key=model_key)
    payload = io.BytesIO(obj["Body"].read())
    artifact = torch.load(payload, map_location="cpu")
    return artifact


def _load_model_artifact_local(model_path: str) -> Dict:
    return torch.load(model_path, map_location="cpu")


def _latest_model_key(bucket: str, prefix: str) -> Optional[str]:
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = [c for c in resp.get("Contents", []) if str(c.get("Key", "")).endswith(".pt")]
    if not contents:
        return None
    latest = max(contents, key=lambda x: x["LastModified"])
    return latest["Key"]


def _list_all_s3_keys(bucket: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys.extend([o["Key"] for o in resp.get("Contents", []) if o.get("Key")])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def _archive_key(src_key: str, archive_prefix: str) -> str:
    p = (archive_prefix or "").lstrip("/")
    if p and not p.endswith("/"):
        p += "/"
    return f"{p}{src_key.lstrip('/')}"


def _move_keys_to_archive(src_bucket: str, dst_bucket: str, keys: List[str], archive_prefix: str) -> List[Dict[str, str]]:
    moved: List[Dict[str, str]] = []
    for src_key in keys:
        dst_key = _archive_key(src_key, archive_prefix)
        if src_bucket == dst_bucket and src_key == dst_key:
            continue
        s3.copy_object(Bucket=dst_bucket, Key=dst_key, CopySource={"Bucket": src_bucket, "Key": src_key})
        s3.delete_object(Bucket=src_bucket, Key=src_key)
        moved.append({"from": f"s3://{src_bucket}/{src_key}", "to": f"s3://{dst_bucket}/{dst_key}"})
    return moved


def _publish_prediction_alert_if_needed(
    *,
    pred_df: pd.DataFrame,
    source_bucket: str,
    source_key: str,
    predictions_output: str,
    upside_threshold_pct: float,
    downside_threshold_pct: float,
    topic_arn: str,
    cumulative_rows_trained: int,
    min_train_rows_for_alerts: int,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "enabled": bool(topic_arn),
        "topic_arn": topic_arn or None,
        "upside_threshold_pct": float(upside_threshold_pct),
        "downside_threshold_pct": float(downside_threshold_pct),
        "published": False,
        "count_upside_hits": 0,
        "count_downside_hits": 0,
        "cumulative_rows_trained": int(cumulative_rows_trained),
        "min_train_rows_for_alerts": int(min_train_rows_for_alerts),
        "suppressed_reason": None,
    }
    if not topic_arn:
        return result
    if PRED_COL not in pred_df.columns:
        return result
    if int(cumulative_rows_trained) < int(min_train_rows_for_alerts):
        result["suppressed_reason"] = "insufficient_cumulative_training_rows"
        return result

    upside_hits = pred_df[pred_df[PRED_COL] >= float(upside_threshold_pct)].copy()
    downside_hits = pred_df[pred_df[PRED_COL] <= float(downside_threshold_pct)].copy()
    result["count_upside_hits"] = int(len(upside_hits))
    result["count_downside_hits"] = int(len(downside_hits))
    if upside_hits.empty and downside_hits.empty:
        return result

    cols = [c for c in [PRED_DATE_COL, PRED_TICKER_COL, PRED_COL, PRED_RANK_COL, PRED_POP_COL] if c in pred_df.columns]
    if not cols:
        cols = [PRED_COL]
    preview_sections = []
    if not upside_hits.empty:
        upside_preview = upside_hits.sort_values(PRED_COL, ascending=False)[cols].head(max(1, int(SNS_MAX_ROWS_IN_MESSAGE)))
        preview_sections.append(
            f"Upside hits (>= {float(upside_threshold_pct):.2f}%):\n{upside_preview.to_json(orient='records')}"
        )
    if not downside_hits.empty:
        downside_preview = downside_hits.sort_values(PRED_COL, ascending=True)[cols].head(max(1, int(SNS_MAX_ROWS_IN_MESSAGE)))
        preview_sections.append(
            f"Downside hits (<= {float(downside_threshold_pct):.2f}%):\n{downside_preview.to_json(orient='records')}"
        )

    subject = (
        f"Model Alert: {len(upside_hits)} upside / {len(downside_hits)} downside threshold hit(s)"
    )
    message = (
        f"Source: s3://{source_bucket}/{source_key}\n"
        f"Predictions file: {predictions_output}\n"
        f"Upside threshold (risk premium %): {float(upside_threshold_pct):.2f}\n"
        f"Downside threshold (risk premium %): {float(downside_threshold_pct):.2f}\n"
        f"Upside hit count: {len(upside_hits)}\n"
        f"Downside hit count: {len(downside_hits)}\n"
        f"{chr(10).join(preview_sections)}"
    )
    sns.publish(TopicArn=topic_arn, Subject=subject[:100], Message=message)
    result["published"] = True
    return result


def predict(df: pd.DataFrame, artifact: Dict, pop_threshold_pct: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    feature_cols = artifact["feature_columns"]
    mean = np.array(artifact["feature_mean"], dtype=np.float32)
    std = np.array(artifact["feature_std"], dtype=np.float32)

    x = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values
    x = (x - mean) / std

    model = MLPRegressor(input_dim=int(artifact["input_dim"]), hidden_dim=int(artifact["hidden_dim"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(x)).numpy().reshape(-1)
    preds = _clip_values(preds, float(artifact.get("prediction_clip_abs", PREDICTION_CLIP_ABS))).astype(np.float32, copy=False)
    out[PRED_COL] = preds

    # Keep explicit identifiers in prediction output for downstream trading logic.
    if TICKER_COL in out.columns:
        out[PRED_TICKER_COL] = out[TICKER_COL].astype(str).str.upper()
    if DATE_COL in out.columns:
        out[PRED_DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.strftime("%Y-%m-%d")
        out[PRED_RANK_COL] = (
            out.groupby(PRED_DATE_COL)[PRED_COL]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
    out[PRED_POP_COL] = out[PRED_COL] >= float(pop_threshold_pct)

    sort_cols = [c for c in [PRED_DATE_COL, PRED_COL] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else [False])
    return out


def _resolve_model_artifact_for_inference(event: Dict) -> Tuple[Dict, str]:
    model_bucket = event.get("model_bucket", MODEL_BUCKET)
    model_key = event.get("model_key")

    if model_bucket and model_key:
        return _load_model_artifact(model_bucket, model_key), f"s3://{model_bucket}/{model_key}"

    if not model_bucket:
        if os.path.exists(BAKED_MODEL_ARTIFACT):
            return _load_model_artifact_local(BAKED_MODEL_ARTIFACT), BAKED_MODEL_ARTIFACT
        raise ValueError(
            "Inference requires MODEL_BUCKET/event.model_bucket or a baked model at /opt/models/reddit-risk-premium/model_latest.pt."
        )

    resolved_key = _latest_model_key(model_bucket, MODEL_PREFIX)
    if not resolved_key:
        if os.path.exists(BAKED_MODEL_ARTIFACT):
            return _load_model_artifact_local(BAKED_MODEL_ARTIFACT), BAKED_MODEL_ARTIFACT
        raise ValueError(f"No model artifact found under s3://{model_bucket}/{MODEL_PREFIX}")
    return _load_model_artifact(model_bucket, resolved_key), f"s3://{model_bucket}/{resolved_key}"


def predict_handler(event, context):
    artifact, model_ref = _resolve_model_artifact_for_inference(event)
    pop_threshold_pct = float(event.get("pop_threshold_pct", 0.0))
    upside_threshold_pct = float(event.get("alert_upside_threshold_pct", event.get("alert_threshold_pct", SNS_ALERT_UPSIDE_THRESHOLD_PCT)))
    downside_threshold_pct = float(event.get("alert_downside_threshold_pct", SNS_ALERT_DOWNSIDE_THRESHOLD_PCT))
    sns_topic_arn = event.get("sns_topic_arn") or SNS_TOPIC_ARN
    output_bucket = event.get("output_bucket") or event.get("dest_bucket") or OUTPUT_BUCKET or MODEL_BUCKET
    if not output_bucket:
        raise ValueError("Prediction requires OUTPUT_BUCKET, MODEL_BUCKET, or event.output_bucket.")
    archive_bucket = event.get("archive_bucket") or ARCHIVE_BUCKET or "archived-data-0293483"
    archive_prefix = event.get("archive_prefix") or ARCHIVE_PREFIX
    outputs: List[Dict[str, object]] = []
    archived: List[Dict[str, str]] = []
    skipped_sources: List[Dict[str, str]] = []
    cumulative_rows_trained = int(artifact.get("cumulative_rows_trained", 0))
    min_train_rows_for_alerts = int(artifact.get("min_train_rows_for_alerts", MIN_TRAIN_ROWS_FOR_ALERTS))

    for src_bucket, src_key in _parse_s3_locations(event):
        if archive_bucket == src_bucket:
            raise ValueError("archive bucket must be different from src_bucket.")

        try:
            df = _read_df_from_s3(src_bucket, src_key)
            pred_df = predict(df, artifact, pop_threshold_pct=pop_threshold_pct)
        except Exception as exc:
            skipped_sources.append({
                "source": f"s3://{src_bucket}/{src_key}",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        base = os.path.basename(src_key).rsplit(".", 1)[0]
        pred_key = f"{OUTPUT_PREFIX}/predictions/{base}_predictions.parquet"
        _write_df_to_s3_parquet(pred_df, output_bucket, pred_key)
        predictions_uri = f"s3://{output_bucket}/{pred_key}"

        signal_cols = [c for c in [PRED_DATE_COL, PRED_TICKER_COL, PRED_COL, PRED_RANK_COL, PRED_POP_COL] if c in pred_df.columns]
        signals_key = None
        if signal_cols:
            signals_df = pred_df[signal_cols].copy()
            signals_key = f"{OUTPUT_PREFIX}/signals/{base}_signals.parquet"
            _write_df_to_s3_parquet(signals_df, output_bucket, signals_key)

        alert_result = _publish_prediction_alert_if_needed(
            pred_df=pred_df,
            source_bucket=src_bucket,
            source_key=src_key,
            predictions_output=predictions_uri,
            upside_threshold_pct=upside_threshold_pct,
            downside_threshold_pct=downside_threshold_pct,
            topic_arn=sns_topic_arn,
            cumulative_rows_trained=cumulative_rows_trained,
            min_train_rows_for_alerts=min_train_rows_for_alerts,
        )
        archived.extend(_move_keys_to_archive(src_bucket, archive_bucket, [src_key], archive_prefix))
        outputs.append({
            "source": f"s3://{src_bucket}/{src_key}",
            "predictions_output": predictions_uri,
            "signals_output": f"s3://{output_bucket}/{signals_key}" if signals_key else None,
            "alert": alert_result,
        })

    if not outputs:
        raise ValueError("No prediction files could be processed successfully.")

    result = {
        "status": "ok",
        "action": "predict",
        "model_artifact": model_ref,
        "pop_threshold_pct": pop_threshold_pct,
        "cumulative_rows_trained": cumulative_rows_trained,
        "min_train_rows_for_alerts": min_train_rows_for_alerts,
        "sources": [item["source"] for item in outputs],
        "outputs": outputs,
        "skipped_sources": skipped_sources,
        "skipped_source_count": len(skipped_sources),
        "archived_bucket": archive_bucket,
        "archived_prefix": archive_prefix,
        "n_archived_objects": len(archived),
        "archived": archived,
    }
    if len(outputs) == 1:
        result["source"] = outputs[0]["source"]
        result["predictions_output"] = outputs[0]["predictions_output"]
        result["signals_output"] = outputs[0]["signals_output"]
        result["alert"] = outputs[0]["alert"]
    return result


def train_handler(event, context):
    locations = _parse_s3_locations(event)
    if not locations:
        raise ValueError("Training requires at least one source S3 object.")

    training_frames: List[pd.DataFrame] = []
    source_uris: List[str] = []
    skipped_sources: List[Dict[str, str]] = []
    for src_bucket, src_key in locations:
        source_uri = f"s3://{src_bucket}/{src_key}"
        try:
            df = _read_df_from_s3(src_bucket, src_key)
            df_with_target = attach_risk_premium_target(df)
            usable_rows = int(df_with_target[TARGET_COL].notna().sum()) if TARGET_COL in df_with_target.columns else 0
            if usable_rows <= 0:
                skipped_sources.append({
                    "source": source_uri,
                    "reason": "no_non_null_targets",
                })
                continue
            training_frames.append(df_with_target)
            source_uris.append(source_uri)
        except Exception as exc:
            skipped_sources.append({
                "source": source_uri,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    if not training_frames:
        raise ValueError("No usable training files remained after filtering invalid or targetless sources.")

    if len(training_frames) == 1:
        df_with_target = training_frames[0]
    else:
        df_with_target = pd.concat(training_frames, ignore_index=True, copy=False)

    write_targets = bool(event.get("write_targets", True))
    target_key = None
    if write_targets:
        if len(locations) == 1:
            target_stem = os.path.basename(locations[0][1]).rsplit(".", 1)[0]
        else:
            target_stem = f"combined_{context.aws_request_id}"
        target_key = f"{OUTPUT_PREFIX}/targets/{target_stem}_with_targets.parquet"
        _write_df_to_s3_parquet(df_with_target, OUTPUT_BUCKET, target_key)

    model_bucket = event.get("model_bucket", MODEL_BUCKET)
    if not model_bucket:
        raise ValueError("MODEL_BUCKET (or event.model_bucket) is required for train.")

    should_resume = bool(event.get("resume_train", TRAIN_RESUME_DEFAULT))
    resume_artifact = None
    resumed_from = None
    if should_resume:
        resume_key = event.get("resume_model_key") or _latest_model_key(model_bucket, MODEL_PREFIX)
        if resume_key:
            resume_artifact = _load_model_artifact(model_bucket, resume_key)
            resumed_from = f"s3://{model_bucket}/{resume_key}"
        elif os.path.exists(BAKED_MODEL_ARTIFACT):
            resume_artifact = _load_model_artifact_local(BAKED_MODEL_ARTIFACT)
            resumed_from = BAKED_MODEL_ARTIFACT

    artifact, metrics = train_model(df_with_target, resume_artifact=resume_artifact)
    model_key = event.get("model_key") or LATEST_MODEL_KEY
    _save_model_artifact(artifact, model_bucket, model_key)

    metrics_key = f"{MODEL_PREFIX}/metrics_{context.aws_request_id}.json"
    s3.put_object(
        Bucket=model_bucket,
        Key=metrics_key,
        Body=json.dumps(metrics).encode("utf-8"),
        ContentType="application/json",
    )

    result = {
        "status": "ok",
        "action": "train",
        "sources": source_uris,
        "source_count": len(source_uris),
        "skipped_sources": skipped_sources,
        "skipped_source_count": len(skipped_sources),
        "targets_output": f"s3://{OUTPUT_BUCKET}/{target_key}" if target_key else None,
        "model_artifact": f"s3://{model_bucket}/{model_key}",
        "resumed_from": resumed_from,
        "metrics": metrics,
        "metrics_output": f"s3://{model_bucket}/{metrics_key}",
    }
    if len(source_uris) == 1:
        result["source"] = source_uris[0]

    archive_after_train = bool(event.get("archive_after_train", True))
    result["archive_after_train"] = archive_after_train
    if not archive_after_train:
        result["archived_bucket"] = None
        result["archived_prefix"] = None
        result["n_archived_objects"] = 0
        result["archived"] = []
        return result

    archive_bucket = event.get("archive_bucket") or ARCHIVE_BUCKET
    archive_prefix = event.get("archive_prefix") or ARCHIVE_PREFIX
    if not archive_bucket:
        raise ValueError("ARCHIVE_BUCKET (or event.archive_bucket) is required.")

    keys_by_bucket: Dict[str, List[str]] = {}
    for src_bucket, src_key in locations:
        if archive_bucket == src_bucket:
            raise ValueError("archive bucket must be different from src_bucket.")
        keys_by_bucket.setdefault(src_bucket, []).append(src_key)

    moved: List[Dict[str, str]] = []
    for src_bucket, src_keys in keys_by_bucket.items():
        moved.extend(_move_keys_to_archive(src_bucket, archive_bucket, src_keys, archive_prefix))

    result["archived_bucket"] = archive_bucket
    result["archived_prefix"] = archive_prefix
    result["n_archived_objects"] = len(moved)
    result["archived"] = moved
    return result


def lambda_handler(event, context):
    action = str((event or {}).get("action", "")).strip().lower()
    if action == "train":
        return train_handler(event, context)
    if action == "predict":
        return predict_handler(event, context)
    # Default keeps prior behavior (train) when action is omitted.
    return train_handler(event, context)
