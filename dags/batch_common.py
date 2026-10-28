import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    customers_set = set()
    if not rows:
        raise ValueError
    for row in rows:
        cus_id = row["customer_id"]
        cur_day = row["business_date"]
        customer_cond = not isinstance(cus_id, str) or not cus_id or cus_id in customers_set
        if customer_cond or cur_day != day:
            raise ValueError

        for feature, (minimum, maximum) in zip(FEATURES, LIMITS):
            value = row[feature]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError
            if not minimum <= value <= maximum:
                raise ValueError
        
        customers_set.add(cus_id)

    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if type(batch_size) is not int or batch_size < 1 or batch_size > 1000:
        raise ValueError
    
    predictions = []
    
    for i, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start: start + batch_size]
        predictions.extend(predict(batch, model))
        print(f"number: {i}, size: {len(batch)}")

    return predictions

def validate_predictions(predictions, expected_ids):
    if len(predictions) != len(expected_ids):
        raise ValueError
    
    actual_ids = [pred["customer_id"] for pred in predictions]

    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError

    if set(actual_ids) != set(expected_ids):
        raise ValueError
    
    for pred in predictions:
        score = pred["score"]
        label = pred["label"]

        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError
        if score < 0 or score > 1:
            raise ValueError
        if type(label) is not int or label != int(score >= 0.5):
            raise ValueError


def replace_day(conn, day, model_version, predictions):
    if not predictions:
        raise ValueError("Empty predictions")
    
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (day + ':' + model_version,)
            )

            cur.execute(
                """
                DELETE FROM predictions 
                WHERE business_date = %s AND model_version = %s
                """,
                (day, model_version)
            )

            cur.executemany(
                """
                INSERT INTO predictions
                    (business_date, model_version, customer_id, score, label)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (day, model_version, pred["customer_id"], pred["score"], pred["label"])
                    for pred in predictions
                ],
            )
