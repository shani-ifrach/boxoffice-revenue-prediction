"""Build one production feature row with the training builder and score artifacts."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.collect_tmdb import tmdb_get
from src.config import BLOCKBUSTER_THRESHOLD_USD
from src.data_cleaning import flatten_movie
from src.feature_engineering import add_historical_features

DEFAULT_HISTORY_PATH = Path("data/processed/movies_clean.csv")
DEFAULT_MODEL_DIR = Path("models/reduced")


def build_prediction_row(movie_record, history_path=DEFAULT_HISTORY_PATH, features=None):
    """Create one pre-release feature row using the training-time builder."""
    movie = flatten_movie(movie_record)
    release_date = pd.to_datetime(movie.get("release_date"), errors="coerce")
    if pd.isna(release_date): raise ValueError("Movie requires a valid release_date for point-in-time features.")
    for column in ("budget_usd", "runtime_minutes"):
        value = pd.to_numeric(movie.get(column), errors="coerce")
        movie[column] = np.nan if pd.isna(value) or value <= 0 else float(value)
    movie["worldwide_revenue_usd"] = np.nan
    movie["profitable"] = pd.NA
    target = pd.DataFrame([movie])
    history = pd.read_csv(history_path, parse_dates=["release_date"])
    row = add_historical_features(target, history)
    return row[features] if features is not None else row


def validate_artifact(artifact, expected_kind):
    """Validate the minimum schema required before loading a model artifact."""
    required = {"artifact_version", "kind", "features", "feature_dtypes", "model"}
    missing = required - set(artifact)
    if missing: raise ValueError(f"Incompatible artifact: missing {sorted(missing)}")
    if artifact["kind"] != expected_kind: raise ValueError(f"Expected {expected_kind}, got {artifact['kind']}")
    if len(artifact["features"]) not in (35, 38): raise ValueError("Unsupported feature contract size.")


def predict_record(movie_record, history_path=DEFAULT_HISTORY_PATH, model_dir=DEFAULT_MODEL_DIR):
    """Score a movie record with the saved revenue and blockbuster artifacts."""
    revenue = joblib.load(model_dir / "revenue_regression.joblib")
    blockbuster = joblib.load(model_dir / "blockbuster_classifier.joblib")
    validate_artifact(revenue, "revenue_regression")
    validate_artifact(blockbuster, "blockbuster_classification")
    if revenue["features"] != blockbuster["features"]: raise ValueError("Revenue and classifier feature contracts differ.")
    row = build_prediction_row(movie_record, history_path, revenue["features"])
    log_prediction = float(revenue["model"].predict(row)[0])
    point = float(np.expm1(np.clip(log_prediction, 0, np.log1p(revenue["prediction_cap_usd"]))))
    width = float(revenue["conformal_absolute_error_usd"])
    raw = blockbuster["model"].predict_proba(row)[:, 1]
    probability = float(blockbuster["calibrator"].predict_proba(raw.reshape(-1, 1))[0, 1])
    return {"tmdb_id": int(movie_record["id"]), "title": movie_record.get("title"), "predicted_revenue_usd": point, "prediction_lower_usd": max(0.0, point-width), "prediction_upper_usd": point+width, "interval_nominal_coverage": revenue["conformal_coverage"], "blockbuster_threshold_usd": BLOCKBUSTER_THRESHOLD_USD, "blockbuster_probability": probability, "blockbuster_decision_threshold": blockbuster["decision_threshold"], "blockbuster_prediction": int(probability >= blockbuster["decision_threshold"])}


def predict_tmdb_movie(tmdb_id, history_path=DEFAULT_HISTORY_PATH, model_dir=DEFAULT_MODEL_DIR):
    """Fetch one movie from TMDB and return its local model predictions."""
    token = os.getenv("TMDB_API_KEY")
    if not token: raise SystemExit("Set TMDB_API_KEY before requesting a TMDB movie.")
    record = tmdb_get(f"movie/{tmdb_id}", token, {"append_to_response": "credits", "language": "en-US"})
    return predict_record(record, history_path, model_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmdb-id", type=int, required=True)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(predict_tmdb_movie(args.tmdb_id, args.history_path, args.model_dir), indent=2))


if __name__ == "__main__": main()
