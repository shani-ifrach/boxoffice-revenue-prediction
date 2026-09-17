"""Build one production feature row with the training builder and score artifacts."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.collect_tmdb import tmdb_get
from src.config import BLOCKBUSTER_THRESHOLD_USD, SCHEMA_VERSION
from src.data_cleaning import flatten_movie
from src.feature_engineering import add_historical_features
from src.train_model import CATEGORICAL_FEATURES, PRE_RELEASE_FEATURES

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
    """Reject unsupported versions and incomplete task-specific contracts."""
    if not isinstance(artifact, dict):
        raise ValueError("Incompatible artifact: expected a dictionary.")
    required = {"artifact_version", "schema_version", "kind", "features", "feature_dtypes", "model"}
    task_fields = {
        "revenue_regression": {"prediction_cap_usd", "conformal_absolute_error_usd", "conformal_coverage", "target_transform"},
        "blockbuster_classification": {"calibrator", "decision_threshold", "blockbuster_threshold_usd"},
        "profitability_classification": {"calibrator", "decision_threshold"},
    }
    if expected_kind not in task_fields:
        raise ValueError(f"Unsupported artifact kind: {expected_kind}")
    required |= task_fields[expected_kind]
    missing = required - set(artifact)
    if missing: raise ValueError(f"Incompatible artifact: missing {sorted(missing)}")
    if artifact["kind"] != expected_kind: raise ValueError(f"Expected {expected_kind}, got {artifact['kind']}")
    if artifact["artifact_version"] != 2:
        raise ValueError("Unsupported artifact_version; expected 2.")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version; expected {SCHEMA_VERSION}.")
    if artifact["features"] != PRE_RELEASE_FEATURES:
        raise ValueError("Unsupported ordered feature contract; expected the final 35 features.")
    expected_dtypes = {f: "category" if f in CATEGORICAL_FEATURES else "number" for f in artifact["features"]}
    if artifact["feature_dtypes"] != expected_dtypes:
        raise ValueError("Feature dtypes do not match the declared contract.")
    method = "predict" if expected_kind == "revenue_regression" else "predict_proba"
    if not callable(getattr(artifact["model"], method, None)):
        raise ValueError(f"Artifact model must implement {method}.")
    fitted_features = getattr(artifact["model"], "feature_names_in_", None)
    if fitted_features is None or list(fitted_features) != artifact["features"]:
        raise ValueError("Fitted model columns do not match the ordered feature contract.")

    def finite_number(name, minimum, maximum=None):
        value = artifact[name]
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{name} is outside its supported range.")

    if expected_kind == "revenue_regression":
        if artifact["target_transform"] != "log1p":
            raise ValueError("Unsupported target transform.")
        finite_number("prediction_cap_usd", 0)
        finite_number("conformal_absolute_error_usd", 0)
        finite_number("conformal_coverage", 0, 1)
        if artifact["prediction_cap_usd"] == 0 or artifact["conformal_coverage"] in (0, 1):
            raise ValueError("Prediction cap and interval coverage must be positive; coverage must be below 1.")
    else:
        finite_number("decision_threshold", 0, 1)
        if not callable(getattr(artifact["calibrator"], "predict_proba", None)):
            raise ValueError("Artifact calibrator must implement predict_proba.")
        if expected_kind == "blockbuster_classification" and artifact["blockbuster_threshold_usd"] != BLOCKBUSTER_THRESHOLD_USD:
            raise ValueError("Blockbuster target definition does not match $400M.")


def predict_record(movie_record, history_path=DEFAULT_HISTORY_PATH, model_dir=DEFAULT_MODEL_DIR):
    """Score a movie record with the saved revenue and blockbuster artifacts."""
    model_dir = Path(model_dir)
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
    return {"tmdb_id": int(movie_record["id"]), "title": movie_record.get("title"), "predicted_revenue_usd": point, "prediction_lower_usd": max(0.0, point-width), "prediction_upper_usd": point+width, "interval_nominal_coverage": revenue["conformal_coverage"], "interval_method": "rolling_validation_residual_quantile", "blockbuster_threshold_usd": BLOCKBUSTER_THRESHOLD_USD, "blockbuster_probability": probability, "blockbuster_decision_threshold": blockbuster["decision_threshold"], "blockbuster_prediction": int(probability >= blockbuster["decision_threshold"])}


def predict_tmdb_movie(tmdb_id, history_path=DEFAULT_HISTORY_PATH, model_dir=DEFAULT_MODEL_DIR):
    """Fetch one movie from TMDB and return its local model predictions."""
    token = os.getenv("TMDB_API_KEY")
    if not token: raise SystemExit("Set TMDB_API_KEY before requesting a TMDB movie.")
    record = tmdb_get(f"movie/{tmdb_id}", token, {"append_to_response": "credits", "language": "en-US"})
    return predict_record(record, history_path, model_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tmdb-id", type=int, help="Fetch a record from TMDB (requires API key).")
    source.add_argument("--record-json", type=Path, help="Score a local TMDB record without network access.")
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    if args.record_json:
        record = json.loads(args.record_json.read_text(encoding="utf-8"))
        result = predict_record(record, args.history_path, args.model_dir)
    else:
        result = predict_tmdb_movie(args.tmdb_id, args.history_path, args.model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
