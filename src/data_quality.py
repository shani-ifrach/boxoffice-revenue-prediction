"""Generate an auditable data-quality profile without arbitrary row deletion."""
import json
from pathlib import Path
import pandas as pd
import numpy as np


def build_quality_report(input_path=Path("data/processed/movies_clean.csv"), output_path=Path("reports/data_quality_report.json")):
    """Profile row counts, missingness, validity rules, and financial sensitivity."""
    data = pd.read_csv(input_path, parse_dates=["release_date"])
    report = {
        "rows": len(data), "unique_tmdb_ids": int(data.tmdb_id.nunique()),
        "duplicate_tmdb_ids": int(data.tmdb_id.duplicated().sum()),
        "year_min": int(data.release_year.min()), "year_max": int(data.release_year.max()),
        "missing_rates": {c: float(data[c].isna().mean()) for c in data.columns},
        "valid_profitability_rows": int(data.profitable.notna().sum()),
        "missing_profitability_rows": int(data.profitable.isna().sum()),
        "profitability_rate_valid_only": float(data.profitable.mean()),
        "runtime_missing_after_nonpositive_rule": int(data.runtime_minutes.isna().sum()),
        "sensitivity_counts_not_removed": {
            "budget_below_1k": int(data.budget_usd.between(0, 1_000, inclusive="neither").sum()),
            "budget_below_100k": int(data.budget_usd.between(0, 100_000, inclusive="neither").sum()),
            "revenue_below_1k": int(data.worldwide_revenue_usd.between(0, 1_000, inclusive="neither").sum()),
            "revenue_below_100k": int(data.worldwide_revenue_usd.between(0, 100_000, inclusive="neither").sum()),
        },
        "financial_quantiles": {c: {str(q): float(data[c].quantile(q)) for q in (0.01, 0.5, 0.99)} for c in ("budget_usd", "worldwide_revenue_usd")},
        "missing_entity_names": {c: int(data[c].isna().sum()) for c in ("director", "production_companies", "cast_top10")},
        "policy": "Only non-positive reported finance values are treated as missing. Tiny positive values are flagged, not removed, because TMDB supplies no defensible universal cutoff.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_analysis_outputs(clean_path, features_path, predictions_path, evaluation_path):
    """Validate analytical keys, evaluation population and saved numeric predictions."""
    frames = {"clean": pd.read_csv(clean_path, low_memory=False),
              "features": pd.read_csv(features_path, low_memory=False),
              "predictions": pd.read_csv(predictions_path, low_memory=False),
              "evaluation": pd.read_csv(evaluation_path, low_memory=False)}
    for name, frame in frames.items():
        if not {"tmdb_id", "title"}.issubset(frame.columns):
            raise ValueError(f"{name}: missing tmdb_id/title columns")
        if frame.empty or frame.tmdb_id.isna().any() or not frame.tmdb_id.is_unique:
            raise ValueError(f"{name}: movie keys must be non-empty, unique and non-null")
    canonical = frames["clean"].set_index("tmdb_id")
    if set(frames["features"].tmdb_id) != set(canonical.index):
        raise ValueError("Feature population must match the cleaned population")
    for name in ("features", "predictions", "evaluation"):
        frame = frames[name]
        titles = frame.tmdb_id.map(canonical.title)
        if titles.isna().any() or not frame.title.equals(titles):
            raise ValueError(f"{name}: title values must match clean data by tmdb_id")
    predictions = frames["predictions"]
    evaluation = frames["evaluation"].set_index("tmdb_id")
    if set(predictions.tmdb_id) != set(evaluation.index):
        raise ValueError("Prediction and evaluation populations must match")
    numeric = ["worldwide_revenue_usd", "prediction_lower_usd", "predicted_revenue_usd",
               "prediction_upper_usd", "blockbuster_probability"]
    if not set(numeric + ["release_year"]).issubset(predictions.columns):
        raise ValueError("Prediction table is missing required numeric/year columns")
    values = predictions[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Prediction outputs must be finite")
    if not predictions.release_year.between(2022, 2024).all():
        raise ValueError("Prediction outputs must contain final-test movies only")
    if not ((values.prediction_lower_usd >= 0)
            & (values.prediction_lower_usd <= values.predicted_revenue_usd)
            & (values.predicted_revenue_usd <= values.prediction_upper_usd)).all():
        raise ValueError("Invalid prediction interval")
    if not values.blockbuster_probability.between(0, 1).all():
        raise ValueError("Invalid blockbuster probability")
    for column in numeric:
        if column not in evaluation or not np.allclose(
                values[column], evaluation.loc[predictions.tmdb_id, column], rtol=1e-9, atol=.01):
            raise ValueError(f"Evaluation does not match predictions: {column}")
    return True


if __name__ == "__main__": build_quality_report()
