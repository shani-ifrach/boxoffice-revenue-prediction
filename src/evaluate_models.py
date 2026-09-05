"""Create actual-vs-predicted and segmented error tables for saved models.

This module is intentionally separate from training. It makes model behavior
visible to a reviewer by showing where errors are concentrated rather than
reporting only one aggregate score.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train_model import PRE_RELEASE_FEATURES


def evaluate(model_name="gradient_boosting", input_path=Path("data/processed/movies_features.csv"), features=PRE_RELEASE_FEATURES):
    """Score the chronological Test period and export diagnostic tables.

    The input model must already be fitted. Revenue metrics are calculated in
    dollar space after reversing the log-target transformation used in training.
    """
    movies = pd.read_csv(input_path).sort_values("release_date")
    movies = movies[movies["budget_usd"].notna() & movies["profitable"].notna()].copy()
    # Keep evaluation aligned with train_model.py: the final test period is
    # made of future release years, not a random or arbitrary row slice.
    test_data = movies[movies["release_year"] >= 2022].copy()
    model = joblib.load(f"models/regression_{model_name}.joblib")
    test_data["predicted_revenue_usd"] = np.maximum(0, np.expm1(model.predict(test_data[features])))
    test_data["absolute_error_usd"] = (test_data["worldwide_revenue_usd"] - test_data["predicted_revenue_usd"]).abs()
    test_data["revenue_band"] = pd.cut(test_data["worldwide_revenue_usd"], bins=[-1, 250_000_000, 500_000_000, float("inf")], labels=["Regular (<$250M)", "Successful ($250M-$500M)", "Blockbuster (>$500M)"])
    Path("reports").mkdir(exist_ok=True)
    test_data[["tmdb_id", "title", "genres", "worldwide_revenue_usd", "predicted_revenue_usd", "absolute_error_usd"]].to_csv("reports/actual_vs_predicted.csv", index=False)
    by_genre = test_data.assign(genre=test_data["genres"].fillna("Unknown").str.split("; ")).explode("genre").groupby("genre").agg(movie_count=("tmdb_id", "count"), mae_usd=("absolute_error_usd", "mean"))
    by_genre.to_csv("reports/error_by_genre.csv")
    by_band = test_data.groupby("revenue_band", observed=True).agg(movie_count=("tmdb_id", "count"), mae_usd=("absolute_error_usd", "mean"), median_actual_revenue_usd=("worldwide_revenue_usd", "median"), median_predicted_revenue_usd=("predicted_revenue_usd", "median")).reset_index()
    by_band.to_csv("reports/error_by_revenue_band.csv", index=False)
    print(json.dumps(json.loads(Path("models/metrics.json").read_text(encoding="utf-8")), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="random_forest", choices=["ridge", "random_forest", "gradient_boosting"])
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    args = parser.parse_args()
    evaluate(args.model_name, args.input_path)


if __name__ == "__main__":
    main()
