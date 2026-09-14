"""Check whether the reduced feature contract remains better across time splits."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_ablation import make_regression_pipeline
from src.train_model import FULL_PRE_RELEASE_FEATURES, REDUCED_PRE_RELEASE_FEATURES


TIME_SPLITS = [
    (2015, 2016, 2017, "early"),
    (2017, 2018, 2019, "middle"),
    (2019, 2020, 2021, "recent"),
]


def score(actual, prediction):
    return {
        "MAE_usd": mean_absolute_error(actual, prediction),
        "RMSE_usd": mean_squared_error(actual, prediction) ** 0.5,
        "R2": r2_score(actual, prediction),
    }


def make_model(features):
    return make_regression_pipeline(features)


def run_stability(input_path, output_path):
    movies = pd.read_csv(input_path, parse_dates=["release_date"])
    movies = movies[movies["worldwide_revenue_usd"].notna()].sort_values("release_date")
    rows = []

    for train_end, validation_start, validation_end, split_name in TIME_SPLITS:
        train_data = movies[movies["release_year"] <= train_end].copy()
        validation_data = movies[movies["release_year"].between(validation_start, validation_end)].copy()
        test_data = movies[movies["release_year"] > validation_end].copy()
        if min(len(train_data), len(validation_data), len(test_data)) == 0:
            continue

        for model_name, features in (("full_model", FULL_PRE_RELEASE_FEATURES), ("reduced_model", REDUCED_PRE_RELEASE_FEATURES)):
            model = make_model(features)
            model.fit(train_data[features], np.log1p(train_data["worldwide_revenue_usd"]))
            validation_prediction = np.maximum(0, np.expm1(model.predict(validation_data[features])))
            test_prediction = np.maximum(0, np.expm1(model.predict(test_data[features])))
            for period, actual, prediction in (("validation", validation_data["worldwide_revenue_usd"], validation_prediction), ("test", test_data["worldwide_revenue_usd"], test_prediction)):
                row = {"split": split_name, "train_end_year": train_end, "validation_years": f"{validation_start}-{validation_end}", "test_start_year": validation_end + 1, "period": period, "model": model_name, "train_rows": len(train_data), "validation_rows": len(validation_data), "test_rows": len(test_data)}
                row.update(score(actual, prediction))
                rows.append(row)

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format="%.4f")
    print(results.to_string(index=False))
    print(f"\nSaved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("reports/feature_stability_results.csv"))
    args = parser.parse_args()
    run_stability(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
