"""Train and save the standalone $400M blockbuster classifier.

The classifier is intentionally separate from revenue regression. It estimates
the probability that a movie will exceed the $400M worldwide-revenue threshold.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.train_model import (
    PRE_RELEASE_FEATURES,
    add_history_from_prior_period,
    classification_metrics,
    make_preprocessor,
)


BLOCKBUSTER_THRESHOLD_USD = 400_000_000


def make_classifier(n_estimators=500):
    return Pipeline([
        ("preprocess", make_preprocessor(PRE_RELEASE_FEATURES)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(input_path, output_dir, threshold_usd=BLOCKBUSTER_THRESHOLD_USD, n_estimators=500):
    movies = pd.read_csv(input_path, parse_dates=["release_date"])
    movies = movies[
        movies["budget_usd"].notna() &
        movies["worldwide_revenue_usd"].notna()
    ].sort_values("release_date").copy()
    movies["blockbuster"] = (
        movies["worldwide_revenue_usd"] > threshold_usd
    ).astype(int)

    train_data = movies[movies["release_year"] <= 2019].copy()
    validation_data = movies[movies["release_year"].between(2020, 2021)].copy()
    test_data = movies[movies["release_year"] >= 2022].copy()
    validation_data = add_history_from_prior_period(validation_data, train_data)
    test_data = add_history_from_prior_period(
        test_data, pd.concat([train_data, validation_data])
    )

    X_train = train_data[PRE_RELEASE_FEATURES]
    X_validation = validation_data[PRE_RELEASE_FEATURES]
    X_test = test_data[PRE_RELEASE_FEATURES]
    y_train = train_data["blockbuster"]
    y_validation = validation_data["blockbuster"]
    y_test = test_data["blockbuster"]

    validation_model = make_classifier(n_estimators)
    validation_model.fit(X_train, y_train)
    validation_probability = validation_model.predict_proba(X_validation)[:, 1]

    combined = pd.concat([train_data, validation_data], ignore_index=True)
    final_model = make_classifier(n_estimators)
    final_model.fit(combined[PRE_RELEASE_FEATURES], combined["blockbuster"])
    test_probability = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "threshold_usd": threshold_usd,
        "feature_count": len(PRE_RELEASE_FEATURES),
        "n_estimators": n_estimators,
        "split": {
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "test_rows": len(test_data),
            "train_years": "2010-2019",
            "validation_years": "2020-2021",
            "test_years": "2022-2024",
        },
        "validation": classification_metrics(
            y_validation,
            validation_probability >= 0.5,
            validation_probability,
        ),
        "test": classification_metrics(
            y_test,
            test_probability >= 0.5,
            test_probability,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_dir / "blockbuster_400m_classifier.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions = test_data[
        ["tmdb_id", "title", "release_year", "worldwide_revenue_usd"]
    ].copy()
    predictions["blockbuster_actual"] = y_test.to_numpy()
    predictions["blockbuster_probability"] = test_probability
    predictions["blockbuster_predicted"] = (test_probability >= 0.5).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved standalone Blockbuster classifier to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/movies_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/blockbuster_400m"),
    )
    parser.add_argument("--threshold-usd", type=int, default=BLOCKBUSTER_THRESHOLD_USD)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()
    train(args.input_path, args.output_dir, args.threshold_usd, args.n_estimators)


if __name__ == "__main__":
    main()

