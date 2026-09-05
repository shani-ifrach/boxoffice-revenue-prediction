"""Train and save a binary $400M two-stage revenue model.

Stage A predicts whether a movie will exceed a configurable revenue threshold.
Stage B is one Random Forest regressor that receives Stage A's probability as an
additional feature and predicts log worldwide revenue.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from src.train_model import (
    PRE_RELEASE_FEATURES,
    add_history_from_prior_period,
    classification_metrics,
    make_preprocessor,
    regression_metrics,
)


def make_classifier(n_estimators):
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


def make_regressor(n_estimators):
    stage_b_features = PRE_RELEASE_FEATURES + ["blockbuster_probability"]
    return Pipeline([
        ("preprocess", make_preprocessor(stage_b_features)),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def add_probability(features, probability):
    result = features.copy()
    result["blockbuster_probability"] = probability
    return result


def out_of_fold_probability(features, target, n_estimators):
    """Create leakage-safe Stage A probabilities for Stage B training rows."""
    probabilities = np.zeros(len(features), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_indices, holdout_indices in splitter.split(features, target):
        classifier = make_classifier(n_estimators)
        classifier.fit(features.iloc[train_indices], target.iloc[train_indices])
        probabilities[holdout_indices] = classifier.predict_proba(
            features.iloc[holdout_indices]
        )[:, 1]
    return probabilities


def train(input_path, output_dir, threshold_usd, classifier_trees, regressor_trees):
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
    if min(len(train_data), len(validation_data), len(test_data)) == 0:
        raise ValueError("Each time split must contain at least one row.")

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

    stage_a_validation = make_classifier(classifier_trees)
    stage_a_validation.fit(X_train, y_train)
    validation_probability = stage_a_validation.predict_proba(X_validation)[:, 1]

    train_probability = out_of_fold_probability(
        X_train, y_train, classifier_trees
    )
    stage_b_validation = make_regressor(regressor_trees)
    stage_b_validation.fit(
        add_probability(X_train, train_probability),
        np.log1p(train_data["worldwide_revenue_usd"]),
    )
    validation_revenue = np.maximum(
        0,
        np.expm1(
            stage_b_validation.predict(
                add_probability(X_validation, validation_probability)
            )
        ),
    )

    combined = pd.concat([train_data, validation_data], ignore_index=True)
    X_combined = combined[PRE_RELEASE_FEATURES]
    y_combined = combined["blockbuster"]

    stage_a_final = make_classifier(classifier_trees)
    stage_a_final.fit(X_combined, y_combined)
    test_probability = stage_a_final.predict_proba(X_test)[:, 1]

    combined_probability = out_of_fold_probability(
        X_combined, y_combined, classifier_trees
    )
    stage_b_final = make_regressor(regressor_trees)
    stage_b_final.fit(
        add_probability(X_combined, combined_probability),
        np.log1p(combined["worldwide_revenue_usd"]),
    )
    test_revenue = np.maximum(
        0,
        np.expm1(
            stage_b_final.predict(add_probability(X_test, test_probability))
        ),
    )

    metrics = {
        "threshold_usd": threshold_usd,
        "feature_count": len(PRE_RELEASE_FEATURES),
        "trees": {
            "stage_a_classifier": classifier_trees,
            "stage_b_regressor": regressor_trees,
        },
        "split": {
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "test_rows": len(test_data),
            "train_years": "2010-2019",
            "validation_years": "2020-2021",
            "test_years": "2022-2024",
        },
        "stage_a_blockbuster_classifier": {
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
        },
        "stage_b_single_revenue_regressor": {
            "validation": regression_metrics(
                validation_data["worldwide_revenue_usd"], validation_revenue
            ),
            "test": regression_metrics(
                test_data["worldwide_revenue_usd"], test_revenue
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stage_a_final, output_dir / "stage_a_400m_classifier.joblib")
    joblib.dump(stage_b_final, output_dir / "stage_b_400m_revenue_regressor.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions = test_data[["tmdb_id", "title", "release_year", "worldwide_revenue_usd"]].copy()
    predictions["blockbuster_actual"] = y_test.to_numpy()
    predictions["blockbuster_probability"] = test_probability
    predictions["blockbuster_predicted"] = (test_probability >= 0.5).astype(int)
    predictions["predicted_revenue_usd"] = test_revenue
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved 400M two-stage artifacts to {output_dir}")


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
        default=Path("models/two_stage_400m"),
    )
    parser.add_argument("--threshold-usd", type=int, default=400_000_000)
    parser.add_argument("--classifier-trees", type=int, default=500)
    parser.add_argument("--regressor-trees", type=int, default=500)
    args = parser.parse_args()
    train(
        args.input_path,
        args.output_dir,
        args.threshold_usd,
        args.classifier_trees,
        args.regressor_trees,
    )


if __name__ == "__main__":
    main()

