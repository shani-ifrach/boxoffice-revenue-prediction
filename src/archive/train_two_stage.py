"""Train a two-stage blockbuster-aware revenue model.

Stage A estimates the probability that a movie will exceed $300M worldwide.
Stage B is one regression model: it receives the regular pre-release features
plus Stage A's predicted probability as an additional signal.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.train_model import (
    PRE_RELEASE_FEATURES,
    add_history_from_prior_period,
    classification_metrics,
    make_pipeline,
    make_preprocessor,
    regression_metrics,
)


BLOCKBUSTER_THRESHOLD = 300_000_000
STAGE_B_FEATURES = PRE_RELEASE_FEATURES + ["blockbuster_probability"]


def make_stage_b_pipeline():
    return Pipeline([
        ("preprocess", make_preprocessor(STAGE_B_FEATURES)),
        ("model", RandomForestClassifier() if False else GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=2,
            loss="huber", random_state=42,
        )),
    ])


def make_blockbuster_classifier():
    return make_pipeline(RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ))


def out_of_fold_probability(X, y):
    """Create leakage-safe Stage A probabilities for Stage B training rows."""
    probabilities = np.zeros(len(X), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_indices, holdout_indices in splitter.split(X, y):
        model = make_blockbuster_classifier()
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        probabilities[holdout_indices] = model.predict_proba(X.iloc[holdout_indices])[:, 1]
    return probabilities


def add_probability(features, probability):
    result = features.copy()
    result["blockbuster_probability"] = probability
    return result


def train():
    movies = pd.read_csv("data/processed/movies_features.csv", parse_dates=["release_date"])
    movies = movies[movies["budget_usd"].notna() & movies["worldwide_revenue_usd"].notna()].copy()
    movies["blockbuster"] = (movies["worldwide_revenue_usd"] > BLOCKBUSTER_THRESHOLD).astype(int)

    train_data = movies[movies["release_year"] <= 2019].copy()
    validation_data = movies[movies["release_year"].between(2020, 2021)].copy()
    test_data = movies[movies["release_year"] >= 2022].copy()
    if min(len(train_data), len(validation_data), len(test_data)) == 0:
        raise ValueError("Each time split must contain at least one row.")

    # Historical features for future periods are rebuilt from information that
    # would have existed at that point in time.
    validation_data = add_history_from_prior_period(validation_data, train_data)
    test_data = add_history_from_prior_period(test_data, pd.concat([train_data, validation_data]))

    X_train = train_data[PRE_RELEASE_FEATURES]
    X_validation = validation_data[PRE_RELEASE_FEATURES]
    X_test = test_data[PRE_RELEASE_FEATURES]
    y_train_class = train_data["blockbuster"]
    y_validation_class = validation_data["blockbuster"]
    y_test_class = test_data["blockbuster"]

    stage_a_validation = make_blockbuster_classifier()
    stage_a_validation.fit(X_train, y_train_class)
    validation_probability = stage_a_validation.predict_proba(X_validation)[:, 1]

    train_probability = out_of_fold_probability(X_train, y_train_class)
    stage_b_validation = make_stage_b_pipeline()
    stage_b_validation.fit(
        add_probability(X_train, train_probability),
        np.log1p(train_data["worldwide_revenue_usd"]),
    )
    validation_revenue_prediction = np.maximum(
        0, np.expm1(stage_b_validation.predict(add_probability(X_validation, validation_probability)))
    )

    combined = pd.concat([train_data, validation_data], ignore_index=True)
    X_combined = combined[PRE_RELEASE_FEATURES]
    y_combined_class = combined["blockbuster"]
    stage_a_final = make_blockbuster_classifier()
    stage_a_final.fit(X_combined, y_combined_class)
    test_probability = stage_a_final.predict_proba(X_test)[:, 1]
    combined_probability = out_of_fold_probability(X_combined, y_combined_class)

    stage_b_final = make_stage_b_pipeline()
    stage_b_final.fit(
        add_probability(X_combined, combined_probability),
        np.log1p(combined["worldwide_revenue_usd"]),
    )
    test_revenue_prediction = np.maximum(
        0, np.expm1(stage_b_final.predict(add_probability(X_test, test_probability)))
    )

    metrics = {
        "threshold_usd": BLOCKBUSTER_THRESHOLD,
        "split": {
            "train_rows": len(train_data), "validation_rows": len(validation_data), "test_rows": len(test_data),
            "train_years": "2010-2019", "validation_years": "2020-2021", "test_years": "2022-2024",
        },
        "stage_a_blockbuster_classifier": {
            "validation": classification_metrics(
                y_validation_class,
                validation_probability >= 0.5,
                validation_probability,
            ),
            "test": classification_metrics(
                y_test_class,
                test_probability >= 0.5,
                test_probability,
            ),
        },
        "stage_b_single_revenue_regressor": {
            "validation": regression_metrics(
                validation_data["worldwide_revenue_usd"], validation_revenue_prediction,
            ),
            "test": regression_metrics(
                test_data["worldwide_revenue_usd"], test_revenue_prediction,
            ),
        },
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(stage_a_final, "models/two_stage_blockbuster_classifier.joblib")
    joblib.dump(stage_b_final, "models/two_stage_revenue_regressor.joblib")
    Path("models/two_stage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
