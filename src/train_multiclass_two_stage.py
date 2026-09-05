"""Train a multiclass blockbuster-aware revenue model.

Stage A predicts a revenue band. Stage B is one regression model that receives
the four Stage A probabilities as additional features and predicts log revenue.
The probabilities used to train Stage B are out-of-fold to avoid target leakage.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.train_model import PRE_RELEASE_FEATURES, add_history_from_prior_period, make_pipeline, make_preprocessor, regression_metrics


REVENUE_BINS = [-np.inf, 100_000_000, 300_000_000, 600_000_000, np.inf]
REVENUE_BAND_LABELS = ["under_100m", "100m_to_300m", "300m_to_600m", "over_600m"]
PROBABILITY_FEATURES = [f"probability_{label}" for label in REVENUE_BAND_LABELS]
STAGE_B_FEATURES = PRE_RELEASE_FEATURES + PROBABILITY_FEATURES


def make_stage_a_classifier():
    return make_pipeline(RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ))


def make_stage_b_regressor():
    return Pipeline([
        ("preprocess", make_preprocessor(STAGE_B_FEATURES)),
        ("model", RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def probability_frame(model, features):
    """Return probabilities in a fixed band order, even if class order changes."""
    probabilities = model.predict_proba(features)
    result = np.zeros((len(features), len(REVENUE_BAND_LABELS)))
    for index, class_name in enumerate(model.named_steps["model"].classes_):
        result[:, REVENUE_BAND_LABELS.index(class_name)] = probabilities[:, index]
    return pd.DataFrame(result, index=features.index, columns=PROBABILITY_FEATURES)


def out_of_fold_probabilities(features, target):
    probabilities = np.zeros((len(features), len(REVENUE_BAND_LABELS)))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_indices, holdout_indices in splitter.split(features, target):
        classifier = make_stage_a_classifier()
        classifier.fit(features.iloc[train_indices], target.iloc[train_indices])
        probabilities[holdout_indices] = probability_frame(
            classifier, features.iloc[holdout_indices]
        ).to_numpy()
    return pd.DataFrame(probabilities, index=features.index, columns=PROBABILITY_FEATURES)


def add_stage_a_probabilities(features, probabilities):
    return features.join(probabilities)


def multiclass_metrics(actual, predicted, probabilities):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision_score(actual, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(actual, predicted, average="macro", zero_division=0),
        "macro_F1": f1_score(actual, predicted, average="macro", zero_division=0),
        "ROC_AUC_ovr": roc_auc_score(
            pd.get_dummies(actual).reindex(columns=REVENUE_BAND_LABELS, fill_value=0),
            probabilities[REVENUE_BAND_LABELS],
            multi_class="ovr",
        ),
    }


def train():
    movies = pd.read_csv("data/processed/movies_features.csv", parse_dates=["release_date"])
    movies = movies[movies["budget_usd"].notna() & movies["worldwide_revenue_usd"].notna()].copy()
    movies["revenue_band"] = pd.cut(
        movies["worldwide_revenue_usd"], bins=REVENUE_BINS, labels=REVENUE_BAND_LABELS,
    ).astype(str)

    train_data = movies[movies["release_year"] <= 2019].copy()
    validation_data = movies[movies["release_year"].between(2020, 2021)].copy()
    test_data = movies[movies["release_year"] >= 2022].copy()
    if min(len(train_data), len(validation_data), len(test_data)) == 0:
        raise ValueError("Each time split must contain at least one row.")

    validation_data = add_history_from_prior_period(validation_data, train_data)
    test_data = add_history_from_prior_period(test_data, pd.concat([train_data, validation_data]))

    X_train = train_data[PRE_RELEASE_FEATURES]
    X_validation = validation_data[PRE_RELEASE_FEATURES]
    X_test = test_data[PRE_RELEASE_FEATURES]
    y_train_band = train_data["revenue_band"]
    y_validation_band = validation_data["revenue_band"]
    y_test_band = test_data["revenue_band"]

    stage_a_validation = make_stage_a_classifier()
    stage_a_validation.fit(X_train, y_train_band)
    validation_probabilities = probability_frame(stage_a_validation, X_validation)
    validation_band_prediction = validation_probabilities.idxmax(axis=1).str.replace("probability_", "", regex=False)

    train_probabilities = out_of_fold_probabilities(X_train, y_train_band)
    stage_b_validation = make_stage_b_regressor()
    stage_b_validation.fit(
        add_stage_a_probabilities(X_train, train_probabilities),
        np.log1p(train_data["worldwide_revenue_usd"]),
    )
    validation_revenue_prediction = np.maximum(
        0,
        np.expm1(stage_b_validation.predict(add_stage_a_probabilities(X_validation, validation_probabilities))),
    )

    combined = pd.concat([train_data, validation_data], ignore_index=True)
    X_combined = combined[PRE_RELEASE_FEATURES]
    y_combined_band = combined["revenue_band"]
    stage_a_final = make_stage_a_classifier()
    stage_a_final.fit(X_combined, y_combined_band)
    test_probabilities = probability_frame(stage_a_final, X_test)
    combined_probabilities = out_of_fold_probabilities(X_combined, y_combined_band)

    stage_b_final = make_stage_b_regressor()
    stage_b_final.fit(
        add_stage_a_probabilities(X_combined, combined_probabilities),
        np.log1p(combined["worldwide_revenue_usd"]),
    )
    test_revenue_prediction = np.maximum(
        0,
        np.expm1(stage_b_final.predict(add_stage_a_probabilities(X_test, test_probabilities))),
    )

    metrics = {
        "revenue_bands": dict(zip(REVENUE_BAND_LABELS, REVENUE_BINS[1:])),
        "split": {
            "train_rows": len(train_data), "validation_rows": len(validation_data), "test_rows": len(test_data),
            "train_years": "2010-2019", "validation_years": "2020-2021", "test_years": "2022-2024",
        },
        "stage_a_multiclass_classifier": {
            "validation": multiclass_metrics(y_validation_band, validation_band_prediction, validation_probabilities.rename(columns=lambda c: c.replace("probability_", ""))),
            "test": multiclass_metrics(
                y_test_band,
                test_probabilities.idxmax(axis=1).str.replace("probability_", "", regex=False),
                test_probabilities.rename(columns=lambda c: c.replace("probability_", "")),
            ),
        },
        "stage_b_single_revenue_regressor": {
            "validation": regression_metrics(validation_data["worldwide_revenue_usd"], validation_revenue_prediction),
            "test": regression_metrics(test_data["worldwide_revenue_usd"], test_revenue_prediction),
        },
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(stage_a_final, "models/multiclass_revenue_band_classifier.joblib")
    joblib.dump(stage_b_final, "models/multiclass_two_stage_revenue_regressor.joblib")
    Path("models/multiclass_two_stage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
