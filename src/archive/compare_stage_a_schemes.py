"""Compare different Stage A revenue classification schemes.

Each candidate uses one Stage A classifier and one Stage B regressor. Stage B
receives the Stage A probabilities as additional features; it is never split
into a separate regressor per revenue category.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.train_model import PRE_RELEASE_FEATURES, add_history_from_prior_period, make_preprocessor


CANDIDATE_SCHEMES = {
    "binary_200m": [200_000_000],
    "binary_300m": [300_000_000],
    "binary_400m": [400_000_000],
    "binary_500m": [500_000_000],
    "binary_600m": [600_000_000],
    "three_bands_100m_300m": [100_000_000, 300_000_000],
    "four_bands_100m_300m_600m": [100_000_000, 300_000_000, 600_000_000],
    "four_bands_100m_300m_500m": [100_000_000, 300_000_000, 500_000_000],
}


def make_classifier(features, n_estimators):
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3, max_features=0.8, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


def make_regressor(features, n_estimators):
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=4, max_features=0.8, random_state=42, n_jobs=-1)),
    ])


def probability_columns(labels):
    return [f"stage_a_probability_{index}" for index in range(len(labels))]


def probability_frame(model, features, labels):
    """Return probabilities in a stable column order for Stage B."""
    result = np.zeros((len(features), len(labels)))
    model_probabilities = model.predict_proba(features)
    for column_index, class_name in enumerate(model.named_steps["model"].classes_):
        result[:, labels.index(class_name)] = model_probabilities[:, column_index]
    return pd.DataFrame(result, index=features.index, columns=probability_columns(labels))


def out_of_fold_probabilities(features, target, labels, n_estimators, folds=5):
    probabilities = np.zeros((len(features), len(labels)))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for train_indices, holdout_indices in splitter.split(features, target):
        classifier = make_classifier(features.columns.tolist(), n_estimators)
        classifier.fit(features.iloc[train_indices], target.iloc[train_indices])
        probabilities[holdout_indices] = probability_frame(classifier, features.iloc[holdout_indices], labels).to_numpy()
    return pd.DataFrame(probabilities, index=features.index, columns=probability_columns(labels))


def classification_metrics(actual, predicted, probabilities, labels):
    result = {
        "stage_a_accuracy": accuracy_score(actual, predicted),
        "stage_a_macro_F1": f1_score(actual, predicted, labels=labels, average="macro", zero_division=0),
    }
    try:
        result["stage_a_ROC_AUC_ovr"] = roc_auc_score(pd.get_dummies(actual).reindex(columns=labels, fill_value=0), probabilities, multi_class="ovr")
    except ValueError:
        result["stage_a_ROC_AUC_ovr"] = np.nan
    if len(labels) == 2:
        result["stage_a_precision"] = precision_score(actual, predicted, pos_label=labels[1], zero_division=0)
        result["stage_a_recall"] = recall_score(actual, predicted, pos_label=labels[1], zero_division=0)
    return result


def regression_metrics(actual, predicted):
    return {
        "stage_b_MAE_usd": mean_absolute_error(actual, predicted),
        "stage_b_RMSE_usd": mean_squared_error(actual, predicted) ** 0.5,
        "stage_b_R2": r2_score(actual, predicted),
    }


def run_comparison(input_path, output_path, classifier_trees=250, regressor_trees=300):
    movies = pd.read_csv(input_path, parse_dates=["release_date"])
    movies = movies[movies["budget_usd"].notna() & movies["worldwide_revenue_usd"].notna()].sort_values("release_date")
    train_data = movies[movies["release_year"] <= 2019].copy()
    validation_data = movies[movies["release_year"].between(2020, 2021)].copy()
    test_data = movies[movies["release_year"] >= 2022].copy()
    validation_data = add_history_from_prior_period(validation_data, train_data)
    test_data = add_history_from_prior_period(test_data, pd.concat([train_data, validation_data]))

    rows = []
    for scheme_name, cut_points in CANDIDATE_SCHEMES.items():
        labels = [f"band_{index}" for index in range(len(cut_points) + 1)]
        bins = [-np.inf, *cut_points, np.inf]
        for frame in (train_data, validation_data, test_data):
            frame["stage_a_target"] = pd.cut(frame["worldwide_revenue_usd"], bins=bins, labels=labels).astype(str)

        X_train = train_data[PRE_RELEASE_FEATURES]
        X_validation = validation_data[PRE_RELEASE_FEATURES]
        X_test = test_data[PRE_RELEASE_FEATURES]
        stage_a_validation = make_classifier(PRE_RELEASE_FEATURES, classifier_trees)
        stage_a_validation.fit(X_train, train_data["stage_a_target"])
        validation_probabilities = probability_frame(stage_a_validation, X_validation, labels)
        validation_prediction = validation_probabilities.idxmax(axis=1).str.replace("stage_a_probability_", "band_", regex=False)
        train_probabilities = out_of_fold_probabilities(X_train, train_data["stage_a_target"], labels, classifier_trees)
        stage_b_features = PRE_RELEASE_FEATURES + probability_columns(labels)
        stage_b_validation = make_regressor(stage_b_features, regressor_trees)
        stage_b_validation.fit(X_train.join(train_probabilities), np.log1p(train_data["worldwide_revenue_usd"]))
        validation_revenue = np.maximum(0, np.expm1(stage_b_validation.predict(X_validation.join(validation_probabilities))))

        combined = pd.concat([train_data, validation_data], ignore_index=True)
        X_combined = combined[PRE_RELEASE_FEATURES]
        stage_a_final = make_classifier(PRE_RELEASE_FEATURES, classifier_trees)
        stage_a_final.fit(X_combined, combined["stage_a_target"])
        test_probabilities = probability_frame(stage_a_final, X_test, labels)
        combined_probabilities = out_of_fold_probabilities(X_combined, combined["stage_a_target"], labels, classifier_trees)
        stage_b_final = make_regressor(stage_b_features, regressor_trees)
        stage_b_final.fit(X_combined.join(combined_probabilities), np.log1p(combined["worldwide_revenue_usd"]))
        test_revenue = np.maximum(0, np.expm1(stage_b_final.predict(X_test.join(test_probabilities))))

        row = {"scheme": scheme_name, "cut_points_usd": ";".join(str(value) for value in cut_points), "stage_a_class_count": len(labels)}
        row.update({f"validation_{key}": value for key, value in classification_metrics(validation_data["stage_a_target"], validation_prediction, validation_probabilities.to_numpy(), labels).items()})
        row.update({f"validation_{key}": value for key, value in regression_metrics(validation_data["worldwide_revenue_usd"], validation_revenue).items()})
        test_prediction = test_probabilities.idxmax(axis=1).str.replace("stage_a_probability_", "band_", regex=False)
        row.update({f"test_{key}": value for key, value in classification_metrics(test_data["stage_a_target"], test_prediction, test_probabilities.to_numpy(), labels).items()})
        row.update({f"test_{key}": value for key, value in regression_metrics(test_data["worldwide_revenue_usd"], test_revenue).items()})
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("validation_stage_b_MAE_usd")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format="%.4f")
    print(results.to_string(index=False))
    print(f"\nSaved Stage A scheme comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("reports/stage_a_scheme_comparison.csv"))
    parser.add_argument("--classifier-trees", type=int, default=250)
    parser.add_argument("--regressor-trees", type=int, default=300)
    args = parser.parse_args()
    run_comparison(args.input_path, args.output_path, args.classifier_trees, args.regressor_trees)


if __name__ == "__main__":
    main()

