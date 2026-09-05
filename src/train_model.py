"""Train leakage-aware revenue and profitability models."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FULL_PRE_RELEASE_FEATURES = ["budget_usd", "log_budget_usd", "runtime_minutes", "release_year", "release_month", "release_season", "primary_genre", "original_language", "genre_count", "country_count", "company_count", "cast_size_top10", "is_franchise", "is_sequel", "is_summer_release", "is_holiday_release", "franchise_previous_movie_count", "franchise_previous_avg_revenue", "franchise_previous_last_revenue", "franchise_previous_max_revenue", "franchise_previous_median_revenue", "franchise_previous_success_rate", "franchise_previous_latest_success", "franchise_previous_avg_rating", "director_previous_movie_count", "director_previous_avg_revenue", "director_previous_max_revenue", "director_previous_success_rate", "director_previous_avg_rating", "production_company_previous_movie_count", "production_company_previous_avg_revenue", "production_company_previous_max_revenue", "production_company_previous_success_rate", "cast_previous_movie_count", "cast_previous_avg_revenue", "cast_previous_max_revenue", "cast_previous_success_rate", "cast_previous_avg_rating"]
REMOVED_FEATURES = {"production_company_previous_max_revenue", "company_count", "log_budget_usd"}
REDUCED_PRE_RELEASE_FEATURES = [feature for feature in FULL_PRE_RELEASE_FEATURES if feature not in REMOVED_FEATURES]
# The production feature contract is now the validated 35-feature version.
PRE_RELEASE_FEATURES = REDUCED_PRE_RELEASE_FEATURES
CATEGORICAL_FEATURES = ["release_season", "primary_genre", "original_language"]


def make_preprocessor(features):
    """Build a reusable preprocessing step for numeric and categorical inputs.

    Numeric missing values are replaced with the training median, while
    categorical missing values use the most frequent category. One-hot encoding
    uses handle_unknown=ignore so a genuinely new genre or language does not
    crash scoring. The preprocessing remains inside the Pipeline to prevent
    fitting transformations on validation or test rows.
    """
    numeric = [column for column in features if column not in CATEGORICAL_FEATURES]
    categorical = [column for column in features if column in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def make_pipeline(estimator, features=PRE_RELEASE_FEATURES):
    """Combine the shared feature preprocessing with one estimator."""
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", estimator)])


def regression_metrics(actual, predicted):
    """Return dollar-space regression metrics for readable business reporting."""
    return {"MAE": mean_absolute_error(actual, predicted), "RMSE": mean_squared_error(actual, predicted) ** 0.5, "R2": r2_score(actual, predicted)}


def classification_metrics(actual, predicted, probability):
    """Return threshold metrics plus ROC-AUC for a binary classifier."""
    return {"accuracy": accuracy_score(actual, predicted), "precision": precision_score(actual, predicted, zero_division=0), "recall": recall_score(actual, predicted, zero_division=0), "F1": f1_score(actual, predicted, zero_division=0), "ROC_AUC": roc_auc_score(actual, probability)}


def calibration_factor(actual, predicted):
    """Learn a multiplicative correction from validation predictions only."""
    valid = (actual.to_numpy() > 0) & (predicted > 0)
    return float(np.median(actual.to_numpy()[valid] / predicted[valid]))


def add_history_from_prior_period(target, history):
    """Attach entity history using only rows from an earlier time period.

    This keeps the validation/test simulation honest: a batch of future films
    cannot use the revenue of another future film that has not been released yet.
    """
    target = target.copy()
    history = history.copy()

    def single_entity(entity, prefix):
        prior = history[history[entity].notna()].groupby(entity).agg(
            count=("tmdb_id", "count"), avg_revenue=("worldwide_revenue_usd", "mean"),
            success_rate=("profitable", "mean"),
        )
        mapped = target[entity].map(prior["count"]).rename(f"{prefix}_previous_movie_count")
        target[f"{prefix}_previous_movie_count"] = mapped
        target[f"{prefix}_previous_avg_revenue"] = target[entity].map(prior["avg_revenue"])
        target[f"{prefix}_previous_success_rate"] = target[entity].map(prior["success_rate"])

    single_entity("director", "director")
    single_entity("collection_id", "franchise")

    def list_entity(list_column, prefix):
        prior = history[["tmdb_id", list_column, "worldwide_revenue_usd", "profitable"]].copy()
        prior["entity_id"] = prior[list_column].fillna("").str.split("; ")
        prior = prior.explode("entity_id")
        prior = prior[prior["entity_id"].ne("")]
        summary = prior.groupby("entity_id").agg(count=("tmdb_id", "count"), avg_revenue=("worldwide_revenue_usd", "mean"), success_rate=("profitable", "mean"))
        expanded = target[["tmdb_id", list_column]].copy()
        expanded["entity_id"] = expanded[list_column].fillna("").str.split("; ")
        expanded = expanded.explode("entity_id")
        expanded = expanded[expanded["entity_id"].ne("")]
        expanded = expanded.join(summary, on="entity_id")
        movie_summary = expanded.groupby("tmdb_id").agg(
            **{f"{prefix}_previous_movie_count": ("count", "max"), f"{prefix}_previous_avg_revenue": ("avg_revenue", "mean"), f"{prefix}_previous_success_rate": ("success_rate", "mean")}
        )
        return target.join(movie_summary, on="tmdb_id", rsuffix="_new")

    target = list_entity("production_company_ids", "production_company")
    target = list_entity("cast_ids_top10", "cast")
    return target


def train(features=PRE_RELEASE_FEATURES):
    """Train, compare, and save the standard revenue and profitability models.

    Validation is used for model decisions and classification thresholds. After
    those decisions are made, each final model is refit on Train plus Validation
    and evaluated once on the future Test period.
    """
    movies = pd.read_csv("data/processed/movies_features.csv")
    movies = movies[movies["budget_usd"].notna() & movies["profitable"].notna()].sort_values("release_date")
    movies["release_year"] = movies["release_year"].astype(int)
    train_data = movies[movies["release_year"] <= 2019]
    validation_data = movies[movies["release_year"].isin([2020, 2021])]
    test_data = movies[movies["release_year"] >= 2022]
    if min(len(train_data), len(validation_data), len(test_data)) == 0:
        raise ValueError("Each time split must contain at least one row.")

    validation_data = add_history_from_prior_period(validation_data, train_data)
    test_data = add_history_from_prior_period(test_data, pd.concat([train_data, validation_data]))
    X_train, X_validation, X_test = train_data[features], validation_data[features], test_data[features]
    y_train_log = np.log1p(train_data["worldwide_revenue_usd"])
    y_validation_revenue, y_test_revenue = validation_data["worldwide_revenue_usd"], test_data["worldwide_revenue_usd"]
    regression_models = {"ridge": Ridge(alpha=10.0), "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=0.8, random_state=42, n_jobs=-1), "gradient_boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.04, max_depth=2, loss="huber", random_state=42)}
    classification_models = {"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"), "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.8, random_state=42, class_weight="balanced", n_jobs=-1), "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=2, random_state=42)}
    metrics = {"regression": {}, "classification": {}, "split": {"train_rows": len(train_data), "validation_rows": len(validation_data), "test_rows": len(test_data), "train_years": "2010-2019", "validation_years": "2020-2021", "test_years": "2022-2024"}}
    Path("models").mkdir(exist_ok=True)

    for name, estimator in regression_models.items():
        validation_model = make_pipeline(estimator, features)
        validation_model.fit(X_train, y_train_log)
        validation_predictions = np.maximum(0, np.expm1(validation_model.predict(X_validation)))
        factor = calibration_factor(y_validation_revenue, validation_predictions)
        final_model = make_pipeline(estimator, features)
        final_model.fit(pd.concat([train_data, validation_data])[features], np.log1p(pd.concat([train_data, validation_data])["worldwide_revenue_usd"]))
        test_predictions = np.maximum(0, np.expm1(final_model.predict(X_test)))
        calibrated_test_predictions = test_predictions * factor
        metrics["regression"][name] = {"calibration_factor": factor, "validation": regression_metrics(y_validation_revenue, validation_predictions), "test": regression_metrics(y_test_revenue, test_predictions), "test_calibrated": regression_metrics(y_test_revenue, calibrated_test_predictions)}
        joblib.dump(final_model, f"models/regression_{name}.joblib")

    y_train, y_validation, y_test = train_data["profitable"].astype(int), validation_data["profitable"].astype(int), test_data["profitable"].astype(int)
    combined = pd.concat([train_data, validation_data])
    for name, estimator in classification_models.items():
        validation_model = make_pipeline(estimator, features)
        validation_model.fit(X_train, y_train)
        validation_probability = validation_model.predict_proba(X_validation)[:, 1]
        thresholds = np.arange(0.30, 0.71, 0.05)
        best_threshold = max(thresholds, key=lambda threshold: f1_score(y_validation, validation_probability >= threshold, zero_division=0))
        final_model = make_pipeline(estimator, features)
        final_model.fit(combined[features], combined["profitable"].astype(int))
        test_probability = final_model.predict_proba(X_test)[:, 1]
        test_predictions = test_probability >= best_threshold
        metrics["classification"][name] = {"threshold": float(best_threshold), "test": classification_metrics(y_test, test_predictions, test_probability)}
        joblib.dump(final_model, f"models/classification_{name}.joblib")

    metrics["baselines"] = {"classification_majority_accuracy_test": float(max(y_test.mean(), 1 - y_test.mean())), "regression_median_revenue_test": regression_metrics(y_test_revenue, np.repeat(train_data["worldwide_revenue_usd"].median(), len(y_test_revenue)))}
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
