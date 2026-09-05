"""Compare revenue-model performance after removing feature groups."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.train_model import PRE_RELEASE_FEATURES, add_history_from_prior_period, make_preprocessor


FEATURE_GROUPS = {
    "full_model": PRE_RELEASE_FEATURES,
    "basic_features_only": [
        "budget_usd", "log_budget_usd", "runtime_minutes", "release_year", "release_month",
        "release_season", "primary_genre", "original_language", "genre_count", "country_count",
        "company_count", "cast_size_top10", "is_franchise",
    ],
    "without_franchise_history": [feature for feature in PRE_RELEASE_FEATURES if not feature.startswith("franchise_previous_")],
    "without_director_history": [feature for feature in PRE_RELEASE_FEATURES if not feature.startswith("director_previous_")],
    "without_company_history": [feature for feature in PRE_RELEASE_FEATURES if not feature.startswith("production_company_previous_")],
    "without_cast_history": [feature for feature in PRE_RELEASE_FEATURES if not feature.startswith("cast_previous_")],
    "budget_and_history": [
        "budget_usd", "log_budget_usd", "franchise_previous_movie_count", "franchise_previous_avg_revenue",
        "franchise_previous_success_rate", "director_previous_movie_count", "director_previous_avg_revenue",
        "director_previous_success_rate", "production_company_previous_movie_count",
        "production_company_previous_avg_revenue", "production_company_previous_success_rate",
        "cast_previous_movie_count", "cast_previous_avg_revenue", "cast_previous_success_rate",
    ],
}


def make_regression_pipeline(features):
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=4, max_features=0.8,
            random_state=42, n_jobs=-1,
        )),
    ])


def metrics(actual, predicted):
    return {
        "MAE_usd": mean_absolute_error(actual, predicted),
        "RMSE_usd": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def run_ablation(input_path=Path("data/processed/movies_features.csv")):
    movies = pd.read_csv(input_path, parse_dates=["release_date"])
    movies = movies[movies["budget_usd"].notna() & movies["worldwide_revenue_usd"].notna()].copy()
    movies = movies.sort_values("release_date")
    train_data = movies[movies["release_year"] <= 2019].copy()
    validation_data = movies[movies["release_year"].between(2020, 2021)].copy()
    test_data = movies[movies["release_year"] >= 2022].copy()
    validation_data = add_history_from_prior_period(validation_data, train_data)
    test_data = add_history_from_prior_period(test_data, pd.concat([train_data, validation_data]))

    rows = []
    for experiment_name, features in FEATURE_GROUPS.items():
        model = make_regression_pipeline(features)
        model.fit(train_data[features], np.log1p(train_data["worldwide_revenue_usd"]))
        validation_prediction = np.maximum(0, np.expm1(model.predict(validation_data[features])))
        test_prediction = np.maximum(0, np.expm1(model.predict(test_data[features])))
        row = {"experiment": experiment_name, "feature_count": len(features)}
        row.update({f"validation_{key}": value for key, value in metrics(validation_data["worldwide_revenue_usd"], validation_prediction).items()})
        row.update({f"test_{key}": value for key, value in metrics(test_data["worldwide_revenue_usd"], test_prediction).items()})
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("validation_MAE_usd")
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/feature_ablation_results.csv", index=False, float_format="%.4f")
    print(results.to_string(index=False))
    print("\nSaved feature ablation results to reports/feature_ablation_results.csv")


if __name__ == "__main__":
    run_ablation()
