"""Compare revenue-model performance after removing feature groups."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.train_model import FULL_PRE_RELEASE_FEATURES, PRE_RELEASE_FEATURES, make_preprocessor


FEATURE_GROUPS = {
    "full_model": FULL_PRE_RELEASE_FEATURES,
    "basic_features_only": [
        "budget_usd", "log_budget_usd", "runtime_minutes", "release_year", "release_month",
        "release_season", "primary_genre", "original_language", "genre_count", "country_count",
        "company_count", "cast_size_top10", "is_franchise",
    ],
    "without_franchise_history": [feature for feature in FULL_PRE_RELEASE_FEATURES if not feature.startswith("franchise_previous_")],
    "without_director_history": [feature for feature in FULL_PRE_RELEASE_FEATURES if not feature.startswith("director_previous_")],
    "without_company_history": [feature for feature in FULL_PRE_RELEASE_FEATURES if not feature.startswith("production_company_previous_")],
    "without_cast_history": [feature for feature in FULL_PRE_RELEASE_FEATURES if not feature.startswith("cast_previous_")],
    "budget_and_history": [
        "budget_usd", "log_budget_usd", "franchise_previous_movie_count", "franchise_previous_avg_revenue",
        "franchise_previous_success_rate", "director_previous_movie_count", "director_previous_avg_revenue",
        "director_previous_success_rate", "production_company_previous_movie_count",
        "production_company_previous_avg_revenue", "production_company_previous_success_rate",
        "cast_previous_movie_count", "cast_previous_avg_revenue", "cast_previous_success_rate",
    ],
    "recommended_reduced_model": [
        feature for feature in FULL_PRE_RELEASE_FEATURES
        if feature not in {
            "production_company_previous_max_revenue",
            "company_count",
            "log_budget_usd",
        }
    ],
}


def build_single_feature_ablation_groups():
    """Create leave-one-feature-out experiments for every model feature."""
    return {
        f"without_{feature}": [candidate for candidate in FULL_PRE_RELEASE_FEATURES if candidate != feature]
        for feature in FULL_PRE_RELEASE_FEATURES
    }


def make_regression_pipeline(features):
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestRegressor(
            n_estimators=120, min_samples_leaf=4, max_features=0.8,
            random_state=42, n_jobs=-1,
        )),
    ])


def metrics(actual, predicted):
    return {
        "MAE_usd": mean_absolute_error(actual, predicted),
        "RMSE_usd": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def run_ablation(input_path=Path("data/processed/movies_features.csv"), output_dir=Path("reports")):
    """Measure whether feature groups improve a fixed Random Forest setup.

    The comparison uses rolling validation only. The final 2022-2024 holdout is
    never opened for feature selection.
    """
    movies = pd.read_csv(input_path, parse_dates=["release_date"])
    movies = movies[movies["worldwide_revenue_usd"].notna()].copy()
    movies = movies.sort_values("release_date")
    rows = []
    experiments = {**FEATURE_GROUPS, **build_single_feature_ablation_groups()}
    for experiment_name, features in experiments.items():
        for train_end, validation_start, validation_end in [(2015, 2016, 2017), (2017, 2018, 2019), (2019, 2020, 2021)]:
            train_data = movies[movies.release_year <= train_end]
            validation_data = movies[movies.release_year.between(validation_start, validation_end)]
            model = make_regression_pipeline(features)
            model.fit(train_data[features], np.log1p(train_data["worldwide_revenue_usd"]))
            prediction = np.maximum(0, np.expm1(model.predict(validation_data[features])))
            row = {"experiment": experiment_name, "feature_count": len(features), "train_end": train_end, "validation_years": f"{validation_start}-{validation_end}"}
            row.update({f"validation_{key}": value for key, value in metrics(validation_data["worldwide_revenue_usd"], prediction).items()})
            rows.append(row)

    results = pd.DataFrame(rows).sort_values("validation_MAE_usd")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "feature_ablation_results.csv", index=False, float_format="%.4f")
    print(results.to_string(index=False))
    print(f"\nSaved feature ablation results to {output_dir / 'feature_ablation_results.csv'}")

    averages = results.groupby(["experiment", "feature_count"], as_index=False).agg(validation_MAE_usd=("validation_MAE_usd", "mean"), validation_RMSE_usd=("validation_RMSE_usd", "mean"), validation_R2=("validation_R2", "mean"))
    averages.to_csv(output_dir / "feature_ablation_summary.csv", index=False, float_format="%.4f")
    full_validation_mae = averages.loc[averages["experiment"].eq("full_model"), "validation_MAE_usd"].iloc[0]
    single_feature_names = {f"without_{feature}" for feature in FULL_PRE_RELEASE_FEATURES}
    single_feature = averages[averages["experiment"].isin(single_feature_names)].copy()
    single_feature["validation_mae_change_usd"] = single_feature["validation_MAE_usd"] - full_validation_mae
    single_feature = single_feature.sort_values("validation_mae_change_usd", ascending=False)
    single_feature.to_csv(output_dir / "single_feature_ablation_results.csv", index=False, float_format="%.4f")
    print("\nLeave-one-feature-out results, ranked by validation MAE increase:")
    print(single_feature[["experiment", "validation_mae_change_usd"]].to_string(index=False))
    print(f"\nSaved {output_dir / 'single_feature_ablation_results.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run_ablation(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
