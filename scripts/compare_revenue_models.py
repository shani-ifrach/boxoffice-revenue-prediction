"""Compare revenue regressors using the production temporal protocol.

This experiment leaves production code and artifacts unchanged. It uses the same
35-feature contract, log-revenue target, three rolling validation windows, and
2022--2024 evaluation period as ``src.train_model``. That period is reported only;
the model ranking is by mean rolling-validation MAE.

Install experiment dependencies and run all candidates:
    .venv/bin/pip install -r requirements-comparison.txt
    .venv/bin/python scripts/compare_revenue_models.py
"""
import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED
from src.train_model import (
    CATEGORICAL_FEATURES,
    PRE_RELEASE_FEATURES,
    ROLLING_FOLDS,
    _safe_dollars,
    make_pipeline,
    regression_metrics,
)


def _dense_pipeline(estimator):
    """Use the production preprocessing, but dense encoding for histogram boosting."""
    numeric = [column for column in PRE_RELEASE_FEATURES if column not in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


class _CatBoostInputPreparer(BaseEstimator, TransformerMixin):
    """Keep categorical columns native for CatBoost while making missing values valid."""

    def fit(self, features, target=None):
        return self

    def transform(self, features):
        prepared = features.loc[:, PRE_RELEASE_FEATURES].copy()
        for column in CATEGORICAL_FEATURES:
            prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
        return prepared


def _model_factories():
    """Return the original candidates and the later exploratory candidates."""
    factories = {
        "ridge_current": lambda: make_pipeline(Ridge(alpha=10.0), PRE_RELEASE_FEATURES),
        "random_forest_current": lambda: make_pipeline(
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            PRE_RELEASE_FEATURES,
        ),
        "gradient_boosting_current": lambda: make_pipeline(
            GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=2,
                loss="huber",
                random_state=RANDOM_SEED,
            ),
            PRE_RELEASE_FEATURES,
        ),
        "hist_gradient_boosting": lambda: _dense_pipeline(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=5.0,
                random_state=RANDOM_SEED,
            )
        ),
    }
    missing = []
    try:
        from catboost import CatBoostRegressor

        factories["catboost_native_categories"] = lambda: Pipeline(
            [
                ("preprocess", _CatBoostInputPreparer()),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.03,
                        depth=6,
                        l2_leaf_reg=5.0,
                        loss_function="RMSE",
                        random_seed=RANDOM_SEED,
                        verbose=False,
                        thread_count=-1,
                        cat_features=CATEGORICAL_FEATURES,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    except ImportError:
        missing.append("catboost")
    try:
        from lightgbm import LGBMRegressor

        factories["lightgbm"] = lambda: make_pipeline(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
            PRE_RELEASE_FEATURES,
        )
    except ImportError:
        missing.append("lightgbm")
    try:
        from xgboost import XGBRegressor

        factories["xgboost"] = lambda: make_pipeline(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            PRE_RELEASE_FEATURES,
        )
    except ImportError:
        missing.append("xgboost")
    return factories, missing


def run(input_path: Path, output_dir: Path, requested_models=None) -> pd.DataFrame:
    """Run all requested available models and persist isolated comparison outputs."""
    movies = pd.read_csv(input_path, parse_dates=["release_date"]).sort_values(
        ["release_date", "tmdb_id"]
    )
    development = movies[movies.release_year <= 2021].copy()
    evaluation = movies[movies.release_year.between(2022, 2024)].copy()
    if evaluation.empty:
        raise ValueError("Evaluation period 2022-2024 is empty.")

    factories, missing = _model_factories()
    requested = requested_models or [
        "ridge_current",
        "random_forest_current",
        "gradient_boosting_current",
        "hist_gradient_boosting",
        "catboost_native_categories",
        "lightgbm",
        "xgboost",
    ]
    unavailable = sorted(set(requested) - set(factories))
    if unavailable:
        raise ValueError(
            f"Requested models are unavailable: {', '.join(unavailable)}. "
            "Install requirements-comparison.txt for the full comparison."
        )

    rows = []
    for name in requested:
        for train_end, validation_start, validation_end in ROLLING_FOLDS:
            train = development[development.release_year <= train_end]
            validation = development[
                development.release_year.between(validation_start, validation_end)
            ]
            model = factories[name]().fit(
                train[PRE_RELEASE_FEATURES], np.log1p(train.worldwide_revenue_usd)
            )
            prediction = _safe_dollars(
                model.predict(validation[PRE_RELEASE_FEATURES]), train.worldwide_revenue_usd
            )
            rows.append(
                {
                    "model": name,
                    "split": "rolling_validation",
                    "train_end_year": train_end,
                    "evaluation_years": f"{validation_start}-{validation_end}",
                    "train_rows": len(train),
                    "evaluation_rows": len(validation),
                    **regression_metrics(validation.worldwide_revenue_usd, prediction),
                }
            )

        final_model = factories[name]().fit(
            development[PRE_RELEASE_FEATURES], np.log1p(development.worldwide_revenue_usd)
        )
        prediction = _safe_dollars(
            final_model.predict(evaluation[PRE_RELEASE_FEATURES]), development.worldwide_revenue_usd
        )
        rows.append(
            {
                "model": name,
                "split": "evaluation_not_used_for_selection",
                "train_end_year": 2021,
                "evaluation_years": "2022-2024",
                "train_rows": len(development),
                "evaluation_rows": len(evaluation),
                **regression_metrics(evaluation.worldwide_revenue_usd, prediction),
            }
        )

    results = pd.DataFrame(rows)
    validation = results[results.split == "rolling_validation"]
    summary = (
        validation.groupby("model", as_index=False)
        .agg(
            mean_rolling_mae_usd=("MAE", "mean"),
            std_rolling_mae_usd=("MAE", "std"),
            mean_rolling_rmse_usd=("RMSE", "mean"),
            mean_rolling_r2=("R2", "mean"),
        )
        .merge(
            results[results.split == "evaluation_not_used_for_selection"][["model", "MAE", "RMSE", "R2"]]
            .rename(columns={"MAE": "evaluation_mae_usd", "RMSE": "evaluation_rmse_usd", "R2": "evaluation_r2"}),
            on="model",
            validate="one_to_one",
        )
        .sort_values("mean_rolling_mae_usd")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "by_split.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input": str(input_path),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "models_run": requested,
                "estimator_parameters": {
                    name: {
                        key: repr(value)
                        for key, value in factories[name]().named_steps["model"].get_params(deep=False).items()
                    }
                    for name in requested
                },
                "packages": {
                    "python": platform.python_version(),
                    **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost", "lightgbm") if name not in missing},
                },
                "random_seed": RANDOM_SEED,
                "features": PRE_RELEASE_FEATURES,
                "rolling_folds": ROLLING_FOLDS,
                "selection_metric": "lowest mean MAE over rolling temporal validation",
                "development_period": "2010-2021",
                "evaluation_period": "2022-2024",
                "evaluation_used_for_selection": False,
                "evaluation_previously_inspected_in_legacy_experiments": True,
                "production_artifacts_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset, e.g. hist_gradient_boosting catboost_native_categories",
    )
    args = parser.parse_args()
    summary = run(args.input_path, args.output_dir, args.models)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nWrote isolated experiment outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
