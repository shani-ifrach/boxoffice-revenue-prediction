"""Train a two-stage model: blockbuster probability plus revenue models."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train_model import PRE_RELEASE_FEATURES, add_history_from_prior_period, make_pipeline


def regression_metrics(actual, predicted):
    return {"MAE": mean_absolute_error(actual, predicted), "RMSE": mean_squared_error(actual, predicted) ** 0.5, "R2": r2_score(actual, predicted)}


def train():
    movies = pd.read_csv("data/processed/movies_features.csv").sort_values("release_date")
    movies = movies[movies["budget_usd"].notna() & movies["profitable"].notna()].copy()
    movies["release_year"] = movies["release_year"].astype(int)
    train_data = movies[movies["release_year"] <= 2019]
    validation_data = add_history_from_prior_period(movies[movies["release_year"].isin([2020, 2021])], train_data)
    test_data = add_history_from_prior_period(movies[movies["release_year"] >= 2022], pd.concat([train_data, validation_data]))
    cutoff = float(train_data["worldwide_revenue_usd"].quantile(0.90))
    train_label = (train_data["worldwide_revenue_usd"] >= cutoff).astype(int)
    validation_label = (validation_data["worldwide_revenue_usd"] >= cutoff).astype(int)
    classifier = make_pipeline(GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=2, random_state=42))
    classifier.fit(train_data[PRE_RELEASE_FEATURES], train_label)
    validation_probability = classifier.predict_proba(validation_data[PRE_RELEASE_FEATURES])[:, 1]
    threshold = 0.5
    regular = train_data[train_data["worldwide_revenue_usd"] < cutoff]
    high = train_data[train_data["worldwide_revenue_usd"] >= cutoff]
    regular_model = make_pipeline(RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=0.8, random_state=42, n_jobs=-1))
    high_model = make_pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1))
    regular_model.fit(regular[PRE_RELEASE_FEATURES], np.log1p(regular["worldwide_revenue_usd"]))
    high_model.fit(high[PRE_RELEASE_FEATURES], np.log1p(high["worldwide_revenue_usd"]))
    validation_prediction = (1 - validation_probability) * np.expm1(regular_model.predict(validation_data[PRE_RELEASE_FEATURES])) + validation_probability * np.expm1(high_model.predict(validation_data[PRE_RELEASE_FEATURES]))
    combined = pd.concat([train_data, validation_data])
    regular = combined[combined["worldwide_revenue_usd"] < cutoff]
    high = combined[combined["worldwide_revenue_usd"] >= cutoff]
    classifier.fit(combined[PRE_RELEASE_FEATURES], (combined["worldwide_revenue_usd"] >= cutoff).astype(int))
    regular_model.fit(regular[PRE_RELEASE_FEATURES], np.log1p(regular["worldwide_revenue_usd"]))
    high_model.fit(high[PRE_RELEASE_FEATURES], np.log1p(high["worldwide_revenue_usd"]))
    test_probability = classifier.predict_proba(test_data[PRE_RELEASE_FEATURES])[:, 1]
    test_prediction = (1 - test_probability) * np.expm1(regular_model.predict(test_data[PRE_RELEASE_FEATURES])) + test_probability * np.expm1(high_model.predict(test_data[PRE_RELEASE_FEATURES]))
    metrics = {"blockbuster_cutoff_usd": cutoff, "train_blockbusters": int(train_label.sum()), "validation_blockbusters": int(validation_label.sum()), "validation": regression_metrics(validation_data["worldwide_revenue_usd"], validation_prediction), "test": regression_metrics(test_data["worldwide_revenue_usd"], test_prediction)}
    Path("models").mkdir(exist_ok=True)
    Path("models/two_stage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(classifier, "models/blockbuster_classifier.joblib")
    joblib.dump(regular_model, "models/blockbuster_regular_regressor.joblib")
    joblib.dump(high_model, "models/blockbuster_high_regressor.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
