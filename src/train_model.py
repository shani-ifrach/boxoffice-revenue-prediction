"""Temporal model selection, final holdout evaluation, and versioned artifacts."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from src.config import BLOCKBUSTER_THRESHOLD_USD, CONFORMAL_COVERAGE, RANDOM_SEED, SCHEMA_VERSION
from src.feature_engineering import add_historical_features

FULL_PRE_RELEASE_FEATURES = ["budget_usd", "log_budget_usd", "runtime_minutes", "release_year", "release_month", "release_season", "primary_genre", "original_language", "genre_count", "country_count", "company_count", "cast_size_top10", "is_franchise", "is_sequel", "is_summer_release", "is_holiday_release", "franchise_previous_movie_count", "franchise_previous_avg_revenue", "franchise_previous_last_revenue", "franchise_previous_max_revenue", "franchise_previous_median_revenue", "franchise_previous_success_rate", "franchise_previous_latest_success", "franchise_previous_avg_rating", "director_previous_movie_count", "director_previous_avg_revenue", "director_previous_max_revenue", "director_previous_success_rate", "director_previous_avg_rating", "production_company_previous_movie_count", "production_company_previous_avg_revenue", "production_company_previous_max_revenue", "production_company_previous_success_rate", "cast_previous_movie_count", "cast_previous_avg_revenue", "cast_previous_max_revenue", "cast_previous_success_rate", "cast_previous_avg_rating"]
REMOVED_FEATURES = {"production_company_previous_max_revenue", "company_count", "log_budget_usd"}
REDUCED_PRE_RELEASE_FEATURES = [feature for feature in FULL_PRE_RELEASE_FEATURES if feature not in REMOVED_FEATURES]
PRE_RELEASE_FEATURES = REDUCED_PRE_RELEASE_FEATURES
CATEGORICAL_FEATURES = ["release_season", "primary_genre", "original_language"]
ROLLING_FOLDS = [(2015, 2016, 2017), (2017, 2018, 2019), (2019, 2020, 2021)]


def add_history_from_prior_period(target, history):
    """Backward-compatible name for archived experiments; uses strict dates."""
    return add_historical_features(target, history)


def make_preprocessor(features):
    """Build preprocessing that imputes missing values and encodes categories."""
    numeric = [c for c in features if c not in CATEGORICAL_FEATURES]
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def make_pipeline(estimator, features=PRE_RELEASE_FEATURES):
    """Wrap an estimator with the project preprocessing contract."""
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", estimator)])


def regression_metrics(actual, predicted):
    """Return the regression metrics reported for model selection and QA."""
    return {"MAE": float(mean_absolute_error(actual, predicted)), "RMSE": float(mean_squared_error(actual, predicted) ** 0.5), "R2": float(r2_score(actual, predicted))}


def classification_metrics(actual, predicted, probability):
    """Return threshold, ranking, calibration, and confusion-matrix metrics."""
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(actual, predicted)), "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0)), "F1": float(f1_score(actual, predicted, zero_division=0)), "ROC_AUC": float(roc_auc_score(actual, probability)), "PR_AUC": float(average_precision_score(actual, probability)), "Brier": float(brier_score_loss(actual, probability)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def _regressors():
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=4, max_features=0.8, random_state=RANDOM_SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.04, max_depth=2, loss="huber", random_state=RANDOM_SEED),
    }


def _classifiers():
    return {
        "logistic_regression": LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=4, max_features=0.8, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=2, random_state=RANDOM_SEED),
    }


def _safe_dollars(log_predictions, training_revenue):
    # Training-derived ceiling prevents Ridge overflow without consulting holdout.
    ceiling = float(np.log1p(training_revenue.max() * 2))
    return np.expm1(np.clip(log_predictions, 0, ceiling))


def _choose_threshold(actual, probability, minimum_recall=0.80):
    candidates = np.linspace(0.05, 0.95, 181)
    eligible = [t for t in candidates if recall_score(actual, probability >= t, zero_division=0) >= minimum_recall]
    if eligible:
        return float(max(eligible, key=lambda t: (precision_score(actual, probability >= t, zero_division=0), f1_score(actual, probability >= t, zero_division=0))))
    return float(max(candidates, key=lambda t: f1_score(actual, probability >= t, zero_division=0)))


def _temporal_regression_selection(data, features):
    rows, residuals = [], {}
    for name, estimator in _regressors().items():
        residuals[name] = []
        for train_end, val_start, val_end in ROLLING_FOLDS:
            train = data[data.release_year <= train_end]
            val = data[data.release_year.between(val_start, val_end)]
            model = make_pipeline(estimator, features)
            model.fit(train[features], np.log1p(train.worldwide_revenue_usd))
            prediction = _safe_dollars(model.predict(val[features]), train.worldwide_revenue_usd)
            residuals[name].extend(np.abs(val.worldwide_revenue_usd.to_numpy() - prediction))
            rows.append({"model": name, "train_end": train_end, "validation_years": f"{val_start}-{val_end}", "train_rows": len(train), "validation_rows": len(val), **regression_metrics(val.worldwide_revenue_usd, prediction)})
    table = pd.DataFrame(rows)
    selected = table.groupby("model").MAE.mean().idxmin()
    return selected, table, np.asarray(residuals[selected])


def _temporal_classifier_selection(data, label, features):
    rows, pooled = [], {}
    for name, estimator in _classifiers().items():
        ys, ps = [], []
        for train_end, val_start, val_end in ROLLING_FOLDS:
            train = data[data.release_year <= train_end]
            val = data[data.release_year.between(val_start, val_end)]
            if train[label].nunique() < 2 or val[label].nunique() < 2: continue
            model = make_pipeline(estimator, features).fit(train[features], train[label])
            probability = model.predict_proba(val[features])[:, 1]
            ys.extend(val[label]); ps.extend(probability)
            rows.append({"model": name, "train_end": train_end, "validation_years": f"{val_start}-{val_end}", "PR_AUC": average_precision_score(val[label], probability), "ROC_AUC": roc_auc_score(val[label], probability), "Brier": brier_score_loss(val[label], probability)})
        pooled[name] = (np.asarray(ys), np.asarray(ps))
    table = pd.DataFrame(rows)
    selected = table.groupby("model").PR_AUC.mean().idxmax()
    y_oof, p_oof = pooled[selected]
    calibrator = LogisticRegression(random_state=RANDOM_SEED).fit(p_oof.reshape(-1, 1), y_oof)
    calibrated = calibrator.predict_proba(p_oof.reshape(-1, 1))[:, 1]
    threshold = _choose_threshold(y_oof, calibrated)
    return selected, table, calibrator, threshold, classification_metrics(y_oof, calibrated >= threshold, calibrated), float(brier_score_loss(y_oof, p_oof)), y_oof, p_oof, calibrated


def _artifact(kind, model, features, **extra):
    return {"artifact_version": 2, "schema_version": SCHEMA_VERSION, "kind": kind, "features": list(features), "feature_dtypes": {f: ("category" if f in CATEGORICAL_FEATURES else "number") for f in features}, "model": model, **extra}


def train(features=PRE_RELEASE_FEATURES, input_path=Path("data/processed/movies_features.csv"), output_dir=Path("models/reduced")):
    """Select, fit, evaluate, and persist the governed production artifacts."""
    movies = pd.read_csv(input_path, parse_dates=["release_date"]).sort_values(["release_date", "tmdb_id"])
    development = movies[movies.release_year <= 2021].copy()
    test = movies[movies.release_year.between(2022, 2024)].copy()
    if test.empty: raise ValueError("Final holdout 2022-2024 is empty; collect all required years before training.")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, regression_cv, residuals = _temporal_regression_selection(development, features)
    reg_model = make_pipeline(_regressors()[selected], features).fit(development[features], np.log1p(development.worldwide_revenue_usd))
    test_prediction = _safe_dollars(reg_model.predict(test[features]), development.worldwide_revenue_usd)
    q = float(np.quantile(residuals, CONFORMAL_COVERAGE, method="higher"))
    reg_artifact = _artifact("revenue_regression", reg_model, features, target_transform="log1p", prediction_cap_usd=float(development.worldwide_revenue_usd.max() * 2), conformal_coverage=CONFORMAL_COVERAGE, conformal_absolute_error_usd=q, selected_model=selected)

    classifier_results, classifier_artifacts = {}, {}
    for label, population in {
        "profitability": development[development.profitable.notna()].assign(profitability=lambda x: x.profitable.astype(int)),
        "blockbuster": development.assign(blockbuster=lambda x: (x.worldwide_revenue_usd > BLOCKBUSTER_THRESHOLD_USD).astype(int)),
    }.items():
        chosen, cv, calibrator, threshold, validation_metrics, uncalibrated_brier, y_oof, raw_oof, calibrated_oof = _temporal_classifier_selection(population, label, features)
        final_model = make_pipeline(_classifiers()[chosen], features).fit(population[features], population[label])
        test_population = test[test.profitable.notna()].assign(profitability=lambda x: x.profitable.astype(int)) if label == "profitability" else test.assign(blockbuster=lambda x: (x.worldwide_revenue_usd > BLOCKBUSTER_THRESHOLD_USD).astype(int))
        raw = final_model.predict_proba(test_population[features])[:, 1]
        prob = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        classifier_results[label] = {"selected_model": chosen, "threshold": threshold, "validation_oof": validation_metrics, "validation_uncalibrated_brier": uncalibrated_brier, "test": classification_metrics(test_population[label], prob >= threshold, prob), "cv": cv.to_dict("records")}
        classifier_artifacts[label] = _artifact(f"{label}_classification", final_model, features, calibrator=calibrator, decision_threshold=threshold, probability_label="calibrated_probability", selected_model=chosen, blockbuster_threshold_usd=BLOCKBUSTER_THRESHOLD_USD if label == "blockbuster" else None)
        curve_rows = []
        for status, values in (("before", raw_oof), ("after", calibrated_oof)):
            observed, predicted = calibration_curve(y_oof, values, n_bins=8, strategy="quantile")
            curve_rows.extend({"calibration": status, "mean_predicted": float(p), "observed_rate": float(o)} for o, p in zip(observed, predicted))
        curve_frame = pd.DataFrame(curve_rows)
        curve_frame.to_csv(output_dir / f"{label}_calibration_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        for status, group in curve_frame.groupby("calibration"):
            ax.plot(group.mean_predicted, group.observed_rate, marker="o", label=status)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title=f"{label.title()} calibration")
        ax.legend(); fig.tight_layout(); fig.savefig(output_dir / f"{label}_calibration_curve.png", dpi=160); plt.close(fig)

    median = float(development.worldwide_revenue_usd.median())
    metrics = {"schema_version": SCHEMA_VERSION, "feature_count": len(features), "selected_revenue_model": selected, "regression_cv": regression_cv.to_dict("records"), "regression_test": regression_metrics(test.worldwide_revenue_usd, test_prediction), "regression_baseline_test": regression_metrics(test.worldwide_revenue_usd, np.repeat(median, len(test))), "conformal": {"nominal_coverage": CONFORMAL_COVERAGE, "absolute_error_usd": q, "test_empirical_coverage": float(np.mean((test.worldwide_revenue_usd >= np.maximum(0, test_prediction-q)) & (test.worldwide_revenue_usd <= test_prediction+q))), "test_average_width_usd": float(np.mean((test_prediction+q)-np.maximum(0, test_prediction-q)))}, "classification": classifier_results, "split": {"development_years": "2010-2021", "development_rows": len(development), "test_years": "2022-2024", "test_rows": len(test)}}
    joblib.dump(reg_artifact, output_dir / "revenue_regression.joblib")
    joblib.dump(classifier_artifacts["profitability"], output_dir / "profitability_classifier.joblib")
    joblib.dump(classifier_artifacts["blockbuster"], output_dir / "blockbuster_classifier.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {"trained_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "packages": {"pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}, "seed": RANDOM_SEED, "input": str(input_path), "input_rows": len(movies), "feature_count": len(features), "selected_model": selected, "split_definition": metrics["split"], "final_holdout_used_for_selection": False}
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    predictions = test[["tmdb_id", "title", "release_year", "genres", "budget_usd", "worldwide_revenue_usd"]].copy()
    predictions["predicted_revenue_usd"] = test_prediction
    predictions["prediction_lower_usd"] = np.maximum(0, test_prediction - q)
    predictions["prediction_upper_usd"] = test_prediction + q
    predictions["absolute_error_usd"] = np.abs(predictions.worldwide_revenue_usd - test_prediction)
    bb_art = classifier_artifacts["blockbuster"]
    bb_raw = bb_art["model"].predict_proba(test[features])[:, 1]
    predictions["blockbuster_probability"] = bb_art["calibrator"].predict_proba(bb_raw.reshape(-1, 1))[:, 1]
    predictions["blockbuster_actual"] = (test.worldwide_revenue_usd > BLOCKBUSTER_THRESHOLD_USD).astype(int).to_numpy()
    predictions["blockbuster_predicted"] = (predictions.blockbuster_probability >= bb_art["decision_threshold"]).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    def dollar_mae_score(estimator, x, actual):
        return -mean_absolute_error(actual, _safe_dollars(estimator.predict(x), development.worldwide_revenue_usd))
    importance = permutation_importance(reg_model, test[features], test.worldwide_revenue_usd, scoring=dollar_mae_score, n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
    importance_frame = pd.DataFrame({"feature": features, "mae_increase_usd_mean": importance.importances_mean, "mae_increase_usd_std": importance.importances_std}).sort_values("mae_increase_usd_mean", ascending=False)
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)
    top = importance_frame.head(12).sort_values("mae_increase_usd_mean")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.barh(top.feature, top.mae_increase_usd_mean, xerr=top.mae_increase_usd_std)
    ax.set(xlabel="Increase in MAE after permutation (USD)", title="Holdout permutation importance")
    fig.tight_layout(); fig.savefig(output_dir / "permutation_importance.png", dpi=160); plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__": train()
