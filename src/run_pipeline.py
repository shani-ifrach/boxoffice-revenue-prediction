"""Reproduce the final 35-feature model, evaluation and descriptive analysis."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_movie_data
from src.data_quality import build_quality_report, validate_analysis_outputs
from src.eda import run_eda
from src.evaluate_models import evaluate
from src.feature_engineering import build_features
from src.merge_raw_extracts import merge_raw_extracts
from src.train_model import PRE_RELEASE_FEATURES, train


def create_results_summary(metrics_path, output_path=Path("reports/model_results.csv")):
    """Flatten model metrics into the summary table consumed by Tableau."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")); rows = []
    for row in metrics["regression_cv"]:
        rows.append({"task": "Regression", "model": row["model"], "split": f"validation_{row['validation_years']}", "MAE_usd": row["MAE"], "RMSE_usd": row["RMSE"], "R2": row["R2"]})
    rows.append({"task": "Regression", "model": metrics["selected_revenue_model"], "split": "final_test", "MAE_usd": metrics["regression_test"]["MAE"], "RMSE_usd": metrics["regression_test"]["RMSE"], "R2": metrics["regression_test"]["R2"]})
    for task, values in metrics["classification"].items():
        result = values["test"]
        rows.append({"task": task.title(), "model": values["selected_model"], "split": "final_test", "accuracy_pct": result["accuracy"]*100, "precision_pct": result["precision"]*100, "recall_pct": result["recall"]*100, "F1_pct": result["F1"]*100, "ROC_AUC_pct": result["ROC_AUC"]*100, "PR_AUC_pct": result["PR_AUC"]*100, "Brier": result["Brier"], "threshold": values["threshold"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.4f")


def run_pipeline(skip_merge=False):
    """Run the final model and validate its existing output tables."""
    if not skip_merge: merge_raw_extracts()
    clean_movie_data(); build_quality_report(); build_features()
    model_dir = Path("models/reduced")
    report_dir = Path("reports")
    train(PRE_RELEASE_FEATURES, output_dir=model_dir)
    evaluate(model_dir / "test_predictions.csv", output_dir=report_dir)
    validate_analysis_outputs(Path("data/processed/movies_clean.csv"),
                              Path("data/processed/movies_features.csv"),
                              model_dir / "test_predictions.csv",
                              report_dir / "actual_vs_predicted.csv")
    run_eda(output_dir=report_dir)
    create_results_summary(model_dir / "metrics.json", report_dir / "model_results.csv")
    print("Pipeline complete: final models in models/reduced; analysis in reports.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args(); run_pipeline(skip_merge=args.skip_merge)


if __name__ == "__main__": main()
