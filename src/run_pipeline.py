"""Run preparation, standard modeling, evaluation, and result export.

This entry point uses raw TMDB extracts already present in data/raw and does not
call the API. EDA, large-run paths, and the standalone blockbuster classifier
remain separate commands so each stage has an explicit input and output.
"""
import json
from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_movie_data
from src.feature_engineering import build_features
from src.train_model import train
from src.evaluate_models import evaluate


def create_results_summary(metrics_path=Path("models/metrics.json"), output_path=Path("reports/model_results.csv")):
    """Convert nested model metrics into one comparison table.

    Monetary metrics remain in dollars so they can be interpreted directly;
    classification metrics are converted to percentages for a recruiter- and
    business-friendly summary. Validation and test results stay separate so
    model selection is not confused with final out-of-sample performance.
    """
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []

    for model_name, values in metrics.get("regression", {}).items():
        for split in ("validation", "test"):
            result = values.get(split, {})
            rows.append({
                "task": "Regression",
                "model": model_name,
                "split": split,
                "MAE_usd": result.get("MAE"),
                "RMSE_usd": result.get("RMSE"),
                "R2": result.get("R2"),
                "accuracy_pct": None,
                "precision_pct": None,
                "recall_pct": None,
                "F1_pct": None,
                "ROC_AUC_pct": None,
            })

    for model_name, values in metrics.get("classification", {}).items():
        result = values.get("test", {})
        rows.append({
            "task": "Classification",
            "model": model_name,
            "split": "test",
            "MAE_usd": None,
            "RMSE_usd": None,
            "R2": None,
            "accuracy_pct": result.get("accuracy", 0) * 100,
            "precision_pct": result.get("precision", 0) * 100,
            "recall_pct": result.get("recall", 0) * 100,
            "F1_pct": result.get("F1", 0) * 100,
            "ROC_AUC_pct": result.get("ROC_AUC", 0) * 100,
            "threshold": values.get("threshold"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.4f")
    return output_path


def run_pipeline():
    """Run cleaning, feature creation, standard training, and evaluation.

    Random Forest is evaluated here because it is the selected revenue model in
    the current large experiment. Alternative models remain available in the
    saved metrics for comparison.
    """
    clean_movie_data()
    build_features()
    train()
    evaluate(model_name="random_forest")
    output_path = create_results_summary()
    print(f"\nSaved organized model results to {output_path}")


if __name__ == "__main__":
    run_pipeline()
