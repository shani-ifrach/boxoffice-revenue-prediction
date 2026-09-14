"""Export final-holdout error analysis from locked training predictions."""
import argparse
from pathlib import Path
import pandas as pd


def evaluate(predictions_path=Path("models/reduced/test_predictions.csv"), output_dir=Path("reports")):
    """Export final-holdout errors grouped by revenue, genre, year, and budget."""
    data = pd.read_csv(predictions_path)
    data["signed_error_usd"] = data["predicted_revenue_usd"] - data["worldwide_revenue_usd"]
    data["revenue_band"] = pd.cut(data.worldwide_revenue_usd, [-1, 250_000_000, 400_000_000, float("inf")], labels=["Regular (<$250M)", "High Grossing ($250M-$400M)", "Blockbuster (>$400M)"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "actual_vs_predicted.csv", index=False)
    exploded = data.assign(genre=data.genres.fillna("Unknown").str.split("; ")).explode("genre")
    by_genre = exploded.groupby("genre").agg(movie_count=("tmdb_id", "nunique"), mae_usd=("absolute_error_usd", "mean"), mean_signed_error_usd=("signed_error_usd", "mean")).reset_index()
    by_genre.to_csv(output_dir / "error_by_genre.csv", index=False)
    by_band = data.groupby("revenue_band", observed=True).agg(movie_count=("tmdb_id", "count"), mae_usd=("absolute_error_usd", "mean"), median_actual_revenue_usd=("worldwide_revenue_usd", "median"), median_predicted_revenue_usd=("predicted_revenue_usd", "median"), interval_coverage=("worldwide_revenue_usd", lambda s: float(((s >= data.loc[s.index, "prediction_lower_usd"]) & (s <= data.loc[s.index, "prediction_upper_usd"])).mean()))).reset_index()
    by_band.to_csv(output_dir / "error_by_revenue_band.csv", index=False)
    for column, name in (("release_year", "year"), ("budget_usd", "budget")):
        working = data.copy()
        if name == "budget": working["budget_band"] = pd.qcut(working.budget_usd, 4, duplicates="drop")
        group = column if name == "year" else "budget_band"
        working.groupby(group, observed=True).agg(movie_count=("tmdb_id", "count"), mae_usd=("absolute_error_usd", "mean")).reset_index().to_csv(output_dir / f"error_by_{name}.csv", index=False)
    data.nlargest(10, "signed_error_usd").to_csv(output_dir / "largest_overpredictions.csv", index=False)
    data.nsmallest(10, "signed_error_usd").to_csv(output_dir / "largest_underpredictions.csv", index=False)
    print(f"Wrote final-holdout diagnostics for {len(data):,} movies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, default=Path("models/reduced/test_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(); evaluate(args.predictions_path, args.output_dir)


if __name__ == "__main__": main()
