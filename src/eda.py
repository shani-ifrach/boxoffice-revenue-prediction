"""Create reproducible EDA tables, charts, and data-backed findings.

EDA is descriptive rather than predictive. Revenue, ROI, votes, and popularity
may be used here to understand historical patterns, but they must not be
presented as pre-release inputs to the production model.
"""
from pathlib import Path

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGURE_SIZE = (10, 6)


def save_table(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def run_eda(input_path=Path("data/processed/movies_features.csv"), output_dir=Path("reports")):
    """Write summary tables, charts, and a short findings file.

    Group comparisons use medians where possible because a small number of
    blockbusters creates strong right skew in revenue and ROI. Genre summaries
    require at least 20 movies to avoid presenting unstable results from tiny
    categories as business conclusions.
    """
    movies = pd.read_csv(input_path)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    financial = movies.dropna(subset=["budget_usd", "worldwide_revenue_usd"]).copy()
    financial["budget_category"] = pd.qcut(financial["budget_usd"], q=4, labels=["Low", "Mid", "High", "Very high"], duplicates="drop")

    # Revenue is highly right-skewed, so medians are used for group comparisons
    # and log axes are used for the budget/revenue relationship.
    season_summary = movies.groupby("release_season", dropna=False).agg(
        movie_count=("tmdb_id", "count"),
        median_revenue_usd=("worldwide_revenue_usd", "median"),
        mean_revenue_usd=("worldwide_revenue_usd", "mean"),
        median_roi=("roi_simple", "median"),
        profitability_rate=("profitable", "mean"),
    ).reset_index().sort_values("median_revenue_usd", ascending=False)
    save_table(season_summary, output_dir / "eda_season_summary.csv")

    genre_data = movies.assign(genre=movies["genres"].fillna("Unknown").str.split("; ")).explode("genre")
    genre_summary = genre_data.groupby("genre", dropna=False).agg(
        movie_count=("tmdb_id", "nunique"),
        median_revenue_usd=("worldwide_revenue_usd", "median"),
        median_roi=("roi_simple", "median"),
        profitability_rate=("profitable", "mean"),
    ).reset_index().query("movie_count >= 20").sort_values("median_revenue_usd", ascending=False)
    save_table(genre_summary, output_dir / "eda_genre_summary.csv")

    budget_summary = financial.groupby("budget_category", observed=True).agg(
        movie_count=("tmdb_id", "count"),
        median_budget_usd=("budget_usd", "median"),
        median_revenue_usd=("worldwide_revenue_usd", "median"),
        median_roi=("roi_simple", "median"),
        profitability_rate=("profitable", "mean"),
    ).reset_index()
    save_table(budget_summary, output_dir / "eda_budget_summary.csv")

    rating_summary = movies.groupby(pd.qcut(movies["vote_average"], q=4, duplicates="drop"), observed=False).agg(
        movie_count=("tmdb_id", "count"), median_revenue_usd=("worldwide_revenue_usd", "median"),
    ).reset_index().rename(columns={"vote_average": "rating_band"})
    save_table(rating_summary, output_dir / "eda_rating_summary.csv")

    director_summary = movies.dropna(subset=["director"]).groupby("director").agg(
        movie_count=("tmdb_id", "count"), median_revenue_usd=("worldwide_revenue_usd", "median"),
        median_roi=("roi_simple", "median"), profitability_rate=("profitable", "mean"),
    ).query("movie_count >= 3").sort_values("median_revenue_usd", ascending=False).reset_index()
    save_table(director_summary, output_dir / "eda_director_summary.csv")

    # Budget and revenue are both heavy-tailed; the log-log view prevents a few
    # blockbusters from hiding the pattern among ordinary releases.
    plt.figure(figsize=FIGURE_SIZE)
    sns.scatterplot(data=financial, x="budget_usd", y="worldwide_revenue_usd", hue="release_season", alpha=0.45)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Budget (USD, log scale)")
    plt.ylabel("Worldwide revenue (USD, log scale)")
    plt.title("Budget and worldwide revenue")
    plt.tight_layout()
    plt.savefig(figures_dir / "budget_vs_revenue.png", dpi=160)
    plt.close()

    plt.figure(figsize=FIGURE_SIZE)
    sns.boxplot(data=financial, x="budget_category", y="roi_simple", showfliers=False)
    plt.ylim(financial["roi_simple"].quantile(0.01), financial["roi_simple"].quantile(0.99))
    plt.xlabel("Budget category")
    plt.ylabel("Simple ROI (revenue - budget) / budget")
    plt.title("ROI by budget category")
    plt.tight_layout()
    plt.savefig(figures_dir / "roi_by_budget_category.png", dpi=160)
    plt.close()

    top_genres = genre_summary.head(10)
    plt.figure(figsize=FIGURE_SIZE)
    sns.barplot(data=top_genres, y="genre", x="median_revenue_usd", color="#4472C4")
    plt.xlabel("Median worldwide revenue (USD)")
    plt.ylabel("Genre")
    plt.title("Genres with the highest median revenue (minimum 20 movies)")
    plt.tight_layout()
    plt.savefig(figures_dir / "revenue_by_genre.png", dpi=160)
    plt.close()

    plt.figure(figsize=FIGURE_SIZE)
    sns.barplot(data=season_summary, x="release_season", y="median_revenue_usd", color="#70AD47")
    plt.ylabel("Median worldwide revenue (USD)")
    plt.title("Median revenue by release season")
    plt.tight_layout()
    plt.savefig(figures_dir / "revenue_by_season.png", dpi=160)
    plt.close()

    correlations = movies[["budget_usd", "worldwide_revenue_usd", "vote_average", "vote_count", "popularity", "roi_simple"]].corr(method="spearman").reset_index().rename(columns={"index": "feature"})
    save_table(correlations, output_dir / "eda_spearman_correlations.csv")

    best_season = season_summary.iloc[0]
    best_roi_season = season_summary.sort_values("median_roi", ascending=False).iloc[0]
    best_genre = genre_summary.iloc[0] if not genre_summary.empty else None
    findings = [
        "# EDA findings",
        "",
        f"- Dataset rows analyzed: {len(movies):,}.",
        f"- Movies with both positive budget and revenue: {len(financial):,}.",
        f"- Highest median-revenue season: {best_season['release_season']} (${best_season['median_revenue_usd']:,.0f}; n={int(best_season['movie_count'])}).",
        f"- Highest median-ROI season: {best_roi_season['release_season']} ({best_roi_season['median_roi']:.2f}; n={int(best_roi_season['movie_count'])}).",
    ]
    if best_genre is not None:
        findings.append(f"- Highest median-revenue genre among genres with at least 20 movies: {best_genre['genre']} (${best_genre['median_revenue_usd']:,.0f}; n={int(best_genre['movie_count'])}).")
    findings.extend([
        "",
        "These are descriptive associations, not causal effects. Genre and director comparisons should be interpreted with their sample sizes and with the source's popularity-based sampling limitation in mind.",
    ])
    (output_dir / "eda_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    print("EDA outputs written to", output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run_eda(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
