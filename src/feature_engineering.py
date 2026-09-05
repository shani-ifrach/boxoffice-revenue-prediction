"""Create leakage-aware features for analysis and modeling."""
from pathlib import Path

import numpy as np
import pandas as pd


def build_features(input_path=Path("data/processed/movies_clean.csv"), output_path=Path("data/processed/movies_features.csv")):
    movies = pd.read_csv(input_path, parse_dates=["release_date"])
    movies["release_month"] = movies["release_date"].dt.month
    movies["release_season"] = movies["release_month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
    movies["primary_genre"] = movies["genres"].fillna("Unknown").str.split("; ").str[0].replace("", "Unknown")
    movies["genre_count"] = movies["genres"].fillna("").str.count(";") + movies["genres"].fillna("").str.len().gt(0).astype(int)
    movies["country_count"] = movies["production_countries"].fillna("").apply(lambda value: len([item for item in value.split("; ") if item]))
    movies["company_count"] = movies["production_companies"].fillna("").apply(lambda value: len([item for item in value.split("; ") if item]))
    movies["is_franchise"] = movies["collection_id"].notna().astype(int)
    movies["is_summer_release"] = movies["release_month"].isin([6, 7, 8]).astype(int)
    movies["is_holiday_release"] = movies["release_month"].isin([11, 12]).astype(int)
    movies["cast_size_top10"] = movies["cast_top10"].fillna("").apply(lambda value: len([name for name in value.split("; ") if name]))
    movies["log_budget_usd"] = np.log1p(movies["budget_usd"])
    movies["budget_category"] = pd.qcut(movies["budget_usd"], q=4, labels=["Low", "Mid", "High", "Very high"], duplicates="drop")
    movies["roi_simple"] = (movies["worldwide_revenue_usd"] - movies["budget_usd"]) / movies["budget_usd"]

    # History is calculated using only earlier releases, so the current movie cannot
    # contribute its own outcome to the director's experience feature.
    director_history = movies[movies["director"].notna()].sort_values(["director", "release_date", "tmdb_id"]).copy()
    director_history["director_previous_movie_count"] = director_history.groupby("director").cumcount()
    director_history["director_previous_avg_revenue"] = director_history.groupby("director")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().mean())
    director_history["director_previous_max_revenue"] = director_history.groupby("director")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().max())
    director_history["director_previous_success_rate"] = director_history.groupby("director")["profitable"].transform(lambda series: series.shift().expanding().mean())
    director_history["director_previous_avg_rating"] = director_history.groupby("director")["vote_average"].transform(lambda series: series.shift().expanding().mean())
    movies = movies.drop(columns=["director_previous_movie_count", "director_previous_avg_revenue", "director_previous_success_rate", "director_previous_avg_rating"], errors="ignore").join(director_history[["tmdb_id", "director_previous_movie_count", "director_previous_avg_revenue", "director_previous_success_rate", "director_previous_avg_rating"]].set_index("tmdb_id"), on="tmdb_id")
    movies = movies.join(director_history[["tmdb_id", "director_previous_max_revenue"]].set_index("tmdb_id"), on="tmdb_id")

    # A collection's earlier box-office performance is a useful proxy for brand
    # strength. The current film is shifted out before the expanding statistics.
    franchise_history = movies[movies["collection_id"].notna()].sort_values(["collection_id", "release_date", "tmdb_id"]).copy()
    franchise_history["franchise_previous_movie_count"] = franchise_history.groupby("collection_id").cumcount()
    franchise_history["franchise_previous_avg_revenue"] = franchise_history.groupby("collection_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().mean())
    franchise_history["franchise_previous_last_revenue"] = franchise_history.groupby("collection_id")["worldwide_revenue_usd"].shift()
    franchise_history["franchise_previous_max_revenue"] = franchise_history.groupby("collection_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().max())
    franchise_history["franchise_previous_median_revenue"] = franchise_history.groupby("collection_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().median())
    franchise_history["franchise_previous_success_rate"] = franchise_history.groupby("collection_id")["profitable"].transform(lambda series: series.shift().expanding().mean())
    franchise_history["franchise_previous_latest_success"] = franchise_history.groupby("collection_id")["profitable"].shift()
    franchise_history["franchise_previous_avg_rating"] = franchise_history.groupby("collection_id")["vote_average"].transform(lambda series: series.shift().expanding().mean())
    movies = movies.drop(columns=["franchise_previous_movie_count", "franchise_previous_avg_revenue", "franchise_previous_success_rate", "franchise_previous_avg_rating"], errors="ignore").join(franchise_history[["tmdb_id", "franchise_previous_movie_count", "franchise_previous_avg_revenue", "franchise_previous_success_rate", "franchise_previous_avg_rating"]].set_index("tmdb_id"), on="tmdb_id")
    movies = movies.join(franchise_history[["tmdb_id", "franchise_previous_last_revenue", "franchise_previous_max_revenue", "franchise_previous_median_revenue", "franchise_previous_latest_success"]].set_index("tmdb_id"), on="tmdb_id")
    movies["is_sequel"] = (movies["franchise_previous_movie_count"].fillna(0) > 0).astype(int)

    # Use only earlier films when estimating the commercial track record of the
    # top-billed cast. Current TMDB person popularity is deliberately excluded.
    cast_history = movies[["tmdb_id", "release_date", "cast_ids_top10", "worldwide_revenue_usd", "profitable", "vote_average"]].copy()
    cast_history["actor_id"] = cast_history["cast_ids_top10"].fillna("").str.split("; ")
    cast_history = cast_history.explode("actor_id")
    cast_history = cast_history[cast_history["actor_id"].ne("")].sort_values(["actor_id", "release_date", "tmdb_id"])
    cast_history["actor_previous_movie_count"] = cast_history.groupby("actor_id").cumcount()
    cast_history["actor_previous_avg_revenue"] = cast_history.groupby("actor_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().mean())
    cast_history["actor_previous_max_revenue"] = cast_history.groupby("actor_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().max())
    cast_history["actor_previous_success_rate"] = cast_history.groupby("actor_id")["profitable"].transform(lambda series: series.shift().expanding().mean())
    cast_history["actor_previous_avg_rating"] = cast_history.groupby("actor_id")["vote_average"].transform(lambda series: series.shift().expanding().mean())
    actor_summary = cast_history.groupby("tmdb_id").agg(
        cast_previous_movie_count=("actor_previous_movie_count", "max"),
        cast_previous_avg_revenue=("actor_previous_avg_revenue", "mean"),
        cast_previous_max_revenue=("actor_previous_max_revenue", "max"),
        cast_previous_success_rate=("actor_previous_success_rate", "mean"),
        cast_previous_avg_rating=("actor_previous_avg_rating", "mean"),
    )
    movies = movies.join(actor_summary, on="tmdb_id")

    # A movie can have multiple production companies. Calculate company history
    # per company, then average the available prior histories back to the movie.
    company_history = movies[["tmdb_id", "release_date", "production_company_ids", "worldwide_revenue_usd", "profitable"]].copy()
    company_history["company_id"] = company_history["production_company_ids"].fillna("").str.split("; ")
    company_history = company_history.explode("company_id")
    company_history = company_history[company_history["company_id"].ne("")].sort_values(["company_id", "release_date", "tmdb_id"])
    company_history["company_previous_movie_count"] = company_history.groupby("company_id").cumcount()
    company_history["company_previous_avg_revenue"] = company_history.groupby("company_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().mean())
    company_history["company_previous_max_revenue"] = company_history.groupby("company_id")["worldwide_revenue_usd"].transform(lambda series: series.shift().expanding().max())
    company_history["company_previous_success_rate"] = company_history.groupby("company_id")["profitable"].transform(lambda series: series.shift().expanding().mean())
    company_summary = company_history.groupby("tmdb_id").agg(
        production_company_previous_movie_count=("company_previous_movie_count", "max"),
        production_company_previous_avg_revenue=("company_previous_avg_revenue", "mean"),
        production_company_previous_max_revenue=("company_previous_max_revenue", "max"),
        production_company_previous_success_rate=("company_previous_success_rate", "mean"),
    )
    movies = movies.join(company_summary, on="tmdb_id")
    movies = movies.sort_values("tmdb_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(output_path, index=False)
    print(f"Saved engineered features to {output_path}")


if __name__ == "__main__":
    build_features()
