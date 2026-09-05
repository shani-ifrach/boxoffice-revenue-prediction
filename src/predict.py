"""Create a full feature row for a TMDB movie and score it.

Unlike the earlier demo, this module calculates historical features from movies
released before the requested movie. It therefore uses the same feature contract as
training instead of asking a user to guess director, cast, or company statistics.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.collect_tmdb import tmdb_get
from src.data_cleaning import flatten_movie
from src.train_model import PRE_RELEASE_FEATURES


DEFAULT_HISTORY_PATH = Path("data/processed/movies_clean.csv")
DEFAULT_REVENUE_MODEL_PATH = Path("models/regression_random_forest.joblib")
DEFAULT_BLOCKBUSTER_MODEL_PATH = Path("models/blockbuster_400m/blockbuster_400m_classifier.joblib")
BLOCKBUSTER_THRESHOLD_USD = 400_000_000


def _entity_history(history, entity_column, entity_value):
    if pd.isna(entity_value) or entity_value in (None, ""):
        return 0, np.nan, np.nan, np.nan
    prior = history[history[entity_column] == entity_value]
    if prior.empty:
        return 0, np.nan, np.nan, np.nan
    return len(prior), prior["worldwide_revenue_usd"].mean(), prior["profitable"].mean(), prior["vote_average"].mean()


def _list_entity_history(history, list_column, entity_ids):
    ids = [value for value in str(entity_ids or "").split("; ") if value]
    if not ids:
        return 0, np.nan, np.nan, np.nan
    count_values, revenue_values, success_values, rating_values = [], [], [], []
    for entity_id in ids:
        prior = history[history[list_column].fillna("").str.split("; ").apply(lambda values: entity_id in values)]
        if not prior.empty:
            count_values.append(len(prior)); revenue_values.append(prior["worldwide_revenue_usd"].mean())
            success_values.append(prior["profitable"].mean()); rating_values.append(prior["vote_average"].mean())
    if not count_values:
        return 0, np.nan, np.nan, np.nan
    return max(count_values), np.mean(revenue_values), np.mean(success_values), np.mean(rating_values)

def _extended_entity_history(history, entity_column, entity_value):
    if pd.isna(entity_value) or entity_value in (None, ""):
        return np.nan, np.nan, np.nan, np.nan
    prior = history[history[entity_column] == entity_value].sort_values("release_date")
    if prior.empty:
        return np.nan, np.nan, np.nan, np.nan
    return prior["worldwide_revenue_usd"].iloc[-1], prior["worldwide_revenue_usd"].max(), prior["worldwide_revenue_usd"].median(), prior["profitable"].iloc[-1]

def _list_max_history(history, list_column, entity_ids):
    ids = [value for value in str(entity_ids or "").split("; ") if value]
    prior_movies = [history[history[list_column].fillna("").str.split("; ").apply(lambda values: entity_id in values)] for entity_id in ids]
    if not prior_movies:
        return np.nan
    combined = pd.concat(prior_movies).drop_duplicates("tmdb_id")
    return combined["worldwide_revenue_usd"].max() if not combined.empty else np.nan


def build_prediction_row(movie_record, history_path=DEFAULT_HISTORY_PATH):
    """Build exactly the columns expected by the saved pipelines.

    Only historical rows with an earlier release date are eligible for aggregate
    features. The current movie's revenue is never read or used as an input.
    """
    movie = flatten_movie(movie_record)
    history = pd.read_csv(history_path, parse_dates=["release_date"])
    release_date = pd.to_datetime(movie["release_date"], errors="coerce")
    history = history[history["release_date"] < release_date].copy()
    history = history[history["worldwide_revenue_usd"].notna()].copy()
    history["profitable"] = history["profitable"].astype(float)
    release_month = release_date.month
    month_to_season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    director_count, director_revenue, director_success, director_rating = _entity_history(history, "director", movie["director"])
    franchise_count, franchise_revenue, franchise_success, franchise_rating = _entity_history(history, "collection_id", movie["collection_id"])
    company_count, company_revenue, company_success, company_rating = _list_entity_history(history, "production_company_ids", movie["production_company_ids"])
    cast_count, cast_revenue, cast_success, cast_rating = _list_entity_history(history, "cast_ids_top10", movie["cast_ids_top10"])
    franchise_last, franchise_max, franchise_median, franchise_latest_success = _extended_entity_history(history, "collection_id", movie["collection_id"])
    director_max = _extended_entity_history(history, "director", movie["director"])[1]
    company_max = _list_max_history(history, "production_company_ids", movie["production_company_ids"])
    cast_max = _list_max_history(history, "cast_ids_top10", movie["cast_ids_top10"])
    budget = float(movie["budget_usd"] or 0)
    genres = movie["genres"] or "Unknown"
    row = {
        "budget_usd": budget, "log_budget_usd": np.log1p(budget), "runtime_minutes": movie["runtime_minutes"],
        "release_year": release_date.year, "release_month": release_month, "release_season": month_to_season[release_month],
        "primary_genre": genres.split("; ")[0], "original_language": movie["original_language"], "genre_count": len([x for x in genres.split("; ") if x]),
        "country_count": len([x for x in (movie["production_countries"] or "").split("; ") if x]), "company_count": len([x for x in (movie["production_companies"] or "").split("; ") if x]),
        "cast_size_top10": len([x for x in (movie["cast_top10"] or "").split("; ") if x]), "is_franchise": int(movie["collection_id"] is not None), "is_sequel": int(franchise_count > 0), "is_summer_release": int(release_month in [6, 7, 8]), "is_holiday_release": int(release_month in [11, 12]),
        "franchise_previous_movie_count": franchise_count, "franchise_previous_avg_revenue": franchise_revenue, "franchise_previous_last_revenue": franchise_last, "franchise_previous_max_revenue": franchise_max, "franchise_previous_median_revenue": franchise_median, "franchise_previous_success_rate": franchise_success, "franchise_previous_latest_success": franchise_latest_success, "franchise_previous_avg_rating": franchise_rating,
        "director_previous_movie_count": director_count, "director_previous_avg_revenue": director_revenue, "director_previous_max_revenue": director_max, "director_previous_success_rate": director_success, "director_previous_avg_rating": director_rating,
        "production_company_previous_movie_count": company_count, "production_company_previous_avg_revenue": company_revenue, "production_company_previous_max_revenue": company_max, "production_company_previous_success_rate": company_success,
        "cast_previous_movie_count": cast_count, "cast_previous_avg_revenue": cast_revenue, "cast_previous_max_revenue": cast_max, "cast_previous_success_rate": cast_success, "cast_previous_avg_rating": cast_rating,
    }
    return pd.DataFrame([row], columns=PRE_RELEASE_FEATURES)


def predict_tmdb_movie(
    tmdb_id,
    history_path=DEFAULT_HISTORY_PATH,
    revenue_model_path=DEFAULT_REVENUE_MODEL_PATH,
    blockbuster_model_path=DEFAULT_BLOCKBUSTER_MODEL_PATH,
):
    """Return one revenue estimate and one independent blockbuster probability."""
    import os

    token = os.getenv("TMDB_API_KEY")
    if not token:
        raise SystemExit("Set TMDB_API_KEY before requesting a movie prediction.")
    movie_record = tmdb_get(
        f"movie/{tmdb_id}",
        token,
        {"append_to_response": "credits", "language": "en-US"},
    )
    row = build_prediction_row(movie_record, history_path)
    revenue_model = joblib.load(revenue_model_path)
    blockbuster_model = joblib.load(blockbuster_model_path)
    predicted_revenue = float(max(0, np.expm1(revenue_model.predict(row)[0])))
    blockbuster_probability = float(blockbuster_model.predict_proba(row)[0, 1])
    return {
        "tmdb_id": tmdb_id,
        "title": movie_record.get("title"),
        "predicted_revenue_usd": predicted_revenue,
        "blockbuster_threshold_usd": BLOCKBUSTER_THRESHOLD_USD,
        "blockbuster_probability": blockbuster_probability,
        "blockbuster_prediction": int(blockbuster_probability >= 0.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmdb-id", type=int, required=True)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--revenue-model-path", type=Path, default=DEFAULT_REVENUE_MODEL_PATH)
    parser.add_argument("--blockbuster-model-path", type=Path, default=DEFAULT_BLOCKBUSTER_MODEL_PATH)
    args = parser.parse_args()
    result = predict_tmdb_movie(args.tmdb_id, args.history_path, args.revenue_model_path, args.blockbuster_model_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
