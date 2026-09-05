"""Create a full feature row for a TMDB movie and score it.

Unlike the earlier demo, this module calculates historical features from movies
released before the requested movie. It therefore uses the same feature contract as
training instead of asking a user to guess director, cast, or company statistics.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.collect_tmdb import tmdb_get
from src.data_cleaning import flatten_movie
from src.train_model import PRE_RELEASE_FEATURES


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
            count_values.append(len(prior))
            revenue_values.append(prior["worldwide_revenue_usd"].mean())
            success_values.append(prior["profitable"].mean())
            rating_values.append(prior["vote_average"].mean())
    if not count_values:
        return 0, np.nan, np.nan, np.nan
    return max(count_values), np.mean(revenue_values), np.mean(success_values), np.mean(rating_values)


def build_prediction_row(movie_record, history_path=Path("data/processed/movies_clean.csv")):
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
    budget = float(movie["budget_usd"] or 0)
    genres = movie["genres"] or "Unknown"
    row = {
        "budget_usd": budget, "log_budget_usd": np.log1p(budget), "runtime_minutes": movie["runtime_minutes"],
        "release_year": release_date.year, "release_month": release_month, "release_season": month_to_season[release_month],
        "primary_genre": genres.split("; ")[0], "original_language": movie["original_language"], "genre_count": len([x for x in genres.split("; ") if x]),
        "country_count": len([x for x in (movie["production_countries"] or "").split("; ") if x]), "company_count": len([x for x in (movie["production_companies"] or "").split("; ") if x]),
        "cast_size_top10": len([x for x in (movie["cast_top10"] or "").split("; ") if x]), "is_franchise": int(movie["collection_id"] is not None),
        "franchise_previous_movie_count": franchise_count, "franchise_previous_avg_revenue": franchise_revenue, "franchise_previous_success_rate": franchise_success, "franchise_previous_avg_rating": franchise_rating,
        "director_previous_movie_count": director_count, "director_previous_avg_revenue": director_revenue, "director_previous_success_rate": director_success, "director_previous_avg_rating": director_rating,
        "production_company_previous_movie_count": company_count, "production_company_previous_avg_revenue": company_revenue, "production_company_previous_success_rate": company_success,
        "cast_previous_movie_count": cast_count, "cast_previous_avg_revenue": cast_revenue, "cast_previous_success_rate": cast_success, "cast_previous_avg_rating": cast_rating,
    }
    return pd.DataFrame([row], columns=PRE_RELEASE_FEATURES)


def predict_tmdb_movie(tmdb_id):
    """Fetch a movie by TMDB ID and return revenue plus profitability probability."""
    import os
    token = os.getenv("TMDB_API_KEY")
    if not token:
        raise SystemExit("Set TMDB_API_KEY before requesting a movie prediction.")
    movie_record = tmdb_get(f"movie/{tmdb_id}", token, {"append_to_response": "credits", "language": "en-US"})
    row = build_prediction_row(movie_record)
    revenue_model = joblib.load("models/regression_random_forest.joblib")
    profit_model = joblib.load("models/classification_gradient_boosting.joblib")
    predicted_revenue = float(max(0, np.expm1(revenue_model.predict(row)[0])))
    profitability_probability = float(profit_model.predict_proba(row)[0, 1])
    return predicted_revenue, profitability_probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmdb-id", type=int, required=True)
    args = parser.parse_args()
    print(predict_tmdb_movie(args.tmdb_id))
