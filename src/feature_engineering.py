"""Shared, point-in-time-safe feature construction for training and inference."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MONTH_TO_SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


def _items(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item for item in str(value).split("; ") if item and item != "nan"]


def add_static_features(frame):
    movies = frame.copy()
    for column in ("budget_usd", "runtime_minutes", "worldwide_revenue_usd", "vote_average"):
        if column in movies: movies[column] = pd.to_numeric(movies[column], errors="coerce")
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    if movies["release_date"].isna().any():
        raise ValueError("Every feature row requires a valid release_date.")
    movies["release_year"] = movies["release_date"].dt.year.astype("int64")
    movies["release_month"] = movies["release_date"].dt.month.astype("int64")
    movies["release_season"] = movies["release_month"].map(MONTH_TO_SEASON)
    movies["primary_genre"] = movies["genres"].fillna("Unknown").str.split("; ").str[0].replace("", "Unknown")
    movies["genre_count"] = movies["genres"].apply(lambda value: len(_items(value)))
    movies["country_count"] = movies["production_countries"].apply(lambda value: len(_items(value)))
    movies["company_count"] = movies["production_companies"].apply(lambda value: len(_items(value)))
    movies["cast_size_top10"] = movies["cast_top10"].apply(lambda value: len(_items(value)))
    movies["is_franchise"] = movies["collection_id"].notna().astype(int)
    movies["is_summer_release"] = movies["release_month"].isin([6, 7, 8]).astype(int)
    movies["is_holiday_release"] = movies["release_month"].isin([11, 12]).astype(int)
    movies["log_budget_usd"] = np.log1p(movies["budget_usd"])
    if "worldwide_revenue_usd" in movies:
        valid = movies["budget_usd"].notna() & movies["worldwide_revenue_usd"].notna()
        movies["roi_simple"] = pd.Series(np.nan, index=movies.index, dtype="float64")
        movies.loc[valid, "roi_simple"] = (movies.loc[valid, "worldwide_revenue_usd"] - movies.loc[valid, "budget_usd"]) / movies.loc[valid, "budget_usd"]
        labels = ["Low", "Mid", "High", "Very high"]
        if movies["budget_usd"].nunique(dropna=True) >= 2:
            codes = pd.qcut(movies["budget_usd"], q=4, labels=False, duplicates="drop")
            maximum = int(codes.max()) if codes.notna().any() else -1
            movies["budget_category"] = codes.map({i: labels[i] for i in range(maximum + 1)})
        else:
            movies["budget_category"] = pd.Series(pd.NA, index=movies.index, dtype="object")
    return movies


class _State:
    def __init__(self):
        self.rows = []

    def add(self, revenue, profitable, rating):
        self.rows.append((float(revenue), profitable, rating))

    def summary(self):
        if not self.rows:
            return {
                "count": 0,
                "avg_revenue": np.nan,
                "last_revenue": np.nan,
                "max_revenue": np.nan,
                "median_revenue": np.nan,
                "success_rate": np.nan,
                "latest_success": np.nan,
                "avg_rating": np.nan,
            }
        revenues = np.array([row[0] for row in self.rows], dtype=float)
        profits = np.array([row[1] for row in self.rows], dtype=float)
        ratings = np.array([row[2] for row in self.rows], dtype=float)
        return {
            "count": len(self.rows),
            "avg_revenue": float(np.mean(revenues)),
            "last_revenue": float(self.rows[-1][0]),
            "max_revenue": float(np.max(revenues)),
            "median_revenue": float(np.median(revenues)),
            "success_rate": float(np.nanmean(profits)) if np.isfinite(profits).any() else np.nan,
            "latest_success": self.rows[-1][1],
            "avg_rating": float(np.nanmean(ratings)) if np.isfinite(ratings).any() else np.nan,
        }


def _aggregate_list(states, keys):
    summaries = [states[key].summary() for key in keys if key in states and states[key].rows]
    if not summaries:
        return {"count": 0, "avg_revenue": np.nan, "max_revenue": np.nan, "success_rate": np.nan, "avg_rating": np.nan}
    return {
        "count": max(item["count"] for item in summaries),
        "avg_revenue": float(np.nanmean([item["avg_revenue"] for item in summaries])),
        "max_revenue": float(np.nanmax([item["max_revenue"] for item in summaries])),
        "success_rate": float(np.nanmean([item["success_rate"] for item in summaries]))
        if any(np.isfinite(item["success_rate"]) for item in summaries)
        else np.nan,
        "avg_rating": float(np.nanmean([item["avg_rating"] for item in summaries]))
        if any(np.isfinite(item["avg_rating"]) for item in summaries)
        else np.nan,
    }


def add_historical_features(target, history=None):
    """Use only history with release_date strictly before each target date.

    All same-day targets are scored before that day's outcomes update state, so
    ordering and TMDB IDs cannot leak results between simultaneous releases.
    """
    target = add_static_features(target).reset_index(drop=True)
    source = add_static_features(history if history is not None else target)
    source = source[source["worldwide_revenue_usd"].notna()].copy()
    dates = sorted(set(source["release_date"]) | set(target["release_date"]))
    directors, franchises, companies, cast = defaultdict(_State), defaultdict(_State), defaultdict(_State), defaultdict(_State)
    generated = {}

    def director_key(row):
        value = row.get("director_id")
        return str(value) if pd.notna(value) and str(value) else row.get("director")

    for date in dates:
        for idx, row in target[target["release_date"].eq(date)].iterrows():
            dkey = director_key(row)
            d = directors[dkey].summary() if dkey else _State().summary()
            fkey = str(row.get("collection_id")) if pd.notna(row.get("collection_id")) else None
            f = franchises[fkey].summary() if fkey else _State().summary()
            c = _aggregate_list(companies, _items(row.get("production_company_ids")))
            a = _aggregate_list(cast, _items(row.get("cast_ids_top10")))
            generated[idx] = {
                "director_previous_movie_count": d["count"], "director_previous_avg_revenue": d["avg_revenue"], "director_previous_max_revenue": d["max_revenue"], "director_previous_success_rate": d["success_rate"], "director_previous_avg_rating": d["avg_rating"],
                "franchise_previous_movie_count": f["count"], "franchise_previous_avg_revenue": f["avg_revenue"], "franchise_previous_last_revenue": f["last_revenue"], "franchise_previous_max_revenue": f["max_revenue"], "franchise_previous_median_revenue": f["median_revenue"], "franchise_previous_success_rate": f["success_rate"], "franchise_previous_latest_success": f["latest_success"], "franchise_previous_avg_rating": f["avg_rating"],
                "production_company_previous_movie_count": c["count"], "production_company_previous_avg_revenue": c["avg_revenue"], "production_company_previous_max_revenue": c["max_revenue"], "production_company_previous_success_rate": c["success_rate"],
                "cast_previous_movie_count": a["count"], "cast_previous_avg_revenue": a["avg_revenue"], "cast_previous_max_revenue": a["max_revenue"], "cast_previous_success_rate": a["success_rate"], "cast_previous_avg_rating": a["avg_rating"],
            }
        for _, row in source[source["release_date"].eq(date)].iterrows():
            profit = float(row["profitable"]) if pd.notna(row.get("profitable")) else np.nan
            rating = float(row["vote_average"]) if pd.notna(row.get("vote_average")) else np.nan
            args = (row["worldwide_revenue_usd"], profit, rating)
            dkey = director_key(row)
            if dkey: directors[dkey].add(*args)
            if pd.notna(row.get("collection_id")): franchises[str(row["collection_id"])].add(*args)
            for key in _items(row.get("production_company_ids")): companies[key].add(*args)
            for key in _items(row.get("cast_ids_top10")): cast[key].add(*args)
    historical = pd.DataFrame.from_dict(generated, orient="index")
    for column in [c for c in historical.columns if c.endswith("_movie_count")]:
        historical[column] = historical[column].astype("float64")
    result = target.drop(columns=list(historical.columns), errors="ignore").join(historical)
    result["is_sequel"] = (result["franchise_previous_movie_count"].fillna(0) > 0).astype(int)
    return result


def build_features(input_path=Path("data/processed/movies_clean.csv"), output_path=Path("data/processed/movies_features.csv")):
    """Build and persist the leakage-safe feature table used by training."""
    movies = add_historical_features(pd.read_csv(input_path, parse_dates=["release_date"]))
    movies = movies.sort_values("tmdb_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(output_path, index=False)
    print(f"Saved {len(movies):,} leakage-safe feature rows to {output_path}")


if __name__ == "__main__": build_features()
