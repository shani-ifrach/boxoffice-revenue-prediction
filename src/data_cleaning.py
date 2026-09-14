"""Normalize and validate the raw TMDB movie extract.

This module creates one analytical row per TMDB movie while retaining identifiers
needed for historical features. It does not silently invent financial values:
TMDB zeros are converted to missing values because, in this source, zero commonly
means that budget or revenue was not reported. The raw JSON remains the source of
truth; this CSV is an explicit, reproducible modeling layer.
"""
import json
from pathlib import Path
import pandas as pd

from src.merge_raw_extracts import merge_raw_extracts


def flatten_movie(record):
    """Convert one nested TMDB response into a flat, one-row movie record.

    TMDB stores genres, companies, cast, crew, and collections as nested lists or
    objects. We keep human-readable names for reporting and stable TMDB IDs for
    joins and historical calculations. Only the first ten cast entries are retained
    because they approximate the billed lead cast while keeping the table compact;
    the IDs are retained separately so actor history can be calculated without
    fuzzy-matching names.

    The function uses ``get`` and empty fallbacks because missing credits or
    collection membership are valid source states, not reasons to discard the
    entire movie. Missing values are handled later at the dataset level, where the
    removal or imputation decision can be measured and documented.
    """
    credits = record.get("credits", {})
    top_cast = credits.get("cast", [])[:10]
    cast = [person.get("name") for person in top_cast if person.get("name")]
    cast_ids = [str(person.get("id")) for person in top_cast if person.get("id")]
    director_people = [person for person in credits.get("crew", []) if person.get("job") == "Director"]
    directors = [person.get("name") for person in director_people if person.get("name")]
    director_ids = [str(person.get("id")) for person in director_people if person.get("id")]
    collection = record.get("belongs_to_collection") or {}
    companies = record.get("production_companies", [])
    return {
        "tmdb_id": record.get("id"), "title": record.get("title"),
        "release_date": record.get("release_date"), "runtime_minutes": record.get("runtime"),
        "original_language": record.get("original_language"), "budget_usd": record.get("budget"),
        "worldwide_revenue_usd": record.get("revenue"), "popularity": record.get("popularity"),
        "vote_average": record.get("vote_average"), "vote_count": record.get("vote_count"),
        "genres": "; ".join(g.get("name", "") for g in record.get("genres", [])),
        "production_countries": "; ".join(c.get("iso_3166_1", "") for c in record.get("production_countries", [])),
        "production_companies": "; ".join(c.get("name", "") for c in companies),
        "production_company_ids": "; ".join(str(c.get("id")) for c in companies if c.get("id")),
        "collection_id": collection.get("id"), "collection_name": collection.get("name"),
        "director": directors[0] if directors else None,
        "director_id": director_ids[0] if director_ids else None,
        "director_ids": "; ".join(director_ids),
        "cast_top10": "; ".join(cast),
        "cast_ids_top10": "; ".join(cast_ids),
    }


def clean_movie_data(raw_dir=Path("data/raw"), output_path=Path("data/processed/movies_clean.csv")):
    """Build the cleaned movie table from the most recent raw JSON extract.

    The latest extract is selected so a newly merged or newly collected file becomes
    the input to the next pipeline step without hard-coding a timestamp. Records are
    de-duplicated by ``tmdb_id`` rather than title: titles are not unique, while the
    TMDB ID is the source's canonical movie identifier.

    Dates and numeric fields are coerced to nullable values so malformed source
    values can be counted instead of crashing the run. Rows without a valid release
    date or a reported worldwide revenue are excluded because release timing and the
    regression target cannot be defined for them. Rows without a reported budget are
    retained for descriptive analysis, but their ``profitable`` label remains
    missing and they are excluded later from budget-dependent modeling.
    """
    merged_path = raw_dir / "tmdb_movies_merged.json"
    raw_files = sorted(raw_dir.glob("tmdb_movies_*.json"))
    if not raw_files:
        raise SystemExit("No raw TMDB extract found. Run src.collect_tmdb first.")
    extract_files = [path for path in raw_files if path.name != merged_path.name]
    if extract_files and (not merged_path.exists() or max(path.stat().st_mtime for path in extract_files) > merged_path.stat().st_mtime):
        merge_raw_extracts(raw_dir, merged_path)
    source_path = merged_path if merged_path.exists() else raw_files[-1]
    records = json.loads(source_path.read_text(encoding="utf-8"))
    movies = pd.DataFrame([flatten_movie(record) for record in records])
    movies = movies.dropna(subset=["tmdb_id"]).drop_duplicates("tmdb_id", keep="last").copy()
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    numeric_columns = ["runtime_minutes", "budget_usd", "worldwide_revenue_usd", "popularity", "vote_average", "vote_count"]
    movies[numeric_columns] = movies[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Zero budgets/revenues mean "not reported" in TMDB more often than a genuine zero.
    # They are kept as missing so the modeling decision is explicit and auditable.
    movies.loc[movies["budget_usd"] <= 0, "budget_usd"] = pd.NA
    movies.loc[movies["worldwide_revenue_usd"] <= 0, "worldwide_revenue_usd"] = pd.NA
    movies.loc[movies["runtime_minutes"] <= 0, "runtime_minutes"] = pd.NA
    movies = movies[movies["release_date"].notna() & movies["worldwide_revenue_usd"].notna()].copy()
    movies["release_year"] = movies["release_date"].dt.year.astype("int64")
    valid_profitability = movies["budget_usd"].notna() & movies["worldwide_revenue_usd"].notna()
    movies["profitable"] = pd.Series(pd.NA, index=movies.index, dtype="Int64")
    movies.loc[valid_profitability, "profitable"] = (
        movies.loc[valid_profitability, "worldwide_revenue_usd"]
        > movies.loc[valid_profitability, "budget_usd"]
    ).astype("int64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(output_path, index=False)
    print(f"Saved {len(movies):,} cleaned movies to {output_path}")


if __name__ == "__main__":
    clean_movie_data()
