"""Collect a reproducible, auditable movie extract from TMDB.

The collector deliberately separates discovery from detail retrieval. The discover
endpoint is used to build a de-duplicated list of movie IDs, and the detail endpoint
is then called once per ID with credits appended. This gives us a stable TMDB key for
joins and preserves the nested source response before any analytical assumptions are
introduced.

The API key is read from the environment rather than committed to source control.
Raw responses are retained because a cleaned table is a derived artifact: if a
cleaning rule changes, we can rebuild it without re-querying the API or losing the
original source representation.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.themoviedb.org/3"


def tmdb_get(endpoint, token, params=None):
    """Request one TMDB endpoint and fail loudly on an unsuccessful response.

    ``params`` is copied into a new dictionary before the API key is added. This
    avoids mutating a caller-owned dictionary and keeps authentication out of the
    endpoint string. A 30-second timeout prevents one stalled request from leaving
    a long collection run hanging indefinitely; ``raise_for_status`` prevents an
    error payload from being mistaken for a valid movie record.
    """
    # TMDB has two credential formats. A v3 API key is passed as ``api_key``;
    # a v4 read-access token is a JWT and must be sent as a Bearer token.
    # Sending a v4 token as ``api_key`` causes a 401 Unauthorized response.
    request_headers = {"accept": "application/json"}
    request_params = dict(params or {})
    if token.startswith("eyJ"):
        request_headers["Authorization"] = f"Bearer {token}"
    else:
        request_params["api_key"] = token

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers=request_headers,
        params=request_params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def collect_movies(start_year, end_year, pages_per_year, output_dir, sort_by="popularity.desc"):
    """Collect movie details for a year range and save the raw extract.

    The discover call is intentionally limited to a configurable number of pages
    per year. TMDB returns a popularity- or vote-ordered slice, not a random sample,
    so the chosen ``sort_by`` value is recorded in metadata and must be treated as
    a sampling limitation during analysis. A set of TMDB IDs removes overlap between
    pages or between separate collection strategies; the final detail call is made
    only once per unique ID.

    The function stores the source payload as JSON rather than flattening it here.
    Flattening is a separate cleaning responsibility, which keeps collection
    reproducible and lets us preserve fields that may become useful later, such as
    collection and credit IDs.
    """
    token = os.getenv("TMDB_API_KEY")
    if not token:
        raise SystemExit("Set TMDB_API_KEY before collecting data.")

    output_dir.mkdir(parents=True, exist_ok=True)
    movie_ids = set()
    for year in range(start_year, end_year + 1):
        for page in range(1, pages_per_year + 1):
            payload = tmdb_get(
                "discover/movie", token,
                {"primary_release_year": year, "sort_by": sort_by, "page": page,
                 "include_adult": "false", "include_video": "false", "language": "en-US"},
            )
            movie_ids.update(item["id"] for item in payload.get("results", []))
            time.sleep(0.1)

    records = []
    for index, movie_id in enumerate(sorted(movie_ids), start=1):
        records.append(tmdb_get(f"movie/{movie_id}", token, {"append_to_response": "credits", "language": "en-US"}))
        time.sleep(0.1)
        if index % 100 == 0:
            print(f"Collected {index}/{len(movie_ids)} movies")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = output_dir / f"tmdb_movies_{timestamp}.json"
    raw_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (output_dir / "collection_metadata.json").write_text(json.dumps({
        "source": "TMDB API",
        "collected_at_utc": timestamp,
        "start_year": start_year,
        "end_year": end_year,
        "pages_per_year": pages_per_year,
        "sort_by": sort_by,
        "movie_count": len(records),
        "detail_endpoint": f"{BASE_URL}/movie/{{movie_id}}?append_to_response=credits",
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} raw movie records to {raw_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--pages-per-year", type=int, default=5)
    parser.add_argument("--sort-by", default="popularity.desc", choices=["popularity.desc", "popularity.asc", "vote_count.desc", "vote_count.asc"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    collect_movies(args.start_year, args.end_year, args.pages_per_year, args.output_dir, args.sort_by)
