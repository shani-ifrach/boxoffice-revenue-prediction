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

from src.merge_raw_extracts import merge_raw_extracts

BASE_URL = "https://api.themoviedb.org/3"
COLLECTION_STRATEGIES = (
    "popularity.desc",
    "popularity.asc",
    "vote_count.desc",
)
SORT_CHOICES = COLLECTION_STRATEGIES + ("vote_count.asc",)


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


def collect_movies(
    start_year, end_year, pages_per_year, output_dir, sort_by="popularity.desc"
):
    """Collect movie details for a year range and save the raw extract.

    The discover call is intentionally limited to a configurable number of pages
    per year. TMDB returns a popularity- or vote-ordered slice, not a random sample,
    so the chosen ``sort_by`` value is recorded in metadata and must be treated as
    a sampling limitation during analysis. A set of TMDB IDs removes overlap between
    pages within this strategy; the later merge removes overlap between strategies.
    The final detail call is made only once per unique ID in this extract.

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
                "discover/movie",
                token,
                {
                    "primary_release_year": year,
                    "sort_by": sort_by,
                    "page": page,
                    "include_adult": "false",
                    "include_video": "false",
                    "language": "en-US",
                },
            )
            movie_ids.update(item["id"] for item in payload.get("results", []))
            time.sleep(0.1)

    records = []
    for index, movie_id in enumerate(sorted(movie_ids), start=1):
        records.append(
            tmdb_get(
                f"movie/{movie_id}",
                token,
                {"append_to_response": "credits", "language": "en-US"},
            )
        )
        time.sleep(0.1)
        if index % 100 == 0:
            print(f"Collected {index}/{len(movie_ids)} movies")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    strategy_slug = sort_by.replace(".", "_")
    run_id = f"{timestamp}_{strategy_slug}"
    raw_path = output_dir / f"tmdb_movies_{run_id}.json"
    raw_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "source": "TMDB API",
        "collected_at_utc": timestamp,
        "start_year": start_year,
        "end_year": end_year,
        "pages_per_year": pages_per_year,
        "sort_by": sort_by,
        "movie_count": len(records),
        "page_count": (end_year - start_year + 1) * pages_per_year,
        "schema_version": "2.0",
        "parameters": {
            "include_adult": False,
            "include_video": False,
            "language": "en-US",
        },
        "detail_endpoint": f"{BASE_URL}/movie/{{movie_id}}?append_to_response=credits",
    }
    (output_dir / f"collection_metadata_{run_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    # Compatibility index: append a compact entry instead of overwriting provenance.
    index_path = output_dir / "collection_metadata.json"
    try:
        existing = (
            json.loads(index_path.read_text(encoding="utf-8"))
            if index_path.exists()
            else []
        )
    except json.JSONDecodeError:
        existing = []
    if isinstance(existing, dict):
        existing = [existing]
    existing.append(metadata)
    index_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} raw movie records to {raw_path}")
    return raw_path


def collect_strategy_mix(
    start_year, end_year, pages_per_year, output_dir, strategies=COLLECTION_STRATEGIES
):
    """Collect the documented sampling mix and merge it by canonical TMDB ID.

    The project originally combined popularity-descending, popularity-ascending,
    and vote-count-descending discovery slices. Keeping each strategy in its own
    timestamped raw extract preserves provenance, while the merged file removes
    overlap before cleaning. Callers may still pass one explicit strategy for a
    targeted or diagnostic collection.
    """
    strategies = tuple(strategies)
    if not strategies:
        raise ValueError("At least one collection strategy is required.")
    invalid = sorted(set(strategies) - set(SORT_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported collection strategies: {invalid}")

    raw_paths = [
        collect_movies(start_year, end_year, pages_per_year, output_dir, strategy)
        for strategy in strategies
    ]
    merged_path = output_dir / "tmdb_movies_merged.json"
    merge_raw_extracts(output_dir, merged_path, source_paths=raw_paths)
    return raw_paths, merged_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Collect TMDB movie extracts. By default, reproduce the project's "
            "three-strategy discovery mix and merge overlaps by tmdb_id."
        )
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--pages-per-year", type=int, default=15)
    parser.add_argument(
        "--sort-by",
        action="append",
        choices=SORT_CHOICES,
        help=(
            "Discovery ordering. Repeat to combine strategies. If omitted, uses "
            "popularity.desc, popularity.asc, and vote_count.desc."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    strategies = args.sort_by or COLLECTION_STRATEGIES
    collect_strategy_mix(
        args.start_year,
        args.end_year,
        args.pages_per_year,
        args.output_dir,
        strategies,
    )
