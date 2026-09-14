"""Merge raw TMDB extracts from different sampling strategies by TMDB ID.

The merge is intentionally performed before flattening so the original TMDB
responses remain available for auditing and for future feature changes.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def merge_raw_extracts(raw_dir=Path("data/raw"), output_path=Path("data/raw/tmdb_movies_merged.json")):
    """Combine extract files and keep one record per TMDB movie ID.

    The same movie may be returned by multiple discover queries. TMDB's numeric
    ID is used instead of the title because titles are not unique and may vary
    by language or release. The output is a derived raw extract and is not
    flattened or analytically filtered at this stage.
    """
    records_by_id = {}
    occurrences = {}
    source_rows = []
    for path in sorted(raw_dir.glob("tmdb_movies_*.json")):
        if path.name == output_path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dates = pd.to_datetime([record.get("release_date") for record in payload], errors="coerce")
        timestamp = path.stem.removeprefix("tmdb_movies_")
        metadata_path = raw_dir / f"collection_metadata_{timestamp}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        source_rows.append({"source_file": path.name, "row_count": len(payload),
                            "collected_at_utc": metadata.get("collected_at_utc", timestamp),
                            "start_year_observed": int(dates.min().year) if dates.notna().any() else None,
                            "end_year_observed": int(dates.max().year) if dates.notna().any() else None,
                            "collection_strategy": metadata.get("sort_by", "unknown_legacy_extract"),
                            "schema_version": metadata.get("schema_version", "legacy")})
        for record in payload:
            records_by_id[record["id"]] = record
            occurrences.setdefault(record["id"], []).append(path.name)
    if not records_by_id:
        raise ValueError(f"No source extracts found in {raw_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(records_by_id.values()), ensure_ascii=False), encoding="utf-8")
    duplicate_rows = [
        {"tmdb_id": movie_id, "source_count": len(files), "source_files": "; ".join(files)}
        for movie_id, files in occurrences.items() if len(files) > 1
    ]
    pd.DataFrame(duplicate_rows, columns=["tmdb_id", "source_count", "source_files"]).to_csv(raw_dir / "duplicate_report.csv", index=False)
    coverage = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": source_rows,
        "source_record_count": sum(row["row_count"] for row in source_rows),
        "unique_movie_count": len(records_by_id),
        "duplicate_movie_count": len(duplicate_rows),
    }
    (raw_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"Merged {len(records_by_id):,} unique movies into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/raw/tmdb_movies_merged.json"))
    args = parser.parse_args()
    merge_raw_extracts(args.raw_dir, args.output_path)
