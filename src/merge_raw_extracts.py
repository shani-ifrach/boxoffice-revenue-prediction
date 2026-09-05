"""Merge raw TMDB extracts from different sampling strategies by TMDB ID."""
import json
import argparse
from pathlib import Path


def merge_raw_extracts(raw_dir=Path("data/raw"), output_path=Path("data/raw/tmdb_movies_merged.json")):
    records_by_id = {}
    for path in sorted(raw_dir.glob("tmdb_movies_*.json")):
        if path.name == output_path.name:
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            records_by_id[record["id"]] = record
    output_path.write_text(json.dumps(list(records_by_id.values()), ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(records_by_id):,} unique movies into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/raw/tmdb_movies_merged.json"))
    args = parser.parse_args()
    merge_raw_extracts(args.raw_dir, args.output_path)
