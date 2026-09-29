import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from src.collect_tmdb import COLLECTION_STRATEGIES, build_parser, collect_strategy_mix
from src.data_cleaning import clean_movie_data
from src.merge_raw_extracts import merge_raw_extracts


def record(movie_id, budget, revenue, runtime=90, date="2020-01-01"):
    return {
        "id": movie_id,
        "title": f"M{movie_id}",
        "release_date": date,
        "runtime": runtime,
        "budget": budget,
        "revenue": revenue,
        "original_language": "en",
        "genres": [],
        "production_countries": [],
        "production_companies": [],
        "credits": {"cast": [], "crew": []},
    }


class DataQualityTests(unittest.TestCase):
    def test_collection_defaults_to_documented_strategy_mix(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.sort_by)
        self.assertEqual(args.pages_per_year, 15)
        self.assertEqual(
            COLLECTION_STRATEGIES,
            ("popularity.desc", "popularity.asc", "vote_count.desc"),
        )

    def test_collection_accepts_repeated_explicit_strategies(self):
        args = build_parser().parse_args(
            [
                "--sort-by",
                "popularity.asc",
                "--sort-by",
                "vote_count.desc",
            ]
        )
        self.assertEqual(args.sort_by, ["popularity.asc", "vote_count.desc"])

    def test_strategy_mix_collects_each_slice_and_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            strategies = ("popularity.desc", "popularity.asc", "vote_count.desc")
            fake_paths = [
                output_dir / f"tmdb_movies_{index}.json" for index in range(3)
            ]
            with (
                patch(
                    "src.collect_tmdb.collect_movies", side_effect=fake_paths
                ) as collect,
                patch("src.collect_tmdb.merge_raw_extracts") as merge,
            ):
                paths, merged = collect_strategy_mix(
                    2010, 2024, 15, output_dir, strategies
                )
            self.assertEqual(paths, fake_paths)
            self.assertEqual(collect.call_count, 3)
            self.assertEqual(
                [call.args[-1] for call in collect.call_args_list],
                list(strategies),
            )
            self.assertEqual(merged, output_dir / "tmdb_movies_merged.json")
            merge.assert_called_once_with(output_dir, merged, source_paths=fake_paths)

    def test_profitability_missingness_and_zero_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            out = Path(tmp) / "clean.csv"
            (raw / "tmdb_movies_a.json").write_text(
                json.dumps(
                    [
                        record(1, 50, 100),
                        record(2, None, 100),
                        record(3, 0, 100),
                        record(4, 100, None),
                        record(5, 50, 0),
                    ]
                )
            )
            clean_movie_data(raw, out)
            frame = pd.read_csv(out)
            self.assertEqual(frame.set_index("tmdb_id").loc[1, "profitable"], 1)
            self.assertTrue(pd.isna(frame.set_index("tmdb_id").loc[2, "profitable"]))
            self.assertTrue(pd.isna(frame.set_index("tmdb_id").loc[3, "profitable"]))
            self.assertNotIn(4, frame.tmdb_id.tolist())
            self.assertNotIn(5, frame.tmdb_id.tolist())

    def test_merge_deduplicates_and_reports_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            out = raw / "tmdb_movies_merged.json"
            (raw / "tmdb_movies_a.json").write_text(json.dumps([record(1, 1, 2)]))
            (raw / "tmdb_movies_b.json").write_text(
                json.dumps([record(1, 2, 3), record(2, 1, 4)])
            )
            merge_raw_extracts(raw, out)
            self.assertEqual(len(json.loads(out.read_text())), 2)
            self.assertEqual(len(pd.read_csv(raw / "duplicate_report.csv")), 1)

    def test_merge_can_limit_sources_to_current_collection_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            out = raw / "tmdb_movies_merged.json"
            current = raw / "tmdb_movies_current.json"
            old = raw / "tmdb_movies_old.json"
            current.write_text(json.dumps([record(1, 1, 2)]))
            old.write_text(json.dumps([record(2, 1, 4)]))
            merge_raw_extracts(raw, out, source_paths=[current])
            merged_ids = [item["id"] for item in json.loads(out.read_text())]
            self.assertEqual(merged_ids, [1])
