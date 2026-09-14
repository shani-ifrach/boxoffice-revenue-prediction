import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.data_cleaning import clean_movie_data
from src.merge_raw_extracts import merge_raw_extracts


def record(movie_id, budget, revenue, runtime=90, date="2020-01-01"):
    return {"id": movie_id, "title": f"M{movie_id}", "release_date": date, "runtime": runtime,
            "budget": budget, "revenue": revenue, "original_language": "en", "genres": [],
            "production_countries": [], "production_companies": [], "credits": {"cast": [], "crew": []}}


class DataQualityTests(unittest.TestCase):
    def test_profitability_missingness_and_zero_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"; raw.mkdir(); out = Path(tmp) / "clean.csv"
            (raw / "tmdb_movies_a.json").write_text(json.dumps([record(1, 50, 100), record(2, None, 100), record(3, 0, 100), record(4, 100, None), record(5, 50, 0)]))
            clean_movie_data(raw, out); frame = pd.read_csv(out)
            self.assertEqual(frame.set_index("tmdb_id").loc[1, "profitable"], 1)
            self.assertTrue(pd.isna(frame.set_index("tmdb_id").loc[2, "profitable"]))
            self.assertTrue(pd.isna(frame.set_index("tmdb_id").loc[3, "profitable"]))
            self.assertNotIn(4, frame.tmdb_id.tolist()); self.assertNotIn(5, frame.tmdb_id.tolist())

    def test_merge_deduplicates_and_reports_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp); out = raw / "tmdb_movies_merged.json"
            (raw / "tmdb_movies_a.json").write_text(json.dumps([record(1, 1, 2)]))
            (raw / "tmdb_movies_b.json").write_text(json.dumps([record(1, 2, 3), record(2, 1, 4)]))
            merge_raw_extracts(raw, out)
            self.assertEqual(len(json.loads(out.read_text())), 2)
            self.assertEqual(len(pd.read_csv(raw / "duplicate_report.csv")), 1)
