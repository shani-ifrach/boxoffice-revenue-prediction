import importlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_quality import validate_analysis_outputs
from src.train_model import FULL_PRE_RELEASE_FEATURES, _choose_threshold
from src.feature_stability import run_stability


class PipelineOutputTests(unittest.TestCase):
    def test_feature_stability_never_scores_final_holdout(self):
        observed_years = []

        class RecordingModel:
            def fit(self, features, target):
                observed_years.extend(features.release_year.tolist())
                return self

            def predict(self, features):
                observed_years.extend(features.release_year.tolist())
                return np.log1p(np.repeat(100, len(features)))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = pd.DataFrame([{**dict.fromkeys(FULL_PRE_RELEASE_FEATURES, 1),
                                  "release_year": year, "release_date": f"{year}-01-01",
                                  "worldwide_revenue_usd": revenue}
                                 for year in (2014, 2016, 2018, 2020, 2022, 2024)
                                 for revenue in (100, 200)])
            data.to_csv(root / "features.csv", index=False)
            with patch("src.feature_stability.make_model", side_effect=lambda features: RecordingModel()):
                run_stability(root / "features.csv", root / "stability.csv")
            self.assertTrue(observed_years)
            self.assertLessEqual(max(observed_years), 2021)
            results = pd.read_csv(root / "stability.csv")
            self.assertEqual(len(results), 6)
            self.assertEqual(set(results.period), {"validation"})

    def test_pipeline_uses_final_model_and_validates_existing_tables(self):
        pipeline = importlib.import_module("src.run_pipeline")
        names = ("merge_raw_extracts", "clean_movie_data", "build_quality_report",
                 "build_features", "train", "evaluate", "run_eda",
                 "create_results_summary", "validate_analysis_outputs")
        with ExitStack() as stack:
            mocks = {name: stack.enter_context(patch.object(pipeline, name)) for name in names}
            pipeline.run_pipeline()
            mocks["train"].assert_called_once()
            self.assertEqual(len(mocks["train"].call_args.args[0]), 35)
            self.assertEqual(mocks["train"].call_args.kwargs["output_dir"], Path("models/reduced"))
            mocks["validate_analysis_outputs"].assert_called_once_with(
                Path("data/processed/movies_clean.csv"), Path("data/processed/movies_features.csv"),
                Path("models/reduced/test_predictions.csv"), Path("reports/actual_vs_predicted.csv"))

    def test_threshold_can_meet_recall_below_previous_search_range(self):
        actual = np.array([1, 1, 1, 1, 0, 0])
        probabilities = np.array([.01, .02, .03, .04, .001, .002])
        threshold = _choose_threshold(actual, probabilities)
        self.assertGreaterEqual(np.mean(probabilities[actual == 1] >= threshold), .8)
        self.assertLess(threshold, .05)

    def test_threshold_rejects_invalid_inputs(self):
        for actual, probabilities in [([], []), ([0, 0], [.1, .2]),
                                       ([1, 0], [np.nan, .2]), ([1, 0], [1.1, .2])]:
            with self.subTest(actual=actual, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    _choose_threshold(actual, probabilities)

    def test_invalid_or_mismatched_analysis_tables_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            movie = pd.DataFrame({"tmdb_id": [1], "title": ["A Film"],
                                  "release_year": [2022]})
            prediction = movie.assign(worldwide_revenue_usd=120, predicted_revenue_usd=100,
                                      prediction_lower_usd=0, prediction_upper_usd=200,
                                      blockbuster_probability=.1)
            paths = [root / name for name in ("clean.csv", "features.csv", "predictions.csv", "evaluation.csv")]
            for path in paths[:2]:
                movie.to_csv(path, index=False)
            for path in paths[2:]:
                prediction.to_csv(path, index=False)
            self.assertTrue(validate_analysis_outputs(*paths))
            for change, message in [({"prediction_lower_usd": 150}, "Invalid prediction interval"),
                                    ({"blockbuster_probability": 1.2}, "Invalid blockbuster"),
                                    ({"title": "Wrong"}, "title values"),
                                    ({"release_year": 2021}, "final-test"),
                                    ({"predicted_revenue_usd": np.nan}, "finite")]:
                with self.subTest(change=change):
                    prediction.assign(**change).to_csv(paths[2], index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_analysis_outputs(*paths)
            prediction.to_csv(paths[2], index=False)
            prediction.assign(predicted_revenue_usd=101).to_csv(paths[3], index=False)
            with self.assertRaisesRegex(ValueError, "Evaluation does not match"):
                validate_analysis_outputs(*paths)
