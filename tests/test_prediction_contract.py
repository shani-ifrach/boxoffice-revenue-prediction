import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression

from src.data_cleaning import flatten_movie
from src.feature_engineering import add_historical_features
from src.predict import build_prediction_row, predict_record, validate_artifact
from src.train_model import FULL_PRE_RELEASE_FEATURES, PRE_RELEASE_FEATURES, _artifact, make_pipeline


def movie(movie_id, date, budget=50, language="en", genre="Drama", revenue=100):
    return {"id": movie_id, "title": f"Movie {movie_id}", "release_date": date,
            "budget": budget, "revenue": revenue, "runtime": 90,
            "original_language": language, "genres": [{"name": genre}],
            "production_countries": [{"iso_3166_1": "US"}],
            "production_companies": [{"id": 1, "name": "Studio"}],
            "credits": {"crew": [{"id": 10, "name": "Director", "job": "Director"}],
                        "cast": [{"id": 20, "name": "Actor"}]}, "vote_average": 7}


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.history_path = Path(self.temporary.name) / "history.csv"
        self.history = pd.DataFrame([
            {**flatten_movie(movie(1, "2018-01-01")), "profitable": 1},
            {**flatten_movie(movie(2, "2019-01-01", budget=60)), "profitable": 1},
        ])
        self.history.to_csv(self.history_path, index=False)
        rows = add_historical_features(self.history)
        self.model = make_pipeline(DummyRegressor()).fit(rows[PRE_RELEASE_FEATURES], np.log1p([100, 200]))
        self.artifact = _artifact("revenue_regression", self.model, PRE_RELEASE_FEATURES,
            target_transform="log1p", prediction_cap_usd=1000,
            conformal_absolute_error_usd=10, conformal_coverage=0.9)

    def test_unknown_category_and_missing_budget_are_scored(self):
        record = movie(99, "2020-01-01", budget=0, language="new_language", genre="New Genre")
        row = build_prediction_row(record, self.history_path, PRE_RELEASE_FEATURES)
        self.assertTrue(pd.isna(row.budget_usd.iloc[0]))
        self.assertEqual(row.original_language.iloc[0], "new_language")
        validate_artifact(self.artifact, "revenue_regression")
        self.assertTrue(np.isfinite(self.model.predict(row)).all())

    def test_training_inference_builder_parity(self):
        for features in (PRE_RELEASE_FEATURES, FULL_PRE_RELEASE_FEATURES):
            for budget in (50, None, 0):
                with self.subTest(features=len(features), budget=budget):
                    record = movie(99, "2020-01-01", budget=budget)
                    flattened = flatten_movie(record)
                    flattened["budget_usd"] = np.nan if not budget else float(budget)
                    flattened["worldwide_revenue_usd"] = np.nan
                    flattened["profitable"] = pd.NA
                    training_row = add_historical_features(pd.DataFrame([flattened]), self.history)[features]
                    inference_row = build_prediction_row(record, self.history_path, features)
                    pd.testing.assert_frame_equal(training_row, inference_row)

    def test_valid_artifact_is_accepted(self):
        self.assertIsNone(validate_artifact(self.artifact, "revenue_regression"))

    def test_classifier_contract_and_complete_prediction(self):
        rows = add_historical_features(self.history)[PRE_RELEASE_FEATURES]
        model = make_pipeline(DummyClassifier(strategy="prior")).fit(rows, [0, 1])
        calibrator = LogisticRegression().fit(np.array([[.1], [.9]]), [0, 1])
        artifact = _artifact("blockbuster_classification", model, PRE_RELEASE_FEATURES,
                             calibrator=calibrator, decision_threshold=.5,
                             blockbuster_threshold_usd=400_000_000)
        validate_artifact(artifact, "blockbuster_classification")
        for change, message in [({"decision_threshold": -1}, "range"),
                                ({"calibrator": object()}, "calibrator"),
                                ({"blockbuster_threshold_usd": 250_000_000}, "400M")]:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, message):
                    validate_artifact({**artifact, **change}, "blockbuster_classification")
        directory = Path(self.temporary.name)
        joblib.dump(self.artifact, directory / "revenue_regression.joblib")
        joblib.dump(artifact, directory / "blockbuster_classifier.joblib")
        result = predict_record(movie(99, "2020-01-01", budget=None, language="unseen"),
                                self.history_path, directory)
        self.assertLessEqual(result["prediction_lower_usd"], result["predicted_revenue_usd"])
        self.assertLessEqual(result["predicted_revenue_usd"], result["prediction_upper_usd"])
        self.assertTrue(0 <= result["blockbuster_probability"] <= 1)

    def test_incompatible_artifact_fails_clearly(self):
        changes = [({"artifact_version": 999}, "artifact_version"),
                   ({"schema_version": "old"}, "schema_version"),
                   ({"features": list(reversed(PRE_RELEASE_FEATURES))}, "ordered feature"),
                   ({"feature_dtypes": {}}, "dtypes"),
                   ({"prediction_cap_usd": float("nan")}, "finite"),
                   ({"conformal_absolute_error_usd": -1}, "range"),
                   ({"model": object()}, "implement predict")]
        for change, message in changes:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, message):
                    validate_artifact({**self.artifact, **change}, "revenue_regression")
        incomplete = dict(self.artifact)
        del incomplete["prediction_cap_usd"]
        with self.assertRaisesRegex(ValueError, "missing.*prediction_cap_usd"):
            validate_artifact(incomplete, "revenue_regression")
