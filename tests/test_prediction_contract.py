import unittest
import pandas as pd
from src.predict import validate_artifact
from src.feature_engineering import add_historical_features


class PredictionContractTests(unittest.TestCase):
    def test_unknown_category_is_supported_by_contract(self):
        artifact = {"artifact_version": 2, "kind": "revenue_regression", "features": [f"f{i}" for i in range(35)], "feature_dtypes": {}, "model": object()}
        self.assertIsNone(validate_artifact(artifact, "revenue_regression"))

    def test_incompatible_artifact_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_artifact({}, "revenue_regression")

    def test_training_inference_builder_parity(self):
        common = {"director": "A", "collection_id": None, "production_company_ids": "p1", "cast_ids_top10": "a", "runtime_minutes": 90, "budget_usd": 10, "original_language": "en", "genres": "Drama", "production_countries": "US", "production_companies": "P", "cast_top10": "Actor", "vote_average": 7}
        history = pd.DataFrame([{**common, "tmdb_id": 1, "release_date": "2019-01-01", "worldwide_revenue_usd": 100, "profitable": 1}])
        target = pd.DataFrame([{**common, "tmdb_id": 2, "release_date": "2020-01-01", "worldwide_revenue_usd": None, "profitable": pd.NA}])
        first = add_historical_features(target, history)
        second = add_historical_features(target.copy(), history.copy())
        pd.testing.assert_frame_equal(first, second)
