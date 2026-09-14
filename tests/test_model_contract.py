import unittest

import pandas as pd

from src.feature_engineering import add_historical_features
from src.train_model import FULL_PRE_RELEASE_FEATURES, PRE_RELEASE_FEATURES


class ModelContractTests(unittest.TestCase):
    def test_production_contract_has_35_features(self):
        self.assertEqual(len(FULL_PRE_RELEASE_FEATURES), 38)
        self.assertEqual(len(PRE_RELEASE_FEATURES), 35)
        self.assertEqual(
            set(FULL_PRE_RELEASE_FEATURES) - set(PRE_RELEASE_FEATURES),
            {"company_count", "log_budget_usd", "production_company_previous_max_revenue"},
        )

    def test_prior_history_replaces_existing_list_entity_features(self):
        history = pd.DataFrame([
            {"tmdb_id": 1, "release_date": "2019-01-01", "director": "A", "collection_id": 10, "production_company_ids": "p1; p2", "cast_ids_top10": "a; b", "worldwide_revenue_usd": 100.0, "profitable": 1.0, "vote_average": 7.0},
            {"tmdb_id": 2, "release_date": "2019-06-01", "director": "A", "collection_id": 10, "production_company_ids": "p1", "cast_ids_top10": "a", "worldwide_revenue_usd": 200.0, "profitable": 1.0, "vote_average": 8.0},
            {"tmdb_id": 3, "release_date": "2018-03-01", "director": "B", "collection_id": 20, "production_company_ids": "p2", "cast_ids_top10": "c", "worldwide_revenue_usd": 50.0, "profitable": 0.0, "vote_average": 6.0},
        ])
        target = pd.DataFrame([{
            "tmdb_id": 99, "release_date": "2020-01-01", "director": "A", "collection_id": 10,
            "production_company_ids": "p1; p2", "cast_ids_top10": "a; b",
            "director_previous_movie_count": 999, "production_company_previous_movie_count": 999,
            "cast_previous_movie_count": 999,
        }])
        for frame in (history, target):
            frame["runtime_minutes"] = 100
            frame["budget_usd"] = 50
            frame["original_language"] = "en"
            frame["genres"] = "Drama"
            frame["production_countries"] = "US"
            frame["production_companies"] = "P"
            frame["cast_top10"] = "Actor"
        result = add_historical_features(target, history)

        self.assertEqual(result.loc[0, "director_previous_movie_count"], 2)
        self.assertEqual(result.loc[0, "franchise_previous_last_revenue"], 200.0)
        self.assertEqual(result.loc[0, "production_company_previous_movie_count"], 2)
        self.assertEqual(result.loc[0, "production_company_previous_max_revenue"], 200.0)
        self.assertEqual(result.loc[0, "cast_previous_movie_count"], 2)
        self.assertEqual(result.loc[0, "cast_previous_max_revenue"], 200.0)
        self.assertFalse(any(column.endswith("_new") for column in result.columns))

    def test_same_day_rows_never_see_each_other(self):
        base = {"release_date": "2020-01-01", "director": "A", "collection_id": 10,
                "production_company_ids": "p1", "cast_ids_top10": "a", "runtime_minutes": 90,
                "budget_usd": 20, "original_language": "en", "genres": "Drama",
                "production_countries": "US", "production_companies": "P", "cast_top10": "Actor",
                "profitable": 1, "vote_average": 7}
        movies = pd.DataFrame([{**base, "tmdb_id": 2, "worldwide_revenue_usd": 100},
                               {**base, "tmdb_id": 1, "worldwide_revenue_usd": 200}])
        result = add_historical_features(movies)
        self.assertEqual(result["director_previous_movie_count"].tolist(), [0, 0])
        self.assertTrue(result["director_previous_avg_revenue"].isna().all())

    def test_missing_profitability_contributes_revenue_not_failure(self):
        base = {"director": "A", "collection_id": None, "production_company_ids": "p1",
                "cast_ids_top10": "a", "runtime_minutes": 90, "budget_usd": None,
                "original_language": "en", "genres": "Drama", "production_countries": "US",
                "production_companies": "P", "cast_top10": "Actor", "vote_average": 7}
        history = pd.DataFrame([{**base, "tmdb_id": 1, "release_date": "2019-01-01", "worldwide_revenue_usd": 100, "profitable": pd.NA}])
        target = pd.DataFrame([{**base, "tmdb_id": 2, "release_date": "2020-01-01", "worldwide_revenue_usd": None, "profitable": pd.NA}])
        row = add_historical_features(target, history).iloc[0]
        self.assertEqual(row.director_previous_avg_revenue, 100)
        self.assertTrue(pd.isna(row.director_previous_success_rate))


if __name__ == "__main__":
    unittest.main()
