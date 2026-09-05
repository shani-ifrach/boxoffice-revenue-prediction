import unittest

import pandas as pd

from src.train_model import FULL_PRE_RELEASE_FEATURES, PRE_RELEASE_FEATURES, add_history_from_prior_period


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
        result = add_history_from_prior_period(target, history)

        self.assertEqual(result.loc[0, "director_previous_movie_count"], 2)
        self.assertEqual(result.loc[0, "franchise_previous_last_revenue"], 200.0)
        self.assertEqual(result.loc[0, "production_company_previous_movie_count"], 2)
        self.assertEqual(result.loc[0, "production_company_previous_max_revenue"], 200.0)
        self.assertEqual(result.loc[0, "cast_previous_movie_count"], 2)
        self.assertEqual(result.loc[0, "cast_previous_max_revenue"], 200.0)
        self.assertFalse(any(column.endswith("_new") for column in result.columns))


if __name__ == "__main__":
    unittest.main()
