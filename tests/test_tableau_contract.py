import unittest
from pathlib import Path

from src.data_quality import validate_analysis_outputs


class DashboardDataTests(unittest.TestCase):
    def test_final_dashboard_data_relationships(self):
        directory = Path("dashboard/tableau/data")
        self.assertTrue(validate_analysis_outputs(
            directory / "movies_clean.csv", directory / "movies_features.csv",
            directory / "test_predictions.csv", directory / "actual_vs_predicted.csv"))
