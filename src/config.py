"""Central, portable project configuration and modeling definitions."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
TABLEAU_DIR = PROJECT_ROOT / "dashboard" / "tableau"

RANDOM_SEED = 42
SCHEMA_VERSION = "2.0"
BLOCKBUSTER_THRESHOLD_USD = 400_000_000
FINAL_TEST_START_YEAR = 2022
FINAL_TEST_END_YEAR = 2024
INTERVAL_NOMINAL_COVERAGE = 0.90
