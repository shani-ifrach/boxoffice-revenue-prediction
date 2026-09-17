#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$PROJECT_DIR/experiments/large_run"
RAW_DIR="$RUN_DIR/data/raw"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

mkdir -p "$RAW_DIR"
cd "$PROJECT_DIR"

echo "[1/5] Collecting popular movies..."
"$PYTHON_BIN" -m src.collect_tmdb \
  --start-year 2010 --end-year 2024 --pages-per-year 15 \
  --sort-by popularity.desc --output-dir "$RAW_DIR"

echo "[2/5] Collecting less popular movies..."
"$PYTHON_BIN" -m src.collect_tmdb \
  --start-year 2010 --end-year 2024 --pages-per-year 15 \
  --sort-by popularity.asc --output-dir "$RAW_DIR"

echo "[3/5] Collecting highly voted movies..."
"$PYTHON_BIN" -m src.collect_tmdb \
  --start-year 2010 --end-year 2024 --pages-per-year 15 \
  --sort-by vote_count.desc --output-dir "$RAW_DIR"

echo "[4/5] Merging raw extracts..."
"$PYTHON_BIN" -m src.merge_raw_extracts \
  --raw-dir "$RAW_DIR" \
  --output-path "$RAW_DIR/tmdb_movies_merged.json"

echo "[5/5] Cleaning, engineering features, training, and evaluating..."
cd "$RUN_DIR"
PYTHONPATH="$PROJECT_DIR" "$PYTHON_BIN" -m src.run_pipeline

echo "Experiment completed. Results: $RUN_DIR/reports/model_results.csv"
