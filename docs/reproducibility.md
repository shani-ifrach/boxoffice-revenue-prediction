# Reproducibility

Use **Python 3.12–3.14**. NumPy 2.5.2 requires at least Python 3.12. Dependencies are
pinned in `requirements.txt`; local verification used Python 3.14.3. CI is configured
for 3.12, 3.13 and 3.14. A configured workflow is not a claim that remote CI has
already passed.

## What is included in the repository

- Current Python source, tests and a synthetic sample movie record.
- `data/processed/movies_clean.csv` and `movies_features.csv`.
- Final 35-feature model artifacts, metrics and training manifest in `models/reduced`.
- Analytical reports, error tables and charts in `reports`.
- The final Tableau workbook and saved dashboard CSV sources.

Original API responses are not committed, and no files under `data/raw` are included
in review archives. Source-quality reports must also be untracked before any public
release.

## Review saved results and run prediction

From the repository root, create a virtual environment, install the pinned
dependencies, then run the tests and offline prediction:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.predict --record-json examples/movie_record.json
```

These commands use macOS/Linux executable paths. On Windows, use
`.venv\Scripts\python.exe` and install with `python.exe -m pip` from that environment.

Once dependencies are installed, tests and saved-model prediction need neither
network access nor a TMDB API key. The prediction reads the included processed
history and saved revenue/blockbuster models. The example is synthetic, and any
outcomes present in it are discarded when building current-film model inputs.

The final dashboard can be opened independently in Tableau:
`dashboard/tableau/boxoffice_dashboard.twbx`.

## Reconstruct the complete pipeline

To reproduce cleaning and training from the original snapshot, use an authorized
local collection and place its timestamped `tmdb_movies_*.json` extracts under
`data/raw`. These responses are not distributed with the project. A merged file
alone is not enough for the default merge step. Then run from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python -m src.run_pipeline
```

The run merges the extracts, cleans data, builds historical features, trains and
evaluates the final models, validates prediction/evaluation tables and generates
EDA. It refreshes `data/processed`, `models/reduced` and analytical outputs in
`reports`. The Tableau workbook and active dashboard CSVs are preserved.

Raw TMDB API responses and timestamped extracts remain local and are never included
in the portfolio archive. A review archive containing code, processed analytical
artifacts, saved models and the dashboard can be built with:

```bash
.venv/bin/python scripts/build_portfolio_package.py
```

The script checks ZIP integrity, verifies that no `data/raw` files are present and
includes SHA-256 file checksums. Its output is a separate delivery artifact rather
than a file to commit to Git.

## New collection and research comparisons

New collection with `src.collect_tmdb` requires `TMDB_API_KEY` as an environment
variable. Do not include the key in the repository or in an archive. Fresh API
responses may differ from the saved 2026-09-04 snapshot and are not guaranteed to
reproduce its published metrics. Some legacy extracts lack individual sampling
metadata; unknown source strategies remain explicit in the coverage report.

Randomized estimators use seed 42. Final artifacts embed preprocessing and the
ordered feature contract; the training manifest records package versions, split
sizes and timestamp.

The 38-feature list is retained only for feature-selection comparisons. Optional
development-validation experiments use the included processed feature table:

```bash
.venv/bin/python -m src.feature_ablation
.venv/bin/python -m src.feature_stability --input-path data/processed/movies_features.csv --output-path reports/feature_stability_results.csv
```

The final pipeline and prediction use the 35-feature model contract. There is no
second full-feature pipeline option or duplicate CSV export layer.

The later [revenue-model comparison](model_comparison.md) uses the same processed
feature table and a separate pinned dependency file. Run it with:

```bash
.venv/bin/pip install -r requirements-comparison.txt
.venv/bin/python scripts/compare_revenue_models.py
```

Its versioned summary, per-window scores, and input checksum are under
`reports/model_comparison/`. This is an exploratory comparison; it does not
replace the final saved artifacts or create a never-inspected test set.
