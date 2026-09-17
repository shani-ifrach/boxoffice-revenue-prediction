# Reproducibility

Use **Python 3.12–3.14**. NumPy 2.5.2 requires at least Python 3.12. Dependencies are
pinned in `requirements.txt`; local verification used Python 3.14.3. CI is configured
for 3.12, 3.13 and 3.14. A configured workflow is not a claim that remote CI has
already passed.

## What is included in the repository

- Current Python source, tests and a sample movie record.
- `data/processed/movies_clean.csv` and `movies_features.csv`.
- Final 35-feature model artifacts, metrics and training manifest in `models/reduced`.
- Analytical reports, error tables and charts in `reports`.
- The final Tableau workbook and saved dashboard CSV sources.
- Available collection metadata, coverage and duplicate reports under `data/raw`.

The large timestamped TMDB response files are not committed. The small raw-directory
reports describe the source data; they cannot replace the actual movie records.

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
history and saved revenue/blockbuster models. Outcomes present in the example
record are discarded when building current-film model inputs.

The final dashboard can be opened independently in Tableau:
`dashboard/tableau/boxoffice_dashboard.twbx`.

## Reconstruct the complete pipeline

To reproduce cleaning and training from the original snapshot, obtain the original
raw-data files separately and place the timestamped `tmdb_movies_*.json` extracts
under `data/raw`. A merged file alone is not enough for the default merge step.
Then run from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python -m src.run_pipeline
```

The run merges the extracts, cleans data, builds historical features, trains and
evaluates the final models, validates prediction/evaluation tables and generates
EDA. It refreshes `data/processed`, `models/reduced` and analytical outputs in
`reports`. The Tableau workbook and active dashboard CSVs are preserved.

The existing local archive `dist/boxoffice_portfolio_submission.zip` can supply
these extracts if provided separately. It is not part of the Git repository and is
not presented here as an available public download. Before publishing a package,
check that it contains only the intended public files and excludes internal notes
and superseded experiments.

If the complete local raw-data collection is available, an archive can be built
with:

```bash
.venv/bin/python scripts/build_portfolio_package.py
```

The script checks ZIP integrity and includes SHA-256 file checksums. Its output is
a separate delivery artifact rather than a file to commit to Git.

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
