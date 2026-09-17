# Test strategy

The suite covers financial missingness, deduplication, strict-date/same-day history,
shared training/inference features, prediction with unseen categories and missing
budgets, and rejection of incompatible model artifacts.

Regression tests check that feature stability excludes final-test years, threshold
selection can satisfy recall below 0.05, and analytical tables reject invalid keys,
intervals, probabilities or mismatched evaluation values. Delivered dashboard data
relationships are checked without modifying the sources.

```bash
.venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

The final 35-feature pipeline is the integration check. Offline prediction tests
saved-model loading and scoring without an API credential. CI is configured for
Python 3.12, 3.13 and 3.14; local verification uses Python 3.14.3.

CSV checks do not establish visual dashboard correctness. Existing Tableau
artifacts and active sources are preserved throughout analysis runs.
