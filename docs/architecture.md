# Architecture

```text
TMDB API (optional collection)
  -> timestamped raw JSON
  -> merge by tmdb_id + coverage/duplicate reports
  -> cleaned movie table (nullable financial labels)
  -> shared strict-date feature builder
  -> temporal validation within 2010–2021
  -> final 35-feature artifacts + 2022–2024 evaluation
  -> validated prediction/evaluation tables, EDA and charts

Tableau dashboard <- saved analytical outputs
```

Collection is separated from local reproduction. Some legacy extracts lack
individual sampling metadata; the coverage report marks unknown strategies.

The final pipeline writes models to `models/reduced`, cleaned/features data to
`data/processed`, and analysis to `reports`. There is one feature contract for final
inference: 35 ordered fields. Artifacts embed preprocessing, task definitions and
runtime manifests, and are validated before prediction.

Feature-ablation and stability scripts document selection using development-period
validation, including comparison with the original 38-feature list. These research
comparisons are not additional models required to reproduce the final dashboard.

The pipeline validates analytical keys, canonical titles, matching prediction and
evaluation populations, final-test years, numeric predictions and intervals. It
writes the direct analysis outputs without a separate export/staging layer.
The final Tableau workbook and active data sources are preserved.
