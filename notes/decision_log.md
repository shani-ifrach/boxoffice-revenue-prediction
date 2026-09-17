# Decision log

- Use `tmdb_id`, not title, as the movie key; titles are not unique.
- For Tableau display/export only, rewrite the `title` field to `Title (Year)`
  when a title is duplicated; keep `tmdb_id` as the join and model key.
- Treat non-positive budget/revenue as unreported. Retain revenue-valid rows for
  regression and blockbuster modeling; require valid budget only for profitability.
- Use a canonical strict-date builder shared by training and inference.
- Use primary genre for the production model to keep the categorical contract stable;
  exploded multi-genre data remains available for EDA. Revisit multi-hot encoding
  only through rolling validation, never the final holdout.
- Select model families by mean rolling-validation MAE (regression) or PR-AUC
  (rare-event classification). The current trainer excludes 2022–2024 from model selection. Earlier exploratory
  stability outputs inspected this period, so a never-inspected holdout is not claimed.
- Optimize classifier threshold for at least 80% validation recall, then precision.
- Use Platt calibration fitted on pooled out-of-fold predictions because the positive
  class is too small for stable isotonic calibration.
- Use a nominal 90% empirical interval from pooled rolling-validation residuals.
  Do not claim standard split-conformal or distribution-free coverage guarantees.
- Use permutation-compatible model pipelines; impurity importance is not presented
  as causal evidence. SHAP was removed because no stable SHAP output is delivered.

- Preserve the final Tableau delivery while reproducing analysis outputs.
- Keep the final pipeline focused on 35 features; retain 38-feature comparisons only
  as research evidence. Avoid a duplicate CSV export layer.
- Treat after-calibration validation diagnostics as calibrator-fit metrics.
- Search a 0.005-step classification threshold grid across the complete [0, 1] range.
