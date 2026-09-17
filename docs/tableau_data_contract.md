# Tableau data contract

`tmdb_id` is the non-null unique analytical key in cleaned, feature and prediction
tables. Exploded genres form a separate one-to-many table. Dashboard display titles
include a year for duplicate titles and a TMDB ID when title and year both repeat.

The analysis writes its direct outputs to `data/processed`, `models/reduced` and
`reports`. There is no additional CSV export or staging layer. These tables are
useful for reviewing the data and predictions without opening Tableau.

`validate_analysis_outputs()` checks keys, canonical titles, matching prediction
and evaluation populations, final-test years, finite numeric outputs, probabilities
and correctly ordered intervals. It also checks that evaluation values match saved
predictions. The same validator checks the delivered dashboard CSV relationships.

The final workbook and `dashboard/tableau/data` contain the delivered dashboard
snapshot and are preserved when the pipeline runs. Current saved metrics and
predictions remain consistent with the dashboard delivery.

Prediction intervals use `prediction_lower_usd`, `predicted_revenue_usd` and
`prediction_upper_usd`. Nominal 90% intervals come from empirical rolling-validation
residual quantiles and have no formal coverage guarantee.

Profitability stays nullable when budget is unknown. Rates must use a non-null
denominator; missing budgets must not become failure labels.
