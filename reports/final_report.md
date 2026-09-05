# Business Insights and Model Evaluation

This report is populated by the analyst after running the pipeline. Numeric findings are intentionally not pre-filled.

## Executive summary

**Decision question:** [What decision could this model support?]

**Finding:** [State the result using the generated metrics and charts.]

## Data quality decisions

- TMDB IDs were used for deduplication.
- Invalid dates were coerced to missing and excluded from release-timing analysis.
- Reported zero budgets and revenues were treated as missing, because they commonly mean “not reported” in this source.
- Movies without a usable revenue target were excluded from supervised regression.

## EDA questions to answer

- Budget vs revenue and whether the relationship changes at higher budgets.
- Revenue and simple gross-over-budget ROI by genre.
- Release month/season differences, with sample sizes shown.
- Rating/popularity association in descriptive analysis only.
- Director history association, with a warning that this is observational.

## Model results

Insert the generated values from `models/metrics.json`. Explain where the model underpredicts blockbusters, whether errors differ by genre, and why the selected model is appropriate for the business use case.

## Recommendations and limitations

Recommendations should be conditional on the model's error and calibration. Do not use the simplified profitability label as an accounting forecast.
