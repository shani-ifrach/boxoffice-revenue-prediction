# Model card — artifact schema 2.0

## Intended use

Estimate worldwide gross for scenario planning and flag films that may exceed
$400M. The models are not accounting-profit forecasts, guarantees, causal tools,
or automated green-light decisions.

## Data and evaluation

The canonical dataset contains 3,607 unique TMDB films released 2010–2024. Revenue
and blockbuster models use every revenue-valid row; profitability uses only 2,816
rows with reported positive budget. Model families are selected on expanding
validation windows 2016–2017, 2018–2019, and 2020–2021. The 618-row 2022–2024
period is reserved for final evaluation in the main trainer. Earlier exploratory
stability outputs also compared models on this period; no single-use or never-inspected
holdout claim is made. Current stability comparisons exclude 2022–2024.

## Final revenue model

Random Forest with 35 pre-release fields, selected by mean validation MAE. Holdout:
MAE $57,521,066; RMSE $155,983,079; R² 0.5193. Development-median baseline MAE is
$94,208,753. The nominal 90% empirical residual interval has $152,957,796 absolute half-width,
90.29% empirical holdout coverage, and $185,513,987 average clipped width.

The interval uses pooled rolling-validation residuals from earlier models. It is
not standard split-conformal and has no formal coverage guarantee. Legacy artifact
field names retain `conformal_*` for compatibility; `interval_method` identifies
the actual method.

Ridge is retained in comparison results. Its training-derived log ceiling prevents
overflow, but its weak temporal performance is not hidden.

## Classifiers

The profitability Logistic Regression is selected by temporal PR-AUC. On the
budget-valid holdout (n=463): precision 75.3%, recall 61.9%, F1 68.0%, ROC-AUC
71.3%, PR-AUC 81.0%, Brier 0.2079.

The $400M Gradient Boosting classifier uses all 618 holdout rows. Its validation-only
threshold is 0.065 after Platt calibration, chosen to retain at least 80% recall.
Holdout precision is 59.6%, recall 81.0%, F1 68.7%, ROC-AUC 98.2%, PR-AUC 76.6%,
and Brier 0.0339 (TN 553, FP 23, FN 8, TP 34). Calibration curves and fit metrics reuse the calibrator-fitting population; they
are not independent validation of calibration improvement. Final-test metrics use
rows outside calibrator fitting. The displayed value is a calibrated
probability under this sample; drift monitoring would be required in production.

## Leakage and parity controls

Current-film outcomes and engagement fields are excluded. Entity history is built
only from strict earlier dates, same-day rows are batched, and profit history ignores
unknown labels. Training and inference share code. Pipelines embed imputation and
unknown-category handling; artifacts embed ordered features and expected types.

## Explainability

Permutation importance is the recommended global diagnostic because it measures
held-out performance impact. Impurity importance may favor continuous or
high-cardinality features and must not be treated as causal. SHAP is not a declared
dependency because no stable, verified SHAP artifact is delivered.

## Risks and limitations

Major risks are sampling bias, current rather than point-in-time TMDB snapshots,
history truncated at 2010, missing budgets, rare blockbusters, COVID/market drift,
and omission of marketing/distribution economics. See `docs/limitations.md`.

## Governance

Seed, runtime versions, split, feature count, timestamp, and input rows are saved in
`models/reduced/training_manifest.json`; metrics are in `metrics.json`. Retrain when
source coverage, definitions, features, or market regime changes.
