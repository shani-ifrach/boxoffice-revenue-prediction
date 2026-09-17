# Final model and business report

## Scope and data quality

The final local dataset contains 3,607 unique TMDB movies released from 2010–2024.
Revenue-valid films remain in canonical history even when budget is missing. Of
these, 2,816 have a valid positive budget and a defined gross-over-budget label;
791 have missing profitability and are excluded from both numerator and denominator.
Seven non-positive runtimes are missing after cleaning. Tiny positive financial
values are measured, not removed by an unsupported cutoff.

## Selection protocol

Model selection uses expanding validation: training through 2015 for 2016–2017,
through 2017 for 2018–2019, and through 2019 for 2020–2021. MAE selects regression;
PR-AUC selects classifiers. Thresholds, calibration, feature ablation, and empirical interval
width use development validation in the current training code. Earlier exploratory
stability comparisons also reported 2022–2024 results; this report does not claim
that the period was inspected only once or can establish the historical order of
all selection decisions. The current stability script uses validation years only.

In the fixed 120-tree feature-ablation experiment, the 35-feature variant averaged
$55.2M validation MAE versus $55.7M for the 38-feature variant. The main family
comparison uses 250-tree Random Forest, with $55.0M mean validation MAE. Ridge
remained far worse and is reported rather than hidden. The final Random Forest was
retrained through 2021 and evaluated on 2022–2024.

## Final results

| Revenue metric | Random Forest | Median baseline |
|---|---:|---:|
| MAE | $57.5M | $94.2M |
| RMSE | $156.0M | $237.4M |
| R² | 0.519 | -0.113 |

MAE improved 39.0%. Error remains concentrated in the upper tail:

| Actual revenue band | n | MAE | Interval coverage |
|---|---:|---:|---:|
| Regular, below $250M | 550 | $27.5M | 97.5% |
| High Grossing, $250M–$400M | 26 | $172.1M | 38.5% |
| Blockbuster, above $400M | 42 | $380.2M | 28.6% |

The nominal 90% empirical rolling-residual interval has 90.3% overall holdout coverage and $185.5M average
clipped width. Segment results show that overall coverage masks poor tail coverage;
the interval represents empirical model error, not a guarantee. It pools residuals
from multiple validation models and is not a standard split-conformal implementation.

## Classification

Profitability Logistic Regression on 463 budget-valid test films: precision 75.3%,
recall 61.9%, F1 68.0%, ROC-AUC 71.3%, PR-AUC 81.0%, Brier 0.2079.

The calibrated $400M Gradient Boosting classifier uses validation threshold 0.065
for a goal of at least 80% recall. On 618 test films: precision 59.6%, recall 81.0%,
F1 68.7%, ROC-AUC 98.2%, PR-AUC 76.6%, Brier 0.0339, with TN/FP/FN/TP of
553/23/8/34. Accuracy alone is misleading because only 42 films are positive.

Calibration fit metrics and before/after curves use the same population that fits
the calibrator. They are diagnostics, not independent evidence of calibration
improvement; the reported final-test classifier metrics are separate.

## Explainability and limitations

Permutation importance identifies budget, franchise status, and prior production-
company revenue as the strongest global contributors. These are predictive, not
causal. TMDB sampling is visibility-biased and current-snapshot rather than true
point-in-time; history begins in 2010; budgets are incomplete; blockbusters are rare;
COVID/market drift matter; and marketing/distribution economics are absent.
