# Business Insights and Model Evaluation

## Executive summary

The project evaluates whether pre-release movie information can support worldwide box-office revenue prediction and two related business decisions: profitability risk and blockbuster risk.

The final model path uses one Random Forest Regressor for revenue and a separate Random Forest Classifier for the probability that a movie will exceed $400M. The classifier does not produce a second revenue estimate.

The large run contains 3,607 cleaned movies. Of these, 2,816 have both a reported budget and worldwide revenue and can be used for supervised modeling.

## Data quality decisions

- TMDB ID is the canonical identifier for deduplication and joins.
- Invalid dates are coerced to missing and excluded when release timing is required.
- Reported zero budgets and revenues are treated as missing because they commonly represent unreported values in TMDB.
- Movies without a reported budget remain available for descriptive analysis but cannot receive a budget-dependent profitability label or enter the supervised model.
- Revenue is not adjusted for inflation.
- Profitability means worldwide gross revenue greater than reported production budget. It is not accounting profit.

## Pre-release feature design

The model uses 38 features from information that could plausibly be available before release:

- budget, runtime, release year/month/season, language, genre, countries, companies, and cast size;
- franchise and sequel indicators;
- prior revenue and success history for the franchise, director, cast, and production companies.

Current-film popularity, vote count, vote average, revenue, ROI, and profitability are excluded from the pre-release feature list because they are affected by post-release performance.

Historical aggregates are shifted and calculated from earlier releases only. This prevents a movie from using its own outcome or a future movie's outcome as part of its history.

## Time-based evaluation

| Split | Years | Rows |
|---|---:|---:|
| Train | 2010–2019 | 2,111 |
| Validation | 2020–2021 | 242 |
| Test | 2022–2024 | 463 |

The chronological split is deliberately used instead of a random split. It better reflects the intended business use case: learning from historical releases and estimating performance for future releases.

Validation is used for model decisions. The final models are refit on Train plus Validation, and the Test period is kept for final out-of-sample evaluation.

## Revenue regression

The target is log-transformed during training because box-office revenue is strongly right-skewed. Predictions are converted back to dollars before calculating business metrics.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Ridge | $90.5M | $237.8M | 0.118 |
| Random Forest | **$72.0M** | **$176.8M** | **0.512** |
| Gradient Boosting | $80.6M | $203.3M | 0.355 |
| Median-revenue baseline | $118.2M | $266.5M | -0.108 |

The selected revenue model is Random Forest. Its Test MAE is approximately 39% lower than the median-revenue baseline.

An MAE of $72M means that the average absolute difference between predicted and actual revenue is about $72M. It does not mean that every movie is within $72M, and it does not mean the model is 72% accurate.

## Profitability classification

The profitability task predicts whether worldwide revenue exceeds reported budget.

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **66.1%** | 73.1% | 69.8% | **71.4%** | 72.5% |
| Random Forest | 62.9% | 71.7% | 64.1% | 67.7% | 71.8% |
| Gradient Boosting | 64.8% | 73.6% | 65.5% | 69.3% | **73.2%** |
| Majority baseline | 60.7% | — | — | — | — |

Logistic Regression gives the strongest F1 and accuracy in this experiment. Gradient Boosting gives the strongest ROC-AUC, so it ranks profitability probabilities slightly better. The choice depends on whether the business prefers balanced classification or probability ranking.

This label should be described as gross revenue exceeding reported budget, not as a complete profit forecast.

## Standalone blockbuster classifier

The separate classifier predicts whether a movie will exceed $400M worldwide revenue.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 89.3% | 92.7% |
| Precision | 32.4% | 56.1% |
| Recall | 92.3% | 88.1% |
| F1 | 48.0% | 68.5% |
| ROC-AUC | 96.6% | 97.6% |

The high Recall means the model identifies most movies that cross the threshold. Precision is lower, so some movies predicted to exceed $400M do not actually cross it.

The classifier is retained as a separate business signal. It does not feed a second revenue model in the final architecture.

## Error analysis

The selected revenue model performs differently across revenue levels:

| Segment | Movies | MAE |
|---|---:|---:|
| Regular, below $250M | 396 | $33.8M |
| Successful, $250M–$500M | 40 | $181.5M |
| Blockbuster, above $500M | 27 | $469.8M |

The model is reasonably useful for typical movies but systematically underpredicts extreme blockbusters. For the blockbuster segment, median actual revenue was approximately $845.6M while median predicted revenue was approximately $513.0M.

This is the most important limitation of the model. TMDB does not capture enough information about marketing scale, cultural momentum, brand awareness, release competition, or campaign reach to reliably identify every global phenomenon.

## Feature ablation

Removing groups of historical features reduced Validation performance:

- Full model, 38 features: Validation R² approximately 0.476.
- Basic features only, 13 features: Validation R² approximately 0.332.
- Removing company history reduced Validation performance.
- Removing cast history slightly improved one Test result but performed worse on Validation.

The full model is therefore retained. The result suggests that the additional history features provide useful signal and are not merely unnecessary complexity.

## Business interpretation

The system should be presented as two complementary outputs:

1. One revenue forecast from the selected Random Forest Regressor.
2. One blockbuster-risk probability from the standalone $400M classifier.

For example:

- Predicted worldwide revenue: $380M.
- Probability of exceeding $400M: 72%.

This is clearer than showing two competing revenue estimates. The revenue model answers “How much might the film earn?” while the classifier answers “How likely is it to cross a business-defined blockbuster threshold?”

## Recommendations

- Use the Random Forest revenue estimate as an input to scenario planning, not as a guaranteed forecast.
- Use the $400M probability to prioritize further review of high-potential films.
- Review predicted Blockbusters manually because false positives remain meaningful.
- Use genre, franchise, budget, and historical production data to compare opportunities, while considering the missing marketing and competition variables.
- Display sample sizes and uncertainty when comparing genres or directors.

## Limitations

- TMDB coverage and budget reporting are incomplete.
- Revenue is not inflation-adjusted.
- Gross revenue is not accounting profit.
- Marketing, distribution costs, exhibitor splits, taxes, and downstream revenue are unavailable.
- The collection process may overrepresent movies that are more visible on TMDB.
- The Test set contains relatively few blockbusters.
- Historical associations should not be interpreted as causal effects.
- Chronological validation reduces leakage risk but does not remove changes in market conditions over time.

## Reproducibility

The main code is under src. Raw data, processed data, fitted models, experiments, and logs are kept locally and excluded from Git because they are large generated artifacts.

The production-facing prediction path is:

- src/train_model.py for the main revenue model;
- src/train_blockbuster_classifier.py for the $400M risk classifier;
- src/predict.py for one revenue estimate and one blockbuster probability;
- dashboard/README.md for the Tableau presentation plan.
