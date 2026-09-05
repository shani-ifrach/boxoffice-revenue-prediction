# Model card

## Model purpose

The project estimates worldwide movie revenue from information that could plausibly be available before release. It also provides two separate classification signals: profitability proxy and blockbuster risk.

## Final model roles

### Revenue model

Random Forest Regressor with 35 pre-release features. Three redundant features were removed after Large Run ablation: `company_count`, `log_budget_usd`, and `production_company_previous_max_revenue`. This is the only revenue model used in the production-facing prediction script. Ridge and Gradient Boosting were evaluated as alternatives.

### Blockbuster model

Random Forest Classifier with the same 35 pre-release features. It estimates the probability of exceeding $400M. It does not generate a second revenue estimate and does not feed another regression model.

### Profitability models

Logistic Regression, Random Forest, and Gradient Boosting were evaluated for the gross-over-budget classification task. This task is separate from revenue prediction.

## Training design

The modeling dataset contains 2,816 movies with both reported budget and worldwide revenue.

| Split | Years | Rows |
|---|---:|---:|
| Train | 2010–2019 | 2,111 |
| Validation | 2020–2021 | 242 |
| Test | 2022–2024 | 463 |

Models are selected or configured using Validation. Final models are refit on Train plus Validation. Test is retained for final out-of-sample evaluation.

Revenue is trained on log1p revenue because the target is heavily right-skewed. Dollar-space metrics are calculated after reversing the transformation.

## Results

### Revenue model

| Metric | Test result |
|---|---:|
| MAE | approximately $73.8M |
| RMSE | approximately $178.2M |
| R² | approximately 0.504 |

The median-revenue baseline had MAE of approximately $118.2M and R² of -0.108.

### Blockbuster classifier

| Metric | Test result |
|---|---:|
| Accuracy | 92.4% |
| Precision | 55.2% |
| Recall | 88.1% |
| F1 | 67.9% |
| ROC-AUC | 97.7% |

## Leakage controls

Current-film revenue, ROI, popularity, vote count, and vote average are not used as pre-release predictors. Historical entity features are shifted so only earlier releases are included.

Preprocessing is fitted inside scikit-learn Pipelines. Numeric values use median imputation and categorical values use most-frequent imputation followed by one-hot encoding.

## Intended use

The outputs can support early scenario planning, prioritization of films for further commercial review, comparison of expected performance across genres and franchises, and dashboard-based communication of uncertainty.

## Not intended for

The model is not a guaranteed financial forecast, an accounting profit calculation, a replacement for marketing or distribution analysis, or an automated green-light decision without human review.

## Known limitations

The largest errors occur for extreme blockbusters. In the Test error analysis, Regular movies had MAE of approximately $34.9M, movies between $250M and $500M had MAE of approximately $177.9M, and movies above $500M had MAE of approximately $489.9M.

The model tends to underpredict global phenomena because TMDB does not contain reliable measures of marketing scale, cultural momentum, release competition, or brand awareness. The blockbuster Test segment also contains relatively few movies.

## Governance and reproducibility

Raw data, processed data, fitted models, and experiment outputs are generated artifacts and are excluded from Git. The source code, documentation, cleaning rules, feature definitions, evaluation design, and model decisions are version controlled.

The model should be retrained when the data collection period or feature definitions change.
