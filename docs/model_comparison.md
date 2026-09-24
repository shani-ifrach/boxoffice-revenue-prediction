# Exploratory revenue-model comparison

This is a follow-up to the production model selection documented in the
[model card](model_card.md). The production trainer selected Random Forest from
Ridge, Random Forest, and Gradient Boosting by mean MAE over three expanding
validation windows. The comparison here adds four exploratory families. It does
not replace the saved production model.

## Protocol and reproducibility

All candidates use the same 3,607-row processed feature table and the ordered
35-feature contract. They fit `log1p(worldwide_revenue_usd)`, convert predictions
back to dollars with the production training-derived cap, and report MAE, RMSE,
and R² in dollars. The three selection windows train through 2015 and validate
on 2016–2017, train through 2017 and validate on 2018–2019, then train through
2019 and validate on 2020–2021. The ranking uses the unweighted mean of the
three validation MAEs. After each candidate is fit on all development years
through 2021, its 2022–2024 result is reported separately.

The original candidates, LightGBM, and XGBoost use the production
numeric imputation/scaling and one-hot categorical preprocessing. Histogram
Gradient Boosting uses equivalent dense one-hot output. CatBoost keeps the same
feature columns but uses its native categorical handling, with missing categories
represented explicitly. These are fixed example settings, not a hyperparameter
search. The code and exact settings are in
[`scripts/compare_revenue_models.py`](../scripts/compare_revenue_models.py) and
the generated [`metadata.json`](../reports/model_comparison/metadata.json).

From the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-comparison.txt
.venv/bin/python scripts/compare_revenue_models.py
```

The optional comparison dependencies are pinned separately from production.
The script records the input SHA-256, runtime versions, seed, features, folds,
model settings, per-window scores, and summary in `reports/model_comparison/`.
It writes no production model artifacts. For a shorter run, pass `--models`
followed by model names from the summary; the full seven-model run is the
comparison documented below.

## Results

Lower MAE is better. Values are rounded to two decimal places in USD millions;
the [summary CSV](../reports/model_comparison/summary.csv) and
[per-window CSV](../reports/model_comparison/by_split.csv) retain full precision.

| Model | Mean validation MAE | 2022–2024 MAE | 2022–2024 R² |
|---|---:|---:|---:|
| Random Forest (current) | **$55.01M** | **$57.52M** | **0.519** |
| LightGBM | $55.58M | $62.56M | 0.416 |
| CatBoost | $55.99M | $61.75M | 0.392 |
| Histogram Gradient Boosting | $56.09M | $61.59M | 0.430 |
| XGBoost | $56.91M | $62.41M | 0.400 |
| Gradient Boosting (original candidate) | $57.06M | $62.74M | 0.380 |
| Ridge (original candidate) | $133.52M | $133.47M | -4.098 |

Random Forest retains the lowest mean validation MAE in this expanded candidate
set and also has the lowest MAE on 2022–2024. The saved production artifact
therefore remains unchanged. This is still a later exploratory comparison, not
a new production selection exercise. A future evaluation on genuinely new
releases, with the candidate and decision rule fixed beforehand, would provide
stronger evidence about whether to replace it.

Earlier feature-stability work also inspected 2022–2024. These later comparisons
inspect it again, so this period cannot be described as a never-seen, single-use
test set. The table is useful as a historical performance check, not an unbiased
new selection result. Full reconstruction from TMDB collection additionally
requires the original raw extracts; this comparison itself runs from the
processed CSV included in the repository.
