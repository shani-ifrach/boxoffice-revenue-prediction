# Tableau dashboard

The dashboard presents financial patterns, individual-film comparisons and
forecast performance using the project’s cleaned data and model outputs.

[View the interactive dashboard on Tableau Public](https://public.tableau.com/app/profile/shani.ifrach4420/viz/BoxOfficeRevenuePrediction-AnalysisModelEvaluation/ExecutiveOverview).

For local review, open [boxoffice_dashboard.twbx](tableau/boxoffice_dashboard.twbx).
This is the final portfolio artifact. Other workbook files are earlier versions.

## Final pages

- **Executive Overview:** dataset-level financial patterns and summary indicators.
- **Movie Explorer:** individual-film performance, saved model outputs and comparisons.
- **Revenue Model Evaluation:** actual versus predicted revenue and model errors.

The separate $400M classifier is documented in the model report; no additional
Blockbuster Risk page is required for the final dashboard.

## Reading the dashboard

Worldwide revenue means TMDB reported worldwide gross. Gross-over-budget
profitability compares reported gross with production budget; it is not accounting
profit. Net return multiple is `(revenue - budget) / budget` and excludes unknown
budgets. Blockbuster means revenue strictly above $400M.

Saved model-quality predictions cover 2022–2024 evaluation movies. An unscored film
has no saved test prediction; it must not be displayed as a zero-dollar forecast.
Peer comparisons based on observed outcomes are retrospective. Historical ratings,
ROI, votes and popularity are descriptive fields, rather than current-film model inputs.

## Data sources

`tableau/data` contains the saved dashboard sources. The final analysis pipeline
produces cleaned data, model predictions and error tables directly in
`data/processed`, `models/reduced` and `reports`. It preserves the delivered
workbook and its active sources. See [data contract](../docs/tableau_data_contract.md).

## Data source attribution

[![The Movie Database (TMDB)](../docs/assets/tmdb-logo.svg)](https://www.themoviedb.org)

This product uses the TMDB API but is not endorsed or certified by TMDB. The logo
above is an unmodified approved TMDB asset.
