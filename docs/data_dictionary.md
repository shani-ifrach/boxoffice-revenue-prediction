# Data dictionary

This document describes the main analytical fields used in the cleaned and feature-engineered movie tables.

## Source and identifiers

| Field | Meaning | Use |
|---|---|---|
| tmdb_id | TMDB's unique movie identifier | Deduplication and joins |
| title | Movie title from TMDB | Reporting and dashboard labels |
| release_date | Reported release date | Time split and release features |
| release_year | Year extracted from release_date | Time split and trend analysis |

## Financial fields

| Field | Meaning | Availability |
|---|---|---|
| budget_usd | Reported production budget | Pre-release predictor, if known |
| worldwide_revenue_usd | Reported worldwide gross revenue | Regression target and historical analysis |
| profitable | 1 when revenue is greater than budget, 0 otherwise, missing when budget/revenue is invalid | Classification target |
| roi_simple | Revenue minus budget, divided by budget | Descriptive analysis only |
| log_budget_usd | log1p of reported budget | Retained engineered field; excluded from the production model |
| budget_category | Budget quartile created for descriptive analysis | Dashboard grouping only |

Zero budget and revenue values are treated as missing during cleaning because they commonly indicate unreported values in TMDB. Rows without revenue cannot define the target and are excluded; rows without budget remain, with nullable profitability. Profitability rates exclude those rows from both numerator and denominator. Profitability is a simplified gross-over-budget proxy, not accounting profit.

## Movie and release attributes

The table also contains runtime_minutes, original_language, genres, primary_genre, genre_count, production_countries, country_count, production_companies, production_company_ids, company_count, collection_id, is_franchise, is_sequel, director, director_id, director_ids, cast_top10, cast_ids_top10, cast_size_top10, release_month, release_season, is_summer_release, and is_holiday_release.

These fields describe the film and can be available before release, subject to the quality and timing of the TMDB record.

## Historical entity features

Historical features are calculated only from strictly earlier release dates. The current movie and every movie released on the same day are excluded from one another's history.

The same pattern is used for franchise, director, cast, and production-company fields:

- previous_movie_count
- previous_avg_revenue
- previous_max_revenue
- previous_success_rate
- previous_avg_rating where available

Additional franchise features include previous_last_revenue, previous_median_revenue, and previous_latest_success.

These fields approximate prior commercial track record and brand familiarity. They are not causal measures of talent, quality, or marketing power.

## Post-release descriptive fields

Popularity, vote_average, vote_count, worldwide_revenue_usd, profitable, and roi_simple are retained for EDA but excluded from the pre-release feature list for the current movie. Current-film popularity and voting signals reflect audience activity after release and would create leakage in a pre-release prediction setting.
