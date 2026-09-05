# Power BI dashboard specification

The `.pbix` file is intentionally not generated automatically. After running the pipeline, import `data/processed/movies_features.csv` and `models/metrics.json` (flattened to a table) into Power BI.

## Pages

1. **Executive Overview:** movie count, average/median revenue, average budget, median ROI, profitability rate, yearly revenue trend, revenue by genre, budget vs revenue, and a revenue distribution.
2. **Movie Performance:** top revenue titles, top simple-ROI titles with a minimum budget filter, genre/season comparisons, and sample sizes.
3. **Prediction:** input controls matching `src/predict.py`; show predicted revenue, profitability probability, and a note that predictions are estimates from a historical model.

Include TMDB attribution in an About/Credits section and label the profitability measure as “gross revenue > reported production budget,” not accounting profit.
