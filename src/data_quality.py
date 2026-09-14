"""Generate an auditable data-quality profile without arbitrary row deletion."""
import json
from pathlib import Path
import pandas as pd


def build_quality_report(input_path=Path("data/processed/movies_clean.csv"), output_path=Path("reports/data_quality_report.json")):
    """Profile row counts, missingness, validity rules, and financial sensitivity."""
    data = pd.read_csv(input_path, parse_dates=["release_date"])
    report = {
        "rows": len(data), "unique_tmdb_ids": int(data.tmdb_id.nunique()),
        "duplicate_tmdb_ids": int(data.tmdb_id.duplicated().sum()),
        "year_min": int(data.release_year.min()), "year_max": int(data.release_year.max()),
        "missing_rates": {c: float(data[c].isna().mean()) for c in data.columns},
        "valid_profitability_rows": int(data.profitable.notna().sum()),
        "missing_profitability_rows": int(data.profitable.isna().sum()),
        "profitability_rate_valid_only": float(data.profitable.mean()),
        "runtime_missing_after_nonpositive_rule": int(data.runtime_minutes.isna().sum()),
        "sensitivity_counts_not_removed": {
            "budget_below_1k": int(data.budget_usd.between(0, 1_000, inclusive="neither").sum()),
            "budget_below_100k": int(data.budget_usd.between(0, 100_000, inclusive="neither").sum()),
            "revenue_below_1k": int(data.worldwide_revenue_usd.between(0, 1_000, inclusive="neither").sum()),
            "revenue_below_100k": int(data.worldwide_revenue_usd.between(0, 100_000, inclusive="neither").sum()),
        },
        "financial_quantiles": {c: {str(q): float(data[c].quantile(q)) for q in (0.01, 0.5, 0.99)} for c in ("budget_usd", "worldwide_revenue_usd")},
        "missing_entity_names": {c: int(data[c].isna().sum()) for c in ("director", "production_companies", "cast_top10")},
        "policy": "Only non-positive reported finance values are treated as missing. Tiny positive values are flagged, not removed, because TMDB supplies no defensible universal cutoff.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__": build_quality_report()
