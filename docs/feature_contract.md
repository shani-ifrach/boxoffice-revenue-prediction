# Feature contract v2.0

The final inference contract contains 35 ordered pre-release features, saved
inside each final artifact. The original 38-feature list remains in
`src/train_model.py` for feature-ablation/stability comparisons only; it is not
an additional deployed or dashboard model.

Numeric missing values are median-imputed inside the fitted pipeline. Categorical
missing values use the training mode and unseen categories are ignored safely by
one-hot encoding. A missing or zero budget is represented as `NaN`, never as zero.

Historical features use the canonical cleaned population: every movie with a
valid release date and revenue may contribute revenue history, even if its budget
is missing. Profitability history ignores rows whose profitability is unknown.
Eligible history always satisfies `historical_release_date < release_date`; all
same-day releases see identical prior history. Director ID, company ID, cast ID,
and collection ID are used when available. The first credited director remains the
primary director for compatibility; all director IDs are retained for future work.

Current-film revenue, profitability, ROI, popularity, vote count, and vote average
are never model inputs. Historical ratings are retained but are subject to the
snapshot limitation described in `limitations.md`.
