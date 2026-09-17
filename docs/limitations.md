# Limitations

- TMDB Discover is a popularity/vote-ordered sample, not a probability sample.
- Source records are current snapshots, not true point-in-time snapshots. Credits,
  ratings, and metadata may have changed after release.
- Entity history is left-truncated at 2010, so early films appear to have less history.
- 791 of 3,607 cleaned movies lack a reported positive budget; their profitability
  is unknown and they are excluded from that classifier and all ROI denominators.
- Blockbusters are rare, making threshold metrics sensitive to a small number of films.
- COVID-era disruption and later market drift reduce temporal comparability.
- Worldwide gross is compared with reported production budget; marketing,
  distribution, exhibitor shares, taxes, and ancillary revenue are unavailable.
- Historical features are computed correctly from earlier release dates, but the
  current TMDB snapshot does not prove every underlying field was available in the
  same form before each release.
- Prediction intervals quantify empirical model error under this sample; they are
  not guarantees and do not include all business uncertainty.

- Earlier exploratory feature-stability reports compared models on 2022–2024. The
  current main selection code uses development validation, but historical holdout
  inspection cannot be undone; no never-inspected or single-use claim is made.
- The nominal 90% interval pools residuals from rolling-validation models. It is an
  empirical error interval, not standard split-conformal. Rare high-revenue films
  have substantially worse coverage than the overall sample.
- Calibration fit metrics reuse the calibrator training population. Only separate
  final-test metrics assess the fitted classifier/calibrator on new rows.
- Some legacy raw extracts lack individual sampling metadata, so exact source
  strategy provenance cannot be reconstructed for every extract.
