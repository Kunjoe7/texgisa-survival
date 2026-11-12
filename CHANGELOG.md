# Changes

## 2.2.0 (2026-11-12)

- Add `skills/texgisa-prior-finder`, a step-by-step procedure for finding expert priors
  (important-feature lists and directional rules) in published sources, with provenance and a
  final review by a domain expert.
- Add `SurvivalRecipe`, the tuned default configuration of TexGISa, with separate
  `predict_risk`, `predict_survival`, `predict_time` and `explain` outputs.
- Add `backends.runner`: one call (`fit_predict`) to 20 established survival models from
  lifelines, scikit-survival, pycox and the original ICDM 2021 code, each run unchanged in its own
  process; `run_author_cli` runs DRO, survival kernets and SODEN through their authors' programs.
- Keep fitted preprocessing, training partitions and calibration diagnostics with the model.
- Benchmark records, configurations and summaries in `benchmarks/`.

## 2.1.0

- Benchmark scripts and per-fold results in `benchmarks/`; defaults set from nested
  cross-validation.

## 2.0.1

- Attribution priors now back-propagate to the network (`create_graph=True`).
- Fix the sign of `concordance_index`, rule thresholds on standardized features,
  save/load, and the Brier-score column alignment.
- Fix `MySATrainer` on recent PyTorch releases.

## 2.0.0

- First public release on GitHub.
