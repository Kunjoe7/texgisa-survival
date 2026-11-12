# Current API

`SurvivalRecipe(recipe="RefineCDF", dataset="custom", config_id=0, config=None, random_state=271, threads=1)` implements `fit(X,time,event)`, `predict_risk(X)`, `predict_survival(X,times=None)`, `predict_time(X)`, `explain(X,integration_steps=20,time_samples=None)` and `save(path)`.

Pandas column names/order must match at prediction. NumPy inputs receive x0,x1,... column names. Features must be finite numeric values; explicit category encoding belongs to the caller. Named domain rules validate their required feature names. No test outcomes are accepted by prediction methods. The reference for TEXGI and all internal transformations are fitted on training subsets.

`metadata_` records fitting/stopping/calibration row indices, epoch selection, calibration, prior diagnostics and output semantics. `config_` records the effective configuration. `feature_names_in_`, `n_features_in_` and `time_grid_` expose input/grid metadata. Recipe save files contain model objects and must be treated as trusted-code artifacts.

`available_recipes()` lists8 callable own variants. `backends.runner.available_methods()` lists20 public-package/author-array routes plus3 author CLI routes. `fit_predict` returns a dictionary rather than falsely promising every native estimator implements every prediction type. `run_author_cli` accepts an original author checkout, argv and an optional interpreter/log path; exceptions preserve author failures. Source code is never modified by the runner.

The inherited `TexGISa` class remains available at package root. Its legacy constructor and rule methods are defined in `api.py`; current benchmark numbers correspond to `SurvivalRecipe`, not a default legacy estimator.
