# Existing-method backends

Native inputs: finite numeric fitting matrix, observed durations, binary event indicator, and query matrix. No user innovation modules are fed into a baseline.

| Route | Provider | Outputs | Dependency/runtime |
|---|---|---|---|
| Cox, Weibull, Lognormal, Loglogistic | lifelines | native risk, survival and median | baselines extra |
| Coxnet, RandomSurvivalForest, GradientBoostingCox | scikit-survival | native risk and survival | baselines extra |
| DeepSurv, CoxTime, CoxCC, DeepHit, LogisticHazard, PCHazard, PMF, MTLR, NMTLR | pycox | estimator-specific risk and/or native curve | baselines extra |
| HarrellNMSE, HarrellNMAE, UnoNMSE, UnoNMAE | original ICDM2021 source | scalar time and negative-time risk, no curve | separate NumPy1.x interpreter; author_source argument |
| DRO | author run_dro_cox_split.py | original CLI outputs/files | run_author_cli; original formats/environment |
| Kernet | author demo_tuna_kernet.py | original CLI outputs/files | run_author_cli; teacher/config files as upstream requires |
| SODEN | author main.py | original CLI outputs/files | run_author_cli; original formats/environment |

`fit_predict(...,python_executable=...,author_source=...)` isolates original scalar-time code. The6 original ICDM2021 source hashes are verified before import. No missing NumPy aliases are injected. Author source: https://github.com/yanlirock/local_global_survival at commit2eee054ccbf8b8ab9750c79d640ce72ef8f836d1.

Classical public configurations use `pen` and Cox `l1`; Coxnet uses `alpha,l1_ratio`; RSF uses `n_estimators,max_depth,min_samples_leaf`; boosted Cox uses `n_estimators,learning_rate,max_depth,min_samples_leaf,subsample`. Neural configurations use the frozen `native_configs.json` keys, including hidden/dropout/lr/wd/batch/epochs/patience and model-specific bins/alpha/sigma. Defaults are the first saved configuration, not an automatic test-selected best model.

Native arrays are not clipped or repaired. Deriving a common evaluation metric from them is a separate operation. Infinities or unsupported outputs remain visible. Author CLI data adapters in the companion reproduction directory document the exact experimental IO changes; they do not alter author losses, networks or optimizers. A method's upstream data-format requirements remain part of the contract.

Optional libraries are installed from their distributions; they are not relicensed as MIT by this wrapper. Keep GPL-licensed dependency terms and any upstream author-source terms when redistributing combined works. This archive does not bundle those author implementations.

## Method provenance

The original scalar-time objectives are described by Wang, Li and Chignell, "Combining Ranking and Point-wise Losses for Training Deep Survival Analysis Models," ICDM2021,689–698, DOI10.1109/ICDM51629.2021.00080. Survival kernets: Chen, JMLR25(40):1–78,2024, https://jmlr.org/papers/v25/22-0667.html . Sample-split DRO: Hu and Chen, JMLR25(246):1–85,2024, https://jmlr.org/papers/v25/23-0888.html . Package-specific method documentation remains the authority for each native estimator.
