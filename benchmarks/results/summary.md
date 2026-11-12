### C-index (mean over outer test folds; best in bold)

| Dataset | folds | TexGISa | Cox | CoxTime | DeepSurv | DeepHit |
|---|---|---|---|---|---|---|
| METABRIC | 20 | **0.643** | 0.632 | 0.633 | 0.639 | 0.635 |
| SUPPORT | 20 | **0.615** | 0.569 | 0.605 | 0.613 | 0.607 |
| FLCHAIN | 20 | 0.795 | **0.796** | 0.792 | 0.793 | 0.791 |
| GBSG | 20 | **0.675** | 0.660 | 0.668 | 0.669 | 0.673 |

### Integrated Brier score (mean over outer test folds; best in bold)

| Dataset | folds | TexGISa | Cox | CoxTime | DeepSurv | DeepHit |
|---|---|---|---|---|---|---|
| METABRIC | 20 | 0.170 | **0.165** | 0.166 | **0.165** | 0.168 |
| SUPPORT | 20 | **0.191** | 0.208 | 0.193 | **0.191** | 0.221 |
| FLCHAIN | 20 | **0.095** | 0.096 | 0.106 | 0.099 | 0.120 |
| GBSG | 20 | **0.178** | 0.190 | 0.181 | 0.179 | 0.188 |

### TexGISa minus baseline, C-index (mean difference; folds where TexGISa is higher)

| Dataset | Cox | CoxTime | DeepSurv | DeepHit |
|---|---|---|---|---|
| METABRIC | +0.0106 (17/20) | +0.0100 (14/20) | +0.0037 (13/20) | +0.0078 (14/20) |
| SUPPORT | +0.0460 (20/20) | +0.0097 (19/20) | +0.0017 (11/20) | +0.0073 (17/20) |
| FLCHAIN | -0.0007 (9/20) | +0.0032 (17/20) | +0.0026 (16/20) | +0.0040 (18/20) |
| GBSG | +0.0145 (19/20) | +0.0072 (15/20) | +0.0055 (15/20) | +0.0022 (10/20) |

### Baselines with the libraries' default settings (C-index / IBS)

Folds where the library-default runs exist (outer seeds 31415 and 27182).

| Dataset | folds | TexGISa | Cox | Cox (library defaults) | DeepSurv | DeepSurv (library defaults) |
|---|---|---|---|---|---|---|
| METABRIC | 10 | 0.645 / 0.170 | 0.633 / 0.164 | 0.633 / 0.164 | 0.637 / 0.165 | 0.611 / 0.176 |
| SUPPORT | 10 | 0.614 / 0.191 | 0.569 / 0.207 | 0.569 / 0.207 | 0.613 / 0.191 | 0.614 / 0.191 |
| FLCHAIN | 10 | 0.796 / 0.095 | 0.796 / 0.096 | 0.782 / 0.103 | 0.793 / 0.099 | 0.793 / 0.099 |
| GBSG | 10 | 0.675 / 0.178 | 0.659 / 0.192 | 0.659 / 0.192 | 0.668 / 0.179 | 0.667 / 0.180 |
