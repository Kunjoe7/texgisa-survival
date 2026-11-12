# Scientific semantics

The current recipes retain actual calibrated log-rate TEXGI gradients, directional domain hypotheses and the474-gene importance membership prior. The q98 reference is fitting-derived. Structural smoothness and magnitude constraints are not a substitute for verified named clinical rules. METABRIC, GBSG and SUPPORT currently have structural priors only because anonymous feature mappings remain unverified.

RefineCDF transforms cumulative hazard at time-bin boundaries as alpha*H+beta*log1p(H), alpha>0,beta>=0; interpolation remains piecewise constant hazard. RefineRiskCal adds c0+c1*tanh(risk/scale) to log interval cumulative hazards. The added layers retain the independent risk score and point prediction of a fixed fitted base. Calibration data are reserved within training; no conformal/individual-calibration guarantee is asserted.

Harrell concordance evaluates the independent risk score; event MAE evaluates the auxiliary time head; curves are separate probability outputs. MAE is observed-event-only, not a censoring-adjusted error for all patients. Censoring-weighted Brier objectives remain part of training/inner selection even though the manuscript reports only concordance and MAE. All recorded metrics and failures remain in the evidence archive.

The current code was integrated from completed development campaigns. A release interface adds a fitted-object return hook without changing the training mathematics. Reproduction workers retain frozen experiment sources. Per-dataset selection of different own recipes is not described as one model. Historical metric-wise best envelopes are not deployable estimators.
