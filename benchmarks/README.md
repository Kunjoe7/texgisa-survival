# Historical and corrected benchmark records

The displayed table and `results/*.json` preserve the original v2.1.0
prediction-only runs. `legacy/` contains the frozen source for reproducing them.
The corrected `run_benchmark.py` uses split-local preprocessing and writes to
`corrected_results/`; no complete corrected search has been reported here.
See `../SCIENTIFIC_NOTES.md` for the method and protocol differences.

The first two outer seeds (31415, 27182) were reused to evaluate the locked v3
variant after an earlier variant. Seeds 16180 and 14142 form the later
confirmation run. All seeds reuse observations from the same four cohorts.

# Benchmark

This folder contains the benchmark reported in the texgisa-survival software paper:
the script that produces it, the exact search spaces, the per-fold results and the
summary tables.

| File | Content |
|---|---|
| `run_benchmark.py` | corrected split-local protocol; new results only |
| `configs.json` | the configurations searched for each method (54 each; 50 for Cox) |
| `results/<dataset>.json` | one entry per method and outer test fold: C-index, IBS, selected configuration |
| `results/summary.md` | the tables below (`python summarize.py --markdown results/summary.md`) |
| `check_refit.py` | refits a historical selected configuration with frozen original source |

## Protocol

* **Data.** METABRIC, SUPPORT and GBSG from `pycox.datasets`; FLCHAIN from
  `sksurv.datasets` (the cause-of-death column is dropped because it reveals the
  outcome; historically, missing creatinine values were set to the cohort median before folds).
* **Outer loop.** 5-fold cross-validation repeated with four seeds (31415, 27182,
  16180, 14142): 20 test folds per dataset. Historical features were standardized on the outer training part before inner splits.
* **Inner loop.** Each outer training part is split 80/20 (stratified by event).
  Every configuration is fit on the 80% and scored by Harrell's C-index on the 20%;
  the best configuration is refit on the whole outer training part and evaluated
  once on the outer test fold. Configurations are selected by inner C-index; the reused and confirmation outer seeds are identified above.
* **Equal budgets.** 54 configurations for every neural method and 50 for the
  penalised Cox model, drawn at random from the spaces in `configs.json`. The
  TexGISa space varies backbone (ResNet or MLP), width, weight averaging, input
  noise, input gate, batch size, Cox-loss weight, early-stopping criterion and
  learning rate. The baseline spaces vary width and depth, dropout, learning
  rate, weight decay and batch size, and for DeepHit the number of intervals,
  `alpha` and `sigma`. All neural networks train for up to 300 epochs with early
  stopping (patience 15) on a random 20% of their training data.
* **Metrics.** Harrell's C-index (lifelines) and the integrated Brier score
  (pycox `EvalSurv`, Kaplan-Meier censoring weights) on 100 time points spanning
  the test fold.

## Results (20 outer test folds per dataset)

C-index, mean over folds (higher is better):

| Dataset | TexGISa | Cox | CoxTime | DeepSurv | DeepHit |
|---|---|---|---|---|---|
| METABRIC | **0.643** | 0.632 | 0.633 | 0.639 | 0.635 |
| SUPPORT | **0.615** | 0.569 | 0.605 | 0.613 | 0.607 |
| FLCHAIN | 0.795 | **0.796** | 0.792 | 0.793 | 0.791 |
| GBSG | **0.675** | 0.660 | 0.668 | 0.669 | 0.673 |

Integrated Brier score, mean over folds (lower is better):

| Dataset | TexGISa | Cox | CoxTime | DeepSurv | DeepHit |
|---|---|---|---|---|---|
| METABRIC | 0.170 | **0.165** | 0.166 | **0.165** | 0.168 |
| SUPPORT | **0.191** | 0.208 | 0.193 | **0.191** | 0.221 |
| FLCHAIN | **0.095** | 0.096 | 0.106 | 0.099 | 0.120 |
| GBSG | **0.178** | 0.190 | 0.181 | 0.179 | 0.188 |

The differences between TexGISa and the strongest baseline on each dataset are
small (at most 0.004 C-index) and of the same size as the variation between folds;
`results/summary.md` lists the paired fold-by-fold differences.

## Baseline settings

`run_benchmark.py` changes two library defaults for the baselines, because with
the defaults some fits fail:

* pycox's Cox loss (DeepSurv) divides by the number of events in a mini-batch. A
  mini-batch without events gives NaN, and the network's weights become NaN.
  The script uses a loss that is 0 for such a batch.
* lifelines' Newton steps can diverge for the L1-penalised Cox model on FLCHAIN.
  The script uses step size 0.25 and up to 2000 steps.

`--library-defaults` runs the libraries unchanged. Those results are stored as
"Cox (library defaults)" and "DeepSurv (library defaults)" for outer seeds 31415
and 27182. With the defaults, DeepSurv's mean C-index on METABRIC falls from
0.637 to 0.611 and Cox's on FLCHAIN from 0.796 to 0.782 (same 10 folds). TexGISa
needs no such change.

## Reproducing

```bash
python -m pip install . pycox torchtuples lifelines scikit-survival
python legacy/run_benchmark.py metabric    # historical table protocol
python run_benchmark.py metabric           # corrected protocol; separate outputs
python legacy/run_benchmark.py flchain --methods Cox DeepSurv --seeds 31415 27182 --library-defaults
python summarize.py
python check_refit.py gbsg TexGISa 27182 3  # refit one stored fold (seconds)
```

A full hyperparameter search requires many model fits. The verification report
retains eight selected-configuration refits across all four datasets; their
C-index differences from the archived records were at most 0.00012. These refits
validate selected records, not the full inner searches. Thread counts and library
versions can affect numerical reproducibility. The new GBSG2 experiment is in
`expert_feedback.py` and `results/expert_feedback.json`; GBSG2 is a distinct cohort.
