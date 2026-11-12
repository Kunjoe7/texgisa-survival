---
name: texgisa-prior-finder
description: Find literature-sourced attribution priors for a right-censored survival dataset and encode them for texgisa-survival. Use when a user wants expert or domain priors (directional feature rules or important-feature lists) for TexGISa and has a data dictionary but no clinician list yet.
---

# Finding attribution priors for TexGISa

This skill turns published knowledge about a dataset's features into the two prior
types that `texgisa-survival` trains with:

* **Directional rules**: "higher values of feature *f* raise (or lower) the event hazard",
  optionally only for rows meeting a condition. TexGISa penalizes time-resolved TEXGI
  attributions whose sign contradicts the rule.
* **Important-feature lists**: a set of features expected to matter. The TEXGI importance
  penalty (Qin et al., ICDM 2025, Eq. 18) promotes listed features whose attribution norm falls
  below the feature average and shrinks the unlisted ones. No sign is implied.

Both are soft constraints. They should encode what the literature supports for *this* outcome,
and nothing more. This is the procedure used for the benchmark priors shipped in
`texgisa_survival/priors.json` and `texgisa_survival/data/gene_prior.json`.

## Inputs to collect first

1. Column names, units and coding of every feature (categorical levels, reference level, scale).
2. The event definition and the time axis (calendar time, tenure, a monetary axis, ...).
3. The population and how the data were collected (cohort, trial, registry, business data).

If a column's meaning cannot be established (for example anonymous columns `x0`, `x1`, ...),
**do not write a rule for it**. Such datasets get structural TEXGI priors only
(`dataset="custom"` in `SurvivalRecipe`). Guessing a mapping would put an unsupported
assumption into training.

## Step 1: Find sources, most specific first

Search in this order and stop adding sources once a direction is well supported:

1. The dataset's own documentation (for example the R `survival` package help pages for
   `flchain` or `nwtco`, the provider's repository or data card).
2. The original study or trial report that produced the data.
3. Clinical or domain guidelines and systematic reviews for the same outcome.
4. Curated knowledge bases for high-dimensional features (for example a cancer-gene catalogue
   for gene-expression columns).

For every candidate rule, keep the URL or citation, the sentence or table that supports it, and
the date accessed.

## Step 2: Decide the prior type

* Write a **directional rule** only when a source states the direction for the same kind of
  outcome (for example "higher stage, higher relapse risk"). If the outcome differs (for example
  a monetary axis instead of calendar time), mark the rule `exploratory` in its basis text.
* Write an **important-feature list** when sources support relevance but not a direction, or for
  many features at once (genes). Build it by intersecting the catalogue with the dataset's
  feature names; record the catalogue file and its hash.
* Never derive a prior from the evaluation data, from survival outcomes, or from any model's
  own importance ranking. That leaks information and makes later results unreliable.

## Step 3: Encode the priors

Directional rules for the `TexGISa` estimator:

```python
from texgisa_survival import TexGISa

model = TexGISa(lambda_expert=0.1, random_state=0)
# feature, relation, threshold ('mean', 'median' or a value in original units), sign
model.add_expert_rule("stage", ">", "median", sign="+1", weight=1.0)
model.add_expert_rule("satisfaction_level", ">=", "mean", sign="-1", weight=1.0)
model.fit(X_train, time_train, event_train)
```

Important-feature lists for `MySATrainer`:

```python
expert_rules = {"important_features": ["TP53", "PIK3CA", "..."]}
```

Record each dataset's priors in the same schema as `texgisa_survival/priors.json`:

```json
{
  "my_dataset": {
    "rules": [{"feature": "stage", "sign": 1, "condition": "above_training_median"}],
    "source": "https://...",
    "basis": "What the source says, and whether it is a literature hypothesis or clinician-confirmed."
  }
}
```

## Step 4: Hand the list to a domain expert

A literature-derived prior is a hypothesis. When an expert is available, show them every rule
with its source and ask them to confirm, remove or add rules. Mark the result in `basis`
(`literature hypothesis` or `clinician-confirmed`). A clinician's list replaces the literature
list for that dataset. For example, the TCGA gene list of the TexGISa ICDM 2025 study was
defined with a medical oncologist at MD Anderson Cancer Center.

## Checklist before training

- [ ] Every rule names a column that exists in the fitting data, with the stated coding.
- [ ] Every rule has a source and a basis sentence; none was inferred from the outcomes.
- [ ] Anonymous or ambiguous columns have no rule.
- [ ] Transfers to a different outcome axis are marked exploratory.
- [ ] The prior file is saved with the experiment so the run can be reproduced.
