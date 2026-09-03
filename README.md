# RetentionEdge

Causal uplift retention targeting on a randomized email-campaign dataset —
proving that predicting who is likely to convert and knowing who to target
with a retention offer are two different problems, backed by a measured
number rather than a qualitative claim.

## Key result

> Uplift-based targeting recovered **709% more incremental conversions**
> than naive probability-ranked targeting, under an identical 20% budget
> (Qini coefficient **0.81** for the uplift model vs. **0.06** for the naive
> ranking). The result holds across budget sizes from 5% to 30%.

Full derivation and per-stage numbers: [`plan.md`](plan.md).
Resume-ready summary: [`resume_bullet.md`](resume_bullet.md).

## Overview

Standard churn-prediction projects stop at "predict who leaves." This
project goes one step further: it shows that ranking customers by predicted
conversion probability — the common approach — wastes budget on customers
who would have converted regardless of treatment, and under-targets the
customers a retention offer would actually persuade. The proof uses a
dataset where treatment was genuinely randomized, so the causal claims are
defensible rather than inferred from an observational proxy.

## Dataset

Kevin Hillstrom's MineThatData e-mail campaign dataset: 64,000 customers
randomly assigned to a men's email, a women's email, or no email, with
visit, conversion, and spend recorded after the campaign. Loaded via
`scikit-uplift` (`sklift.datasets.fetch_hillstrom`).

| segment       | rows   |
| ------------- | ------ |
| No E-Mail     | 21,306 |
| Mens E-Mail   | 21,307 |
| Womens E-Mail | 21,387 |

This project uses the women's-email arm vs. no-email as a clean binary
treatment (42,693 rows). See [`src/retentionedge/data/load.py`](src/retentionedge/data/load.py).

## Methodology

1. **Data acquisition** — load and cache the dataset, filter to a two-arm treatment/control split.
2. **Randomization check** — confirm treatment assignment was genuinely balanced across features (Welch's t-test / chi-square + standardized mean difference).
3. **Classical ML baseline** — Logistic Regression, Random Forest, and XGBoost predicting conversion probability, ignoring treatment status.
4. **Naive targeting baseline** — rank customers by predicted conversion probability, the policy this project disproves.
5. **Causal uplift modeling** — T-learner and X-learner CATE (Conditional Average Treatment Effect) estimation, implemented by hand.
6. **Uplift evaluation** — Qini curve and Qini coefficient, the correct evaluation metric for uplift models.
7. **Budget-constrained comparison** — the centerpiece result: incremental conversions recovered by random, naive, and uplift targeting under an identical budget.
8. **Robustness check** — confirm the result holds across multiple budget sizes.

## Results

### Classical ML baseline

Stratified train/val/test split (25,615 / 8,539 / 8,539), ~0.73% positive rate. Tuned via `RandomizedSearchCV`, scored on average precision, threshold chosen by maximizing F1.

| model               | roc_auc   | precision | recall | f1    |
| ------------------- | --------- | --------- | ------ | ----- |
| Logistic Regression | 0.567     | 0.012     | 0.129  | 0.022 |
| Random Forest       | **0.578** | 0.018     | 0.081  | 0.030 |
| XGBoost             | 0.534     | 0.017     | 0.032  | 0.022 |

All three are weak predictors (barely above random). This is expected for
this dataset and is central to the project's thesis — see
[Why the lift is this large](#why-the-lift-is-this-large) below.

### Causal uplift modeling

Uplift quadrant fractions (X-learner CATE, control-arm baseline conversion probability):

| quadrant     | T-learner | X-learner |
| ------------ | --------- | --------- |
| persuadable  | 42.3%     | 43.3%     |
| sure_thing   | 26.7%     | 26.9%     |
| lost_cause   | 23.9%     | 23.7%     |
| sleeping_dog | 7.2%      | 6.0%      |

### Uplift evaluation (Qini)

| ranking                        | Qini coefficient |
| ------------------------------ | ---------------- |
| X-learner CATE                 | **0.813**        |
| T-learner CATE                 | 0.757            |
| Naive classifier's probability | 0.064            |

### Budget-constrained comparison (20% budget, 8,538 customers)

| policy                        | incremental conversions | incremental revenue |
| ----------------------------- | ----------------------- | ------------------- |
| Random                        | 13.66                   | $878.80             |
| Naive (predicted probability) | 38.21                   | $5,497.34           |
| Uplift (X-learner CATE)       | **309.22**              | **$36,123.67**      |

### Robustness across budget sizes

| budget | naive incremental | uplift incremental | uplift over naive |
| ------ | ----------------- | ------------------ | ----------------- |
| 5%     | 32.9              | 247.0              | +650%             |
| 10%    | 27.7              | 281.6              | +918%             |
| 20%    | 38.2              | 309.2              | +709%             |
| 30%    | 67.2              | 330.3              | +392%             |

Uplift targeting wins at every budget tested, not only at the 20% cutoff.

## Why the lift is this large

Published uplift-modeling results typically report single-digit to
low-double-digit percentage lift. Here the naive baseline is unusually weak
(best classifier ROC-AUC 0.578), so its "top 20%" pick is barely different
from random at identifying who the email actually persuades (Qini
coefficient 0.064, near zero) — it mostly selects customers who look likely
to buy for reasons unrelated to the email. The uplift models explicitly
optimize for treatment-effect heterogeneity instead of raw conversion
likelihood (Qini 0.76–0.81), so the gap between them is much larger here
than in datasets where the naive baseline is itself a reasonably strong
predictor. The mechanism (uplift beats naive) replicates the literature; the
magnitude is specific to this baseline's weakness and is reported as
measured, not adjusted toward the expected range.

## Tech stack

| category               | tools                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| Language / environment | Python 3.11+, `uv`                                                                                            |
| Data handling          | pandas, NumPy, PyArrow                                                                                        |
| Classical ML           | scikit-learn (Logistic Regression, Random Forest), XGBoost                                                    |
| Model selection        | `RandomizedSearchCV`, stratified train/val/test split                                                         |
| Explainability         | SHAP                                                                                                          |
| Causal inference       | hand-implemented T-learner and X-learner meta-learners, `scikit-uplift` (Qini metrics, cross-check T-learner) |
| Statistics             | Welch's t-test, chi-square test, standardized mean difference                                                 |
| Evaluation             | Qini curve / Qini coefficient (AUUC), budget-constrained incremental-conversion analysis                      |
| Visualization          | matplotlib, seaborn                                                                                           |
| Testing                | pytest (18 tests across data, features, models, evaluation)                                                   |
| Notebook / progress    | Jupyter, tqdm                                                                                                 |

## Project structure

```text
RetentionEdge/
  pyproject.toml
  README.md
  plan.md
  resume_bullet.md
  retention_uplift_project_spec.md
  src/retentionedge/
    data/          # dataset load + treatment/control filtering
    features/      # preprocessing pipeline, randomization balance check
    models/        # baseline classifiers, naive targeting, uplift meta-learners
    evaluation/    # Qini curve, budget-constrained comparison
  notebooks/
    report.ipynb   # the single narrative notebook
  reports/
    figures/       # generated plots
  tests/
```

## Installation

```bash
uv sync --extra dev
```

## Usage

Run the test suite:

```bash
uv run pytest
```

Run the full narrative notebook (loads data, trains all models, evaluates,
produces the headline result — takes roughly 3–5 minutes, tqdm progress on
the long-running steps):

```bash
uv run jupyter lab
# open notebooks/report.ipynb, Run All
```

## Testing

18 tests across four files, run against the real dataset (no mocking):

| file                 | covers                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `test_data.py`       | dataset shape/columns, two-arm treatment filter                                                 |
| `test_features.py`   | preprocessing pipeline output shape, randomization balance check                                |
| `test_models.py`     | stratified splitting, all three classifiers, T-/X-learner CATE, quadrant segmentation           |
| `test_evaluation.py` | incremental-conversion math, Qini coefficient, budget comparison, percentage-improvement helper |

## Limitations and future work

- **No confidence interval on the headline number yet** — the budget sweep
  confirms the result isn't a single-cutoff artifact, but a bootstrap over
  CATE estimates (≥50 resamples) would put a variance estimate on the 709%
  figure rather than a single point estimate.
- **Two meta-learners only** — an R-learner or doubly-robust estimator would
  be a useful third cross-check, at the cost of a heavier dependency
  (`causalml`) that doesn't install cleanly on Windows without a C++ build
  toolchain.
- **One treatment arm evaluated** — the men's-email arm is symmetric and
  untested; results are not yet confirmed to generalize across campaigns.
- **No deployment** — this is a research/analysis project evaluated on the
  repository, notebook, and this README, not a hosted service.

## License

MIT — see [`LICENSE`](LICENSE).
