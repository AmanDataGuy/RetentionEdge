# RetentionEdge — Classical ML + Causal Uplift Retention Targeting

This is the sole source of truth for building this project. It's Aman's one classical ML project for a campus placement that tests classical ML and math fundamentals heavily. It needs to be rigorous, defensible in an interview, and genuinely different from the generic "diabetes-prediction-XGBoost" clones that are everywhere. Follow the stages in order.

## Why this project, and why this dataset

Standard churn-prediction projects (predict who leaves, using XGBoost, stop there) are extremely common and low-differentiation. This project goes one step further: it proves that predicting churn risk and knowing who to target with a retention offer are two different problems, and that naive targeting (send offers to whoever looks most likely to churn or convert) wastes budget on people who were never going to respond to the offer either way.

Use the **Kevin Hillstrom MineThatData e-mail marketing dataset** (64,000 customers, publicly available, bundled with the `scikit-uplift` and `causalml` Python packages, or downloadable directly). This is the standard dataset for uplift modeling because the treatment was **actually randomized**: customers were randomly assigned to receive a men's email campaign, a women's email campaign, or no email, and their subsequent visit/conversion/spend was recorded. Real randomization matters, it's what makes causal claims from this dataset defensible, unlike churn datasets with a synthetically invented "treatment" column.

## Stage 1 — Setup and data acquisition

- Set up a clean Python project (uv or venv), pinned dependencies.
- Load the Hillstrom dataset via `scikit-uplift` (`sklift.datasets.fetch_hillstrom()`) or `causalml`'s bundled version. Confirm row count (~64,000), treatment column (`segment`: mens email / womens email / no email), outcome columns (`visit`, `conversion`, `spend`).
- For a clean binary treatment setup, filter to two segments: "no email" (control) and one campaign (e.g., "womens email") as treatment. Document this choice.

## Stage 2 — EDA and feature engineering

- Explore feature distributions (recency, history/spend, channel, zip code type, newbie flag, etc.).
- Check for class imbalance in the outcome (conversion is typically rare, a few percent), note this explicitly since it affects both classifier and uplift model choices.
- Feature engineering: encode categoricals, handle any skew in numeric features (e.g., log-transform spend history), create any interaction features that make domain sense.
- Confirm the randomization actually worked: check that treatment and control groups have statistically similar feature distributions (a simple balance check, e.g., compare means with a t-test or standardized mean difference per feature). This is a real statistics skill (hypothesis testing) and worth stating explicitly in the report, it's evidence you understand why randomization matters, not just how to call a library function.

## Stage 3 — Classical ML baseline (this covers the core knowledge-area checklist)

Build a baseline model predicting conversion probability **ignoring treatment status for now** (i.e., a standard supervised classification problem):
- Train/validation/test split, done properly (stratified given class imbalance).
- Compare at least three models: Logistic Regression, Random Forest, XGBoost.
- Proper preprocessing pipeline (scaling where needed, encoding, handled inside a pipeline object, not ad hoc).
- Cross-validation and hyperparameter tuning (GridSearchCV, RandomizedSearchCV, or Optuna) for at least the tree-based models.
- Evaluate with precision, recall, F1, ROC-AUC, and a confusion matrix at a chosen threshold, justify the threshold choice given the class imbalance.
- Report feature importance (native for RF/XGBoost, or SHAP values for a clearer, model-agnostic explanation).

This stage alone should already give you concrete, defensible answers for classical ML interview questions, real numbers, real comparisons, real reasoning about model choice.

## Stage 4 — The naive targeting baseline (set up the thing you're about to disprove)

- Using the best classifier from Stage 3, rank all treated customers by predicted conversion probability.
- Define a naive targeting policy: "if we could only afford to email the top N% of customers, target by highest predicted conversion probability."
- This is the baseline you'll compare the causal approach against. State clearly why this is naive: predicted conversion probability tells you who's likely to convert, not who converts *because* of the email. Some of those top-ranked customers would have converted anyway (the "sure things"), so emailing them wastes budget.

## Stage 5 — Causal uplift modeling

- Implement at least two meta-learner approaches for estimating CATE (Conditional Average Treatment Effect) per customer: **T-learner** (separate models for treatment and control groups) and **X-learner** (propensity-weighted, generally more robust under imbalanced treatment/control sizes). Use `causalml` or `scikit-uplift`'s implementations rather than writing meta-learners from scratch, unless you want the extra rigor of implementing one by hand to demonstrate you understand the mechanics, worth doing for at least the T-learner since it's simple (two models, subtract predictions).
- Estimate a propensity score model if group sizes are imbalanced (needed properly for the X-learner).
- Segment customers into the classic uplift quadrants: persuadables (respond only if treated), sure things (convert regardless), lost causes (never convert), sleeping dogs (convert less if treated), report what fraction of the customer base falls into each bucket per your model. This is a well-known framework in uplift modeling and shows you understand the concept, not just the code.

## Stage 6 — Evaluate the uplift model properly

Classification metrics (accuracy, F1) don't apply to uplift models since you never observe both potential outcomes for the same customer. Use the correct evaluation instead:
- **Qini curve** and **Qini coefficient** (or the closely related uplift curve and AUUC, area under the uplift curve).
- Plot the Qini curve for your uplift model against a random-targeting baseline curve, and against the naive probability-ranking baseline from Stage 4.

## Stage 7 — Prove the naive approach fails, with a real number

This is the centerpiece result, don't skip it or leave it qualitative:
- Under a fixed budget constraint (e.g., "we can only email the top 20% of customers"), compute the actual incremental conversions (or incremental revenue, using the `spend` column) recovered by: (a) random targeting, (b) naive probability-ranked targeting, (c) uplift-based targeting.
- Report the real percentage improvement of (c) over (b), and of (b) over (a). Expect something in the range the published literature reports (single-digit to low-double-digit percentage lift), don't force a number, report what you actually get, and explain it if it's smaller or larger than expected.
- This gives you the same kind of concrete, defensible metric you already have in your other projects (RetailGraph's 0/20 vector-only failure, Resolv's $2,120 leaked with the harness off), a real, quantified proof that the naive approach is measurably worse.

## Stage 8 — Robustness check (optional but strong if time allows)

- Test sensitivity of the result to budget size (does uplift targeting still win at 5%, 10%, 30% budgets, or only at one specific cutoff?).
- Check stability of the CATE estimates via bootstrapping or across a few random train/test splits, report variance, not just a single point estimate.

## Stage 9 — Packaging

- Clean, modular pipeline (`data/`, `features/`, `models/`, `evaluation/`), not a single sprawling notebook, mirrors the "modular pipeline" pattern already common among strong peer projects.
- One clear notebook or report that walks through the story: baseline classifier → naive targeting → uplift modeling → Qini evaluation → the budget-constrained comparison → the final number.
- README with the real headline number front and center (e.g., "Uplift-based targeting recovered X% more incremental conversions than naive probability-ranked targeting under an identical budget").
- **Do not deploy this live.** No hosting needed for a classical ML project like this, a clean GitHub repo with a good README and notebook is what gets evaluated, not a live demo link. This also avoids adding another hosting cost on top of the current AWS situation.

## Stage 10 — Resume bullet (draft once built, don't write this before the real numbers exist)

Something in the shape of: "Built a customer retention targeting system on a randomized-experiment dataset (64K customers): compared Logistic Regression, Random Forest, and XGBoost for conversion prediction, then applied causal uplift modeling (T-learner, X-learner CATE estimation) to prove naive probability-based targeting under-targets persuadable customers, recovering X% more incremental conversions under an identical budget, validated via Qini curve analysis." Fill in the real X once Stage 7 is done, never estimate or round up.
