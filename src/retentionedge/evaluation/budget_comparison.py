"""Budget-constrained targeting comparison — the centerpiece result.

Under a fixed budget ("we can only email the top N% of customers"), compares
three targeting policies:
- random: pick customers at random
- naive: rank by predicted conversion probability (the classical-ML baseline)
- uplift: rank by predicted CATE (the causal uplift model)

The metric that matters is *incremental* conversions, not raw conversion
rate: raw rate mixes in "sure things" who'd have converted with no email at
all, which is exactly the naive policy's blind spot.

Because treatment (email) was genuinely randomized in this dataset, we can
score any targeting policy retrospectively: rank the whole customer base by
the policy, take the top budget_pct as "who we'd choose to target," then
compare the *actual* observed conversion rate of the ones who happened to
be emailed vs. the ones who happened not to be, within that group.
Randomization means that difference is an unbiased treatment-effect
estimate for the group — even though the policy didn't know who was
emailed when it ranked them.
"""

from __future__ import annotations

import pandas as pd

RANDOM_STATE = 42


def incremental_conversions(df: pd.DataFrame, targeted_idx: pd.Index, treatment_col: str, outcome_col: str) -> float:
    """Incremental outcome recovered if the whole targeted group were emailed.

    = (observed outcome rate among the targeted-and-treated) minus
      (observed outcome rate among the targeted-and-control),
    scaled by the size of the *whole* targeted group (not just the treated
    half) — randomization makes the treated/control split within the
    targeted group two equivalent samples of the same population, so this
    rate difference applies to all of them, not only the ones who happened
    to be emailed historically.
    """
    targeted = df.loc[targeted_idx]
    treated = targeted.loc[targeted[treatment_col] == 1]
    control = targeted.loc[targeted[treatment_col] == 0]
    if len(treated) == 0 or len(control) == 0:
        return 0.0
    rate_diff = treated[outcome_col].mean() - control[outcome_col].mean()
    return float(rate_diff * len(targeted))


def compare_targeting_policies(
    df: pd.DataFrame,
    naive_scores: pd.Series,
    uplift_scores: pd.Series,
    budget_pct: float = 0.20,
    treatment_col: str = "treated",
    outcome_col: str = "conversion",
    revenue_col: str = "spend",
) -> pd.DataFrame:
    """Compare random / naive / uplift targeting at a fixed budget.

    `naive_scores` and `uplift_scores` are per-customer ranking scores
    (higher = target first), indexed the same as `df`, e.g. a classifier's
    predicted conversion probability and a CATE model's predicted uplift.
    """
    budget_n = max(1, int(len(df) * budget_pct))

    random_idx = df.sample(n=budget_n, random_state=RANDOM_STATE).index
    naive_idx = naive_scores.sort_values(ascending=False).index[:budget_n]
    uplift_idx = uplift_scores.sort_values(ascending=False).index[:budget_n]

    rows = []
    for policy, idx in [("random", random_idx), ("naive_probability", naive_idx), ("uplift", uplift_idx)]:
        rows.append(
            {
                "policy": policy,
                "budget_n": budget_n,
                "incremental_conversions": incremental_conversions(df, idx, treatment_col, outcome_col),
                "incremental_revenue": incremental_conversions(df, idx, treatment_col, revenue_col),
            }
        )
    return pd.DataFrame(rows)


def pct_improvement(a: float, b: float) -> float:
    """Percentage improvement of `a` over baseline `b`."""
    if b == 0:
        return float("inf") if a > 0 else 0.0
    return (a - b) / abs(b) * 100
