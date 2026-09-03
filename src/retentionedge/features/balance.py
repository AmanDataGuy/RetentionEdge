"""Randomization balance check.

The Hillstrom dataset's causal claims only hold if treatment was actually
randomized. This checks that: for every feature, the treatment and control
groups should look statistically the same *before* any email was sent.

- Numeric features: Welch's t-test (means) + standardized mean difference
  (SMD) — SMD is the effect-size version, robust to the huge sample size
  making even tiny differences "significant" by p-value alone.
- Categorical features: chi-square test of independence (SMD doesn't apply
  to categories, only to numeric distributions).

A feature is flagged if |SMD| > 0.1 (the standard causal-inference rule of
thumb for "meaningful imbalance") or p < 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SMD_THRESHOLD = 0.1
P_VALUE_THRESHOLD = 0.05


def _standardized_mean_diff(treated: pd.Series, control: pd.Series) -> float:
    pooled_std = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_std)


def check_randomization_balance(
    df: pd.DataFrame, treatment_col: str, feature_cols: list[str]
) -> pd.DataFrame:
    """One row per feature: test used, statistic, p-value, SMD (numeric only), flag."""
    treated = df.loc[df[treatment_col] == 1]
    control = df.loc[df[treatment_col] == 0]

    rows = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            statistic, p_value = stats.ttest_ind(treated[col], control[col], equal_var=False)
            smd = _standardized_mean_diff(treated[col], control[col])
            test = "welch_t"
        else:
            contingency = pd.crosstab(df[col], df[treatment_col])
            statistic, p_value, _, _ = stats.chi2_contingency(contingency)
            smd = np.nan
            test = "chi2"

        flag = (not np.isnan(smd) and abs(smd) > SMD_THRESHOLD) or p_value < P_VALUE_THRESHOLD
        rows.append(
            {
                "feature": col,
                "test": test,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "smd": smd,
                "flag": flag,
            }
        )

    return pd.DataFrame(rows)
