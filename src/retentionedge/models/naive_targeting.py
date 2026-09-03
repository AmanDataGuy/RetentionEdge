"""Naive targeting baseline.

The policy this whole project sets out to disprove: "rank customers by
predicted conversion probability, email the top N%." This is naive because
predicted P(conversion) tells you who's *likely to convert*, not who
converts *because of the email* — some top-ranked customers were always
going to convert (sure things), so emailing them wastes budget. The
budget-constrained comparison quantifies exactly how much budget that wastes.
"""

from __future__ import annotations

import pandas as pd


def rank_by_predicted_probability(model, df_treated: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Predicted conversion probability per customer, sorted descending."""
    proba = model.predict_proba(df_treated[feature_cols])[:, 1]
    return pd.Series(proba, index=df_treated.index, name="predicted_proba").sort_values(ascending=False)


def naive_top_n_policy(ranked: pd.Series, n_pct: float) -> pd.Index:
    """Index of the top n_pct fraction of customers by predicted probability."""
    cutoff = max(1, int(len(ranked) * n_pct))
    return ranked.index[:cutoff]
