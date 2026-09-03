"""Hillstrom dataset acquisition.

Source: sklift.datasets.fetch_hillstrom (pulls from the public sklift-datasets
mirror on first call, caches under ~/scikit-learn-data by default). We add our
own parquet cache under data/raw/ so repeated runs don't hit the network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklift.datasets import fetch_hillstrom

RAW_CACHE = Path(__file__).resolve().parents[3] / "data" / "raw" / "hillstrom.parquet"

TREATMENT_COL = "segment"
CONTROL_LABEL = "No E-Mail"
TREATMENT_LABELS = {"womens": "Womens E-Mail", "mens": "Mens E-Mail"}


def load_hillstrom(*, use_cache: bool = True) -> pd.DataFrame:
    """Load the full 64,000-row Hillstrom dataset as one flat DataFrame.

    Columns: recency, history_segment, history, mens, womens, zip_code,
    newbie, channel (features) + segment (treatment arm) + visit,
    conversion, spend (outcomes).
    """
    if use_cache and RAW_CACHE.exists():
        return pd.read_parquet(RAW_CACHE)

    bunch = fetch_hillstrom(target_col="all")
    df = pd.concat([bunch.data, bunch.treatment, bunch.target], axis=1)

    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_CACHE)
    return df


def filter_two_arm(df: pd.DataFrame, treatment: str = "womens") -> pd.DataFrame:
    """Keep only the control arm and one campaign arm for a clean binary treatment.

    Default arm is "womens" (documented choice — arbitrary between the
    two campaigns, kept as a parameter since the men's arm is symmetric).
    """
    if treatment not in TREATMENT_LABELS:
        raise ValueError(f"treatment must be one of {list(TREATMENT_LABELS)}, got {treatment!r}")

    label = TREATMENT_LABELS[treatment]
    mask = df[TREATMENT_COL].isin([CONTROL_LABEL, label])
    out = df.loc[mask].copy()
    out["treated"] = (out[TREATMENT_COL] == label).astype(int)
    return out
