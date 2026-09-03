"""Feature preprocessing pipeline.

Turns the raw Hillstrom columns into model-ready numeric arrays:
- categorical columns -> one-hot encoded
- `history` (dollar spend, right-skewed) -> log1p then scaled
- remaining numeric columns -> scaled

Everything lives inside one sklearn ColumnTransformer so downstream code
(baseline models, uplift models) just does `build_preprocessor()` and
`.fit_transform()` instead of hand-rolling encoding logic per model.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["zip_code", "channel", "history_segment"]
NUMERIC_FEATURES = ["recency", "mens", "womens", "newbie"]
LOG_FEATURE = ["history"]  # dollar spend is right-skewed, log-transform before scaling

# The full feature set fed into every model in this project. Deliberately
# excludes `segment`/`treated` (that's the treatment assignment, not a
# predictive feature) and the outcome columns (`visit`, `conversion`, `spend`).
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + LOG_FEATURE


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: one-hot + scale + log-transform, ready to drop into a Pipeline."""
    log_then_scale = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("log_history", log_then_scale, LOG_FEATURE),
        ]
    )
