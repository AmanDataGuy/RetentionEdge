"""Causal uplift modeling: T-learner and X-learner CATE estimation.

CATE = Conditional Average Treatment Effect: how much *more* likely is this
specific customer to convert *because* they were emailed, vs. not emailed.
This is the number naive probability-ranking can't see.

All functions here take an already-preprocessed numeric feature array `X`
(output of `features.preprocessing.build_preprocessor()`, fit once) plus a
0/1 `treatment` array and a 0/1 `y` (conversion) array. Using one shared
preprocessor for every sub-model, instead of each model fitting its own,
avoids the treated/control one-hot columns silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklift.models import TwoModels

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# T-learner: two outcome models (treated arm, control arm), CATE = difference
# ---------------------------------------------------------------------------


@dataclass
class TLearner:
    model_treated: object
    model_control: object

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.model_treated.predict_proba(X)[:, 1] - self.model_control.predict_proba(X)[:, 1]


def t_learner_by_hand(X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> TLearner:
    """Hand-rolled T-learner: fit two classifiers, subtract predicted probabilities."""
    base = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    model_treated = clone(base).fit(X[treatment == 1], y[treatment == 1])
    model_control = clone(base).fit(X[treatment == 0], y[treatment == 0])
    return TLearner(model_treated, model_control)


def t_learner_sklift(X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> TwoModels:
    """Same construction via sklift's TwoModels — cross-check for the hand-rolled version."""
    base_trmnt = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    base_ctrl = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    model = TwoModels(estimator_trmnt=base_trmnt, estimator_ctrl=base_ctrl, method="vanilla")
    model.fit(X, y, treatment)
    return model


# ---------------------------------------------------------------------------
# X-learner (Kunzel et al. 2019) — sklift has no XLearner, so implemented by
# hand: impute individual treatment effects using the T-learner's *other*
# arm model, fit a regressor on those imputed effects per arm, then combine
# the two with a propensity model.
# ---------------------------------------------------------------------------


@dataclass
class XLearner:
    tau_treated: object  # imputed-effect regressor fit on treated units
    tau_control: object  # imputed-effect regressor fit on control units
    propensity: object  # P(treatment=1 | X), weights the two effect estimates

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        g = self.propensity.predict_proba(X)[:, 1]
        return g * self.tau_control.predict(X) + (1 - g) * self.tau_treated.predict(X)


def x_learner_by_hand(X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> XLearner:
    t_learner = t_learner_by_hand(X, treatment, y)
    treated_mask, control_mask = treatment == 1, treatment == 0

    # Impute each unit's individual treatment effect using the model fit on
    # the *opposite* arm (the only counterfactual estimate we have for them).
    d_treated = y[treated_mask] - t_learner.model_control.predict_proba(X[treated_mask])[:, 1]
    d_control = t_learner.model_treated.predict_proba(X[control_mask])[:, 1] - y[control_mask]

    effect_base = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    tau_treated = clone(effect_base).fit(X[treated_mask], d_treated)
    tau_control = clone(effect_base).fit(X[control_mask], d_control)

    propensity = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X, treatment)
    return XLearner(tau_treated=tau_treated, tau_control=tau_control, propensity=propensity)


# ---------------------------------------------------------------------------
# Uplift quadrant segmentation
# ---------------------------------------------------------------------------


def segment_uplift_quadrants(
    cate: np.ndarray, baseline_proba: np.ndarray, cate_threshold: float | None = None, baseline_threshold: float | None = None
) -> pd.Series:
    """Classic uplift quadrants from CATE sign + predicted baseline (control-arm) conversion rate.

    - persuadable: CATE > threshold (converts only if treated)
    - sleeping_dog: CATE < -threshold (converts LESS if treated)
    - sure_thing: no meaningful effect, but high baseline conversion (converts regardless)
    - lost_cause: no meaningful effect, low baseline conversion (never converts)

    `cate_threshold` defaults to half a standard deviation of the CATE
    distribution — CATE estimates are noisy, so treating every nonzero value
    as a "real" effect (threshold=0) would put almost nobody in the
    no-effect quadrants. Half a std is a standard noise-floor heuristic;
    pass an explicit value to use a different bar for "meaningful effect."
    """
    if cate_threshold is None:
        cate_threshold = 0.5 * float(np.std(cate))
    if baseline_threshold is None:
        baseline_threshold = float(np.median(baseline_proba))

    quadrant = np.select(
        [cate > cate_threshold, cate < -cate_threshold, baseline_proba >= baseline_threshold],
        ["persuadable", "sleeping_dog", "sure_thing"],
        default="lost_cause",
    )
    return pd.Series(quadrant, name="uplift_quadrant")
