"""Qini curve evaluation for uplift models.

Classification metrics (accuracy, F1) don't apply here: for any single
customer we only ever observe one outcome (treated OR control), never both,
so there's no "correct CATE" to score against directly. The Qini curve
sidesteps this by ranking customers by predicted uplift and tracking
cumulative *incremental* conversions (treated conversions minus a
control-scaled baseline) as you target more of the ranked list — the model
compares itself against what actually happened in the randomized groups,
not against a per-customer label it doesn't have.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklift.metrics import qini_auc_score, qini_curve


def qini_coefficient(y_true: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray) -> float:
    """Qini coefficient: area under the Qini curve, normalized. Higher is better."""
    return float(qini_auc_score(y_true, uplift_scores, treatment))


def plot_qini(y_true: np.ndarray, treatment: np.ndarray, scores_by_model: dict[str, np.ndarray]):
    """Overlay the Qini curve for each named model, plus the random-targeting diagonal.

    `scores_by_model` maps a label (e.g. "T-learner", "naive probability") to
    its uplift/ranking score array — higher score means "target this
    customer first."
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, scores in scores_by_model.items():
        x, y = qini_curve(y_true, scores, treatment)
        qini = qini_coefficient(y_true, scores, treatment)
        ax.plot(x, y, label=f"{label} (Qini={qini:.4f})")

    # random targeting: a straight line from (0, 0) to (n, total incremental conversions)
    x_random, y_random = qini_curve(y_true, np.random.default_rng(0).random(len(y_true)), treatment)
    ax.plot([x_random[0], x_random[-1]], [y_random[0], y_random[-1]], "k--", label="random targeting")

    ax.set_xlabel("number of customers targeted")
    ax.set_ylabel("cumulative incremental conversions")
    ax.set_title("Qini curve: uplift models vs. naive vs. random targeting")
    ax.legend()
    fig.tight_layout()
    return fig
