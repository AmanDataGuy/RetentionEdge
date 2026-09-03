"""Classical ML baseline.

A plain supervised classifier for P(conversion | features), ignoring
treatment status. This is deliberately "just" a churn/conversion predictor —
it exists so the naive targeting policy (rank by this probability) can be
built, which the causal uplift model later proves wrong.

Three models compared: Logistic Regression (linear baseline), Random Forest
and XGBoost (tuned via RandomizedSearchCV). Conversion is rare (~0.9% here),
so:
- splits are stratified
- tuning is scored on average_precision (PR-AUC), not accuracy — accuracy on
  a 99%-negative class is meaningless
- the decision threshold is chosen by maximizing F1 on the PR curve instead
  of the default 0.5, which would predict "no conversion" for almost everyone
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from retentionedge.features.preprocessing import FEATURE_COLUMNS, build_preprocessor

TARGET_COL = "conversion"
RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Stratified train/val/test split (stratified on `conversion`, since it's rare)."""
    X, y = df[FEATURE_COLUMNS], df[TARGET_COL]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, stratify=y_train_val, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10, cv: int = 3) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    param_distributions = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [4, 6, 8, None],
        "clf__min_samples_leaf": [1, 5, 20],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="average_precision",
        random_state=RANDOM_STATE,
        n_jobs=1,  # ponytail: multiprocess workers (n_jobs=-1) crash in constrained sandboxes; single-process is slower but reliable everywhere
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10, cv: int = 3) -> Pipeline:
    # scale_pos_weight = ratio of negatives to positives, XGBoost's way of
    # handling class imbalance (equivalent in spirit to class_weight="balanced").
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    param_distributions = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 4, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="average_precision",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def _best_f1_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Pick the probability cutoff that maximizes F1, instead of a blind 0.5.

    With ~1% positives, 0.5 would classify almost nobody as "will convert" —
    this scans the actual precision/recall tradeoff and picks the best point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """precision, recall, F1, ROC-AUC, confusion matrix — at the F1-optimal threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = _best_f1_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def shap_feature_importance(model: Pipeline, X_sample: pd.DataFrame):
    """Model-agnostic SHAP values for the fitted classifier, on a feature sample."""
    import shap

    preprocess, clf = model.named_steps["preprocess"], model.named_steps["clf"]
    X_transformed = preprocess.transform(X_sample)
    explainer = shap.Explainer(clf, X_transformed, feature_names=preprocess.get_feature_names_out())
    return explainer(X_transformed)
