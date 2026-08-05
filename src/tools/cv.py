"""Nested, group-aware cross-validation.

Windows from the same trial are correlated, so every split (outer
evaluation and inner hyperparameter search) is grouped by trial_index
via StratifiedGroupKFold. Scaling and model fitting happen inside
each outer training fold only, avoiding leakage into the test fold.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from config import (
    INNER_CV_SPLITS,
    MODEL_PARAM_GRIDS,
    N_SPLITS,
    RANDOM_STATE,
    REFIT_METRIC,
    SCORING,
)
from tools.models import build_pipeline


def make_group_cv(n_splits: int) -> StratifiedGroupKFold:
    "grouped + stratified splitter, shared by the outer and inner CV"
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )


def _score_fold(y_true, y_pred, y_proba) -> dict:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def run_nested_cv(
    X: np.ndarray, y_target: np.ndarray, groups: np.ndarray, model_name: str
) -> dict:
    "outer CV for the performance estimate, inner CV for tuning"
    outer_cv = make_group_cv(N_SPLITS)
    inner_cv = make_group_cv(INNER_CV_SPLITS)
    param_grid = MODEL_PARAM_GRIDS[model_name]

    fold_scores = []
    for train_idx, test_idx in outer_cv.split(X, y_target, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        search = GridSearchCV(
            build_pipeline(model_name),
            param_grid,
            cv=inner_cv,
            scoring=REFIT_METRIC,
        )
        search.fit(X_train, y_train, groups=groups[train_idx])

        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]
        fold_scores.append(_score_fold(y_test, y_pred, y_proba))

    return {
        metric: np.array([fold[metric] for fold in fold_scores])
        for metric in SCORING
    }


def run_cv_for_target(
    X: np.ndarray, y_target: np.ndarray, groups: np.ndarray, model_names: list
) -> dict:
    "run nested CV for every candidate model on one target"
    return {
        model_name: run_nested_cv(X, y_target, groups, model_name)
        for model_name in model_names
    }
