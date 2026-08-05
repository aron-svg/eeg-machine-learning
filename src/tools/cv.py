"""Nested, group-aware cross-validation.

Windows from the same trial are correlated, so every split (outer
evaluation and inner hyperparameter search) is grouped by trial_index
via StratifiedGroupKFold. Scaling and model fitting happen inside
each outer training fold only, avoiding leakage into the test fold.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from config import (
    INNER_CV_SPLITS,
    MODEL_PARAM_GRIDS,
    N_JOBS_PER_MODEL,
    N_SPLITS,
    RANDOM_STATE,
    REFIT_METRIC,
    SCORING,
)
from tools.models import build_pipeline
from tools.preprocessing import get_target_vector


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
            n_jobs=N_JOBS_PER_MODEL,
        )
        search.fit(X_train, y_train, groups=groups[train_idx])

        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]
        fold_scores.append(_score_fold(y_test, y_pred, y_proba))

    return {
        metric: np.array([fold[metric] for fold in fold_scores])
        for metric in SCORING
    }


def run_cv_for_all_targets(
    X: np.ndarray,
    y_bin: np.ndarray,
    groups: np.ndarray,
    target_names: list,
    model_names: list,
) -> dict:
    """run nested CV for every (target, model) pair in parallel

    Targets and models are independent of each other, so every pair
    gets its own worker instead of looping targets sequentially.
    """
    pairs = [
        (target_name, model_name)
        for target_name in target_names
        for model_name in model_names
    ]
    results = Parallel(n_jobs=len(pairs))(
        delayed(run_nested_cv)(
            X, get_target_vector(y_bin, target_name), groups, model_name
        )
        for target_name, model_name in pairs
    )

    output = {target_name: {} for target_name in target_names}
    for (target_name, model_name), result in zip(pairs, results):
        output[target_name][model_name] = result
    return output
