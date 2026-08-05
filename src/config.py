"""Store the general parameters for the application in this file."
This includes settings for database connections, API keys,
and other configuration options that can be easily modified without
changing the main codebase."""

import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_INPUT_FOLDER = os.path.join(ROOT_DIR, "data", "input")
DATA_OUTPUT_FOLDER = os.path.join(ROOT_DIR, "data", "output")

# --- Data loading ---
# NPZ_GLOB_PATTERN already supports multiple subject files for the
# future LOSO phase, even though load_subject_data() only accepts
# exactly one file for now.
FEATURE_KEY = "X"
LABEL_KEY = "y"
GROUP_KEY = "trial_index"
NPZ_GLOB_PATTERN = "*_features.npz"

# --- Targets ---
TARGET_NAMES = ["arousal", "valence", "dominance"]
TARGET_COLUMN_INDEX = {name: i for i, name in enumerate(TARGET_NAMES)}

# --- Binarization ---
# label >= BINARIZATION_THRESHOLD -> 1 ("high")
# label <  BINARIZATION_THRESHOLD -> 0 ("low")
BINARIZATION_THRESHOLD = 5.0
CLASS_LABELS = {0: "low", 1: "high"}

# --- Cross-validation ---
# N_SPLITS=3: with 24 trial-groups and dominance's class imbalance
# (39/148), StratifiedGroupKFold produces degenerate single-class
# test folds at 5 splits (roc_auc_score would raise on those folds).
# 3 splits keeps both classes present in every outer fold.
N_SPLITS = 3
INNER_CV_SPLITS = 3
RANDOM_STATE = 42
# each model's own GridSearchCV stays single-core: parallelism happens
# one process per (target, model) pair (see tools/cv.py), so nesting
# more cores here would oversubscribe the machine instead of helping.
N_JOBS_PER_MODEL = 1
SCORING = ["balanced_accuracy", "f1", "roc_auc"]
REFIT_METRIC = "balanced_accuracy"

# --- Models ---
# Full catalog of candidate models. Each (target, model) pair runs as
# its own process (see tools/cv.py), so pick ACTIVE_MODELS such that
# len(TARGET_NAMES) * len(ACTIVE_MODELS) stays comfortably under the
# machine's core count (e.g. 3 targets x 4 models = 12 workers, fine
# on a 16-core machine).
MODEL_PARAM_GRIDS = {
    "logistic_elasticnet": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0.2, 0.5, 0.8],
    },
    "random_forest": {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 5, 10],
    },
    "extra_trees": {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 5, 10],
    },
    "gradient_boosting": {
        "clf__n_estimators": [100, 300],
        "clf__learning_rate": [0.01, 0.1],
        "clf__max_depth": [2, 3],
    },
    "svm_rbf": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.001],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 9, 15],
        "clf__weights": ["uniform", "distance"],
    },
    "lda": {
        "clf__shrinkage": [None, "auto", 0.1, 0.5],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(64,), (128, 64)],
        "clf__alpha": [0.0001, 0.001, 0.01],
    },
}

# Which models actually run this time: pick any subset of the keys
# above (e.g. 4 of the 8) and re-run.
ACTIVE_MODELS = [
    "logistic_elasticnet",
    "random_forest",
    "svm_rbf",
    "extra_trees",
]

# --- Output ---
RESULTS_FILENAME = "classification_results.json"
