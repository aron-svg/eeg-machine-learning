"""Build the candidate model pipelines.

Models whose estimator already does its own feature weighting (L1/
ElasticNet sparsity, tree impurity/gain, ...) skip explicit selection.
Models with none (svm_rbf, knn, lda, mlp) get a SelectKBest step added
ahead of them, see MODELS_WITHOUT_EMBEDDED_SELECTION below; its k is
tuned like any other hyperparameter (MODEL_PARAM_GRIDS' "select__k").
Every estimator must support predict_proba, since tools/cv.py scores
roc_auc from it.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RANDOM_STATE

MODELS_WITHOUT_EMBEDDED_SELECTION = {"svm_rbf", "knn", "lda", "mlp"}


def _build_logistic_elasticnet() -> LogisticRegression:
    # l1_ratio (set via MODEL_PARAM_GRIDS) selects the elasticnet mix;
    # sklearn >=1.8 infers this from l1_ratio without a penalty kwarg.
    return LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def _build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    )


def _build_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    )


def _build_gradient_boosting() -> GradientBoostingClassifier:
    # no class_weight support in sklearn; classes are rebalanced
    # upstream via nothing here, so expect weaker minority recall.
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def _build_svm_rbf() -> SVC:
    return SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _build_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier()


def _build_lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr")


def _build_mlp() -> MLPClassifier:
    return MLPClassifier(max_iter=2000, random_state=RANDOM_STATE)


MODEL_REGISTRY = {
    "logistic_elasticnet": _build_logistic_elasticnet,
    "random_forest": _build_random_forest,
    "extra_trees": _build_extra_trees,
    "gradient_boosting": _build_gradient_boosting,
    "svm_rbf": _build_svm_rbf,
    "knn": _build_knn,
    "lda": _build_lda,
    "mlp": _build_mlp,
}


def get_estimator(model_name: str):
    "instantiate a fresh, unfitted estimator for the given model name"
    return MODEL_REGISTRY[model_name]()


def build_pipeline(model_name: str) -> Pipeline:
    "scaler [+ SelectKBest] + classifier, fit as a unit per CV fold"
    steps = [("scaler", StandardScaler())]
    if model_name in MODELS_WITHOUT_EMBEDDED_SELECTION:
        steps.append(("select", SelectKBest(score_func=f_classif)))
    steps.append(("clf", get_estimator(model_name)))
    return Pipeline(steps)
