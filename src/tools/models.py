"""Build the candidate model pipelines.

Feature selection is embedded in the estimators themselves (L1/
ElasticNet sparsity, tree impurity/gain, ...) rather than a separate
selection step, so each model does its own feature weighting across
the 515 mixed-family features. Every estimator must support
predict_proba, since tools/cv.py scores roc_auc from it.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RANDOM_STATE


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
    "scaler + classifier, fit as a unit inside each CV fold"
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", get_estimator(model_name)),
        ]
    )
