"""Build the candidate model pipelines.

Feature selection is embedded in the estimators themselves (L1/
ElasticNet sparsity, Random Forest impurity-based splits) rather than
a separate selection step, so the model does its own feature
weighting across the 515 mixed-family features.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


MODEL_REGISTRY = {
    "logistic_elasticnet": _build_logistic_elasticnet,
    "random_forest": _build_random_forest,
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
