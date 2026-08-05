"""Turn continuous SAM-scale labels into binary classification targets."""

import numpy as np

from config import BINARIZATION_THRESHOLD, TARGET_COLUMN_INDEX


def binarize_targets(
    y: np.ndarray, threshold: float = BINARIZATION_THRESHOLD
) -> np.ndarray:
    "label >= threshold -> 1 (high), label < threshold -> 0 (low)"
    return (y >= threshold).astype(int)


def get_target_vector(y_bin: np.ndarray, target_name: str) -> np.ndarray:
    "select the binarized column for one target (arousal/valence/...)"
    column = TARGET_COLUMN_INDEX[target_name]
    return y_bin[:, column]
