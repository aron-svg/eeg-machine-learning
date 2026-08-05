"""Load precomputed EEG feature archives produced upstream."""

import glob
import os

import numpy as np

from config import (
    DATA_INPUT_FOLDER,
    FEATURE_KEY,
    GROUP_KEY,
    LABEL_KEY,
    NPZ_GLOB_PATTERN,
)
from logger_init import logger

REQUIRED_KEYS = (FEATURE_KEY, LABEL_KEY, GROUP_KEY)


def load_npz_file(path: str) -> dict:
    "load one .npz archive and check that the required keys exist"
    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.keys()}
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return data


def load_subject_data(
    input_folder: str = DATA_INPUT_FOLDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """load the single-subject feature file for the intra-subject phase

    Multi-subject (LOSO) support will loop over every matched file and
    concatenate X/y/groups here, keeping subject_id around for the
    outer split. Not implemented yet: exactly one file is required.
    """
    pattern = os.path.join(input_folder, NPZ_GLOB_PATTERN)
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 file matching {pattern}, "
            f"found {len(matches)}: {matches}"
        )

    path = matches[0]
    logger.info(f"Loading feature file: {path}")
    data = load_npz_file(path)

    X = data[FEATURE_KEY]
    y = data[LABEL_KEY]
    groups = data[GROUP_KEY]
    logger.info(f"Loaded X{X.shape}, y{y.shape}, {len(set(groups))} groups")
    return X, y, groups
