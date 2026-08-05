"""Orchestrate the intra-subject emotion classification pipeline."""

from config import MODEL_PARAM_GRIDS, REFIT_METRIC, TARGET_NAMES
from data_loader import load_subject_data
from logger_init import logger
from tools import (
    binarize_targets,
    format_results_for_output,
    run_cv_for_all_targets,
    save_results,
)

MODEL_NAMES = list(MODEL_PARAM_GRIDS)


def main_process() -> None:
    "run nested, group-aware CV for every target/model pair, in parallel"
    logger.info("Starting the machine learning process")

    X, y, groups = load_subject_data()
    y_bin = binarize_targets(y)

    logger.info(f"Running cross-validation for {TARGET_NAMES} x {MODEL_NAMES}")
    all_results = run_cv_for_all_targets(
        X, y_bin, groups, TARGET_NAMES, MODEL_NAMES
    )

    formatted_results = format_results_for_output(all_results)
    for target_name, target_result in formatted_results.items():
        best_model = target_result["best_model"]
        score = target_result["models"][best_model][REFIT_METRIC]["mean"]
        logger.info(
            f"{target_name}: best={best_model} {REFIT_METRIC}={score:.3f}"
        )

    save_results(formatted_results)
