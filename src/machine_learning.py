"""Orchestrate the intra-subject emotion classification pipeline."""

from config import MODEL_PARAM_GRIDS, REFIT_METRIC, TARGET_NAMES
from data_loader import load_subject_data
from logger_init import logger
from tools import (
    binarize_targets,
    format_results_for_output,
    get_target_vector,
    run_cv_for_target,
    save_results,
)

MODEL_NAMES = list(MODEL_PARAM_GRIDS)


def main_process() -> None:
    "run nested, group-aware CV for each target and save the results"
    logger.info("Starting the machine learning process")

    X, y, groups = load_subject_data()
    y_bin = binarize_targets(y)

    all_results = {}
    for target_name in TARGET_NAMES:
        logger.info(f"Running cross-validation for target: {target_name}")
        y_target = get_target_vector(y_bin, target_name)
        all_results[target_name] = run_cv_for_target(
            X, y_target, groups, MODEL_NAMES
        )

    formatted_results = format_results_for_output(all_results)
    for target_name, target_result in formatted_results.items():
        best_model = target_result["best_model"]
        score = target_result["models"][best_model][REFIT_METRIC]["mean"]
        logger.info(
            f"{target_name}: best={best_model} {REFIT_METRIC}={score:.3f}"
        )

    save_results(formatted_results)
