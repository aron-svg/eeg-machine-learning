"""Summarize, compare and persist cross-validation results."""

import json
import os

from config import DATA_OUTPUT_FOLDER, REFIT_METRIC, RESULTS_FILENAME
from logger_init import logger


def summarize_cv_results(cv_results: dict) -> dict:
    "per-fold score arrays -> {metric: {mean, std}}"
    return {
        metric: {"mean": float(scores.mean()), "std": float(scores.std())}
        for metric, scores in cv_results.items()
    }


def select_best_model(results_by_model: dict) -> tuple:
    """pick the model with the highest mean REFIT_METRIC

    results_by_model: {model_name: {metric: {mean, std}}} (summarized)
    returns (best_model_name, its summary dict)
    """
    best_model_name = max(
        results_by_model,
        key=lambda name: results_by_model[name][REFIT_METRIC]["mean"],
    )
    return best_model_name, results_by_model[best_model_name]


def format_results_for_output(all_target_results: dict) -> dict:
    """assemble the final nested results structure

    all_target_results: {target_name: {model_name: raw cv_results}}
    """
    output = {}
    for target_name, results_by_model in all_target_results.items():
        summarized = {
            model_name: summarize_cv_results(cv_results)
            for model_name, cv_results in results_by_model.items()
        }
        best_model_name, _ = select_best_model(summarized)
        output[target_name] = {
            "models": summarized,
            "best_model": best_model_name,
        }
    return output


def save_results(
    results: dict, output_folder: str = DATA_OUTPUT_FOLDER
) -> None:
    "write the formatted results as JSON in the output folder"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, RESULTS_FILENAME)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    logger.info(f"Results saved to {output_path}")
