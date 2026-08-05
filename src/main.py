import os

from config import DATA_INPUT_FOLDER
from logger_init import logger
from machine_learning import main_process


def _check_input():
    if not os.listdir(DATA_INPUT_FOLDER):
        logger.error("Input folder is empty")
        raise ValueError("Input folder is empty")


if __name__ == "__main__":
    logger.info("Starting the main process")
    _check_input()

    # Start the machine learning process
    main_process()
