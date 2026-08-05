"""Store the general parameters for the application in this file."
 This includes settings for database connections, API keys, 
 and other configuration options that can be easily modified without 
 changing the main codebase."""


import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_INPUT_FOLDER = os.path.join(ROOT_DIR, "data/input")
DATA_OUTPUT_FOLDER = os.path.join(ROOT_DIR, "data/output")