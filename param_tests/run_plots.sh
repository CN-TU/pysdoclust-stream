#!/bin/bash

# Define data and JSON folder paths
DATA_FOLDER="../evaluation_tests/data"
JSON_FOLDER="json/all_var"
RESULT_FOLDER="results/all_var"

python3 plot_results.py "$RESULT_FOLDER/fert/"
python3 plot_results.py "$RESULT_FOLDER/retail/"
python3 plot_results.py "$RESULT_FOLDER/cong/"
python3 plot_results.py "$RESULT_FOLDER/occupancy/"
python3 plot_results.py "$RESULT_FOLDER/flow/"