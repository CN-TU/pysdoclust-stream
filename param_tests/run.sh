#!/bin/bash

# Define data and JSON folder paths
DATA_FOLDER="../evaluation_tests/data"
JSON_FOLDER="json/all_var"

# Run the python scripts with the specified data and JSON configurations
python3 run_params.py --data "$DATA_FOLDER/real/retail.arff" --json "$JSON_FOLDER/retail.json"
python3 run_params.py --data "$DATA_FOLDER/real/fert_vs_gdp.arff" --json "$JSON_FOLDER/fert.json"
python3 run_params.py --data "$DATA_FOLDER/example/concept_drift.arff" --json "$JSON_FOLDER/cong.json"
python3 run_params.py --data "$DATA_FOLDER/real/flow.arff" --json "$JSON_FOLDER/flow.json"
python3 run_params.py --data "$DATA_FOLDER/real/occupancy.arff" --json "$JSON_FOLDER/occupancy.json"
