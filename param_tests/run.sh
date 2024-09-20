#!/bin/bash

python3 run_params.py --data ../evaluation_tests/data/real/retail.arff --json json/retail_Tk.json
python3 run_params.py --data ../evaluation_tests/data/real/fert_vs_gdp.arff --json json/fert_Tk.json
python3 run_params.py --data ../evaluation_tests/data/real/flow.arff --json json/flow_Tk.json
python3 run_params.py --data ../evaluation_tests/data/real/occupancy.arff --json json/occupancy_Tk.json
python3 run_params.py --data ../evaluation_tests/data/example/concept_drift.arff --json json/cong_Tk.json