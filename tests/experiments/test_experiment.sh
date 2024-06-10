#!/bin/bash
# Copyright 2023 SuperDARN Canada, University of Saskatchewan
# Author: Theodore Kolkman
#
# Test script to run experiment_unittests.py for a single experiment for all
# 5 sites.
#
# Dependencies:
#       - borealis_env3.9 virtual environment has been sourced
#
# Parameter EXPERIMENT: experiment to be tested. Must exist within the Borealis
#                       experiments top level directory

if [[ $# -ne 1 ]]; then
    printf "Usage: ./test_experiment.sh EXPERIMENT\n"
    exit 1
fi

EXPERIMENT=$1

RADAR_IDS=("sas" "pgr" "inv" "cly" "rkn")

printf "Testing $EXPERIMENT at all sites.\n"
pwd

for site in ${RADAR_IDS[*]}; do
    printf "\n$site:\n"
    python3 $BOREALISPATH/tests/experiments/experiment_unittests.py \
            --site_id $site \
            --experiment $EXPERIMENT
done
