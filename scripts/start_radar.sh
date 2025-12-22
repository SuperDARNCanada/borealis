#!/bin/bash
source "/home/radar/.profile"
NOW=$(date +'%Y-%m-%d %H:%M:%S')

# Start new make_atq.py process
${BOREALISPATH}/borealis_env${PYTHON_VERSION}/bin/python3 \
  ${BOREALISPATH}/scheduler/make_atq.py \
  /home/radar/borealis_schedules \
  2>&1

if [[ -z $(atq) ]]; then		# Check if atq is empty
	echo "${NOW} START_RADAR: FAIL - atq is empty. No radar processes scheduled."
	exit 1
fi

echo "${NOW} START_RADAR: SUCCESS"
