#!/bin/bash
source "/home/radar/.profile"
NOW=$(date +'%Y-%m-%d %H:%M:%S')

# Remove all scheduled experiments from at queue
for i in $(atq | awk '{print $1}')
do
	atrm $i
done

# Check if Borealis screen is still running
retVal=0
if screen -ls | grep -q borealis; then
	# Kill Borealis processes
	screen -X -S borealis quit
	retVal=$?
else
	echo "${NOW} STOP_RADAR: FAIL - Radar not running, no Borealis screens found"
	exit 1
fi

if [[ $retVal -ne 0 ]]; then
	echo "${NOW} STOP_RADAR: FAIL - could not kill Borealis screen."
	exit 1
fi

if [[ -n $(atq) ]]; then			# Check if atq is not empty
	echo "${NOW} STOP_RADAR: FAIL - could not clear atq. Radar processes still scheduled."
	exit 1
fi

echo "${NOW} STOP_RADAR: SUCCESS"
