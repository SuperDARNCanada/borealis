#!/bin/bash
source "/home/radar/.profile"
LOGFILE="/home/radar/logs/start_stop.log"

# Kill remote_server.py process
PID=$(pgrep -f remote_server.py) # Get PID of remote_server.py process
pkill -9 -f remote_server.py

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
fi

sleep 1
NOW=$(date +'%Y%m%d %H:%M:%S')
if ps -p $PID &> /dev/null; then	 # Check if remote_server.py process still running
	echo "${NOW} STOP: FAIL - could not kill remote_server.py process." | tee -a $LOGFILE
	exit 1
fi

if [[ -n $(atq) ]]; then		# Check if atq is not empty
	echo "${NOW} STOP: FAIL - could not clear atq. Radar processes still scheduled." | tee -a $LOGFILE
	exit 1
fi

if [[ $retVal -ne 0 ]]; then
	echo "${NOW} STOP: FAIL - could not kill Borealis screen." | tee -a $LOGFILE
	exit 1
fi

echo "${NOW} STOP: SUCCESS - radar processes stopped." | tee -a $LOGFILE
