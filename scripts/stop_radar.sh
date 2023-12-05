#!/bin/bash
source "${HOME}/.profile"
LOGFILE="/data/borealis_logs/start_stop.log"

# Kill remote_server.py process
/usr/bin/pkill -9 -f remote_server.py

# Remove all scheduled experiments from at queue
for i in $(atq | awk '{print $1}')
do 
	atrm $i
done  

# Kill Borealis
screen -X -S borealis quit
retVal=$?

NOW=$(date +'%Y%m%d %H:%M:%S')
if [[ $retVal -ne 0 ]]; then
	echo "${NOW} STOP: Could not stop radar." | tee -a $LOGFILE
else
	echo "${NOW} STOP: Radar processes stopped." | tee -a $LOGFILE
fi
