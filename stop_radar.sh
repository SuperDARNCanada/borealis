#!/bin/bash

/usr/bin/pkill -9 -f remote_server.py
source /home/radar/.profile
for i in `atq | awk '{print $1}'`
do 
	atrm $i
done  

NOW=`date +'%Y%m%d %H:%M:%S'`

screen -X -S borealis quit | tee -a /data/borealis_logs/start_stop.log

retVal=$?
if [[ $retVal -ne 0 ]]; then
	echo "${NOW} STOP: Could not stop radar." | tee -a /data/borealis_logs/start_stop.log
else
	echo "${NOW} STOP: Radar processes stopped." | tee -a /data/borealis_logs/start_stop.log
fi
