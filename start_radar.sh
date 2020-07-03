#!/bin/bash

/usr/bin/pkill -9 -f remote_server.py
source $HOME/.profile

NOW=`date +'%Y%m%d %H:%M:%S'`

/usr/bin/nohup /usr/bin/python3 $BOREALISPATH/scheduler/remote_server.py --scd-dir=/home/radar/borealis_schedules --emails-filepath=/home/radar/borealis_schedules/emails.txt >/home/radar/logs/scd.out 2>&1 &

retVal=$?
if [[ $retVal -ne 0 ]]; then
	echo "${NOW} START: Could not start radar." | tee -a /data/borealis_logs/start_stop.log
else
	echo "${NOW} START: Radar processes started." | tee -a /data/borealis_logs/start_stop.log
fi
