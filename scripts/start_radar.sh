#!/bin/bash
source "${HOME}/.profile"
source "${BOREALISPATH}/borealis_env${PYTHON_VERSION}/bin/activate"
LOGFILE="/data/borealis_logs/start_stop.log"

# Stop current remote_server.py process
/usr/bin/pkill -9 -f remote_server.py

# Start new remote_server.py process
nohup python3 $BOREALISPATH/scheduler/remote_server.py \
		--scd-dir=/home/radar/borealis_schedules \
		--emails-filepath=/home/radar/borealis_schedules/emails.txt \
		>> /home/radar/logs/scd.out 2>&1 &

pid=$!	# Get pid of remote_server.py process
sleep 1

NOW=$(date +'%Y%m%d %H:%M:%S')
if ps -p $pid > /dev/null; then	 # Check if remote_server.py process still running
	echo "${NOW} START: Radar processes started." | tee -a $LOGFILE
else
	echo "${NOW} START: Could not start radar." | tee -a $LOGFILE
fi
