#!/bin/bash
source "/home/radar/.profile"
source "${BOREALISPATH}/borealis_env${PYTHON_VERSION}/bin/activate"
LOGFILE="/home/radar/logs/start_stop.log"

# Stop current remote_server.py process
/usr/bin/pkill -9 -f remote_server.py

# Start new remote_server.py process
nohup python3 $BOREALISPATH/scheduler/remote_server.py \
		--scd-dir=/home/radar/borealis_schedules \
		>> /home/radar/logs/scd.out 2>&1 &

PID=$!	# Get pid of remote_server.py process
sleep 1

NOW=$(date +'%Y%m%d %H:%M:%S')
if ! ps -p $PID &> /dev/null; then	 # Check if remote_server.py process still running
	echo "${NOW} START: FAIL - remote_server.py failed to start." | tee -a $LOGFILE
	exit 1
fi

if [[ -z $(atq) ]]; then		# Check if atq is empty
	echo "${NOW} START: FAIL - atq is empty. No radar processes scheduled." | tee -a $LOGFILE
	exit 1
fi

echo "${NOW} START: SUCCESS - radar processes scheduled." | tee -a $LOGFILE
