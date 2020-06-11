#!/bin/bash

/usr/bin/pkill -9 -f remote_server.py
source $HOME/.profile
/usr/bin/nohup /usr/bin/python3 $BOREALISPATH/scheduler/remote_server.py --scd-dir=/home/radar/borealis_schedules --emails-filepath=/home/radar/borealis_schedules/emails.txt >/home/radar/logs/scd.out 2>&1 &
