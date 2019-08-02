#!/bin/bash

pkill -9 -f remote_server.py
nohup python3 $BOREALISPATH/scheduler/remote_server.py --scd-dir=/home/radar/borealis_schedules/borealis_schedules --emails-filepath=/home/radar/borealis_schedules/borealis_schedules/emails.txt >/dev/null &
