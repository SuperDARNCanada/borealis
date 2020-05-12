#!/bin/bash

/usr/bin/pkill -9 -f remote_server.py
source /home/radar/.profile
for i in `atq | awk '{print $1}'`
do 
	atrm $i
done  

screen -X -S borealis quit

