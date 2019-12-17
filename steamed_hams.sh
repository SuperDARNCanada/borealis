#!/bin/bash

# PROJECT BOREALIS BOOTER
# 12-JUNE-2018
# Adam Lozinsky

# Title Header.

# ARGS:
# $1 : experiment module to run, ex. normalscan
# $2 : run-type, including release, debug, and python-profiling
# $3 : scheduling mode type for this time period, ex. common, special, discretionary

echo ""
echo "Project Borealis Booter"
echo "v2.3-Alpha Season 1 Episode 3"
echo "-----------------------------------------------------------------------------------"

rm -r /dev/shm/*
screen -X -S borealis quit
sleep 1

# These are the commands to in each window.
if [ "$3" = "release" ]; then
    start_brian="python3 -O brian/brian.py; bash"
    start_exphan="sleep 0.001s; python3 -O experiment_handler/experiment_handler.py "$1" "$2" ; bash;"
    start_radctrl="sleep 0.001s; python3 -O radar_control/radar_control.py; bash;"
    start_datawrite="sleep 0.001s;python3 -O data_write/data_write.py --file-type=hdf5 --enable-raw-acfs --enable-bfiq --enable-antenna-iq; bash;"
    start_usrp_driver="sleep 0.001s; source mode "$3"; usrp_driver > usrp_output.txt; bash"
    start_dsp="sleep 0.001s; source mode "$3"; signal_processing; bash;"
    start_rt="sleep 0.001s; source borealisrt_env/bin/activate; python3 -O realtime/realtime.py; bash;"
elif [ "$2" = "python-profiling" ]; then  # uses source mode release for C code.
    start_brian="python3 -O -m cProfile -o testing/python_testing/brian.cprof brian/brian.py; bash"
    start_exphan="sleep 0.001s; python3 -O -m cProfile -o testing/python_testing/experiment_handler.cprof experiment_handler/experiment_handler.py "$1" "$2" ; bash;"
    start_radctrl="sleep 0.001s; python3 -O -m cProfile -o testing/python_testing/radar_control.cprof radar_control/radar_control.py; bash;"
    start_datawrite="sleep 0.001s; python3 -O -m cProfile -o testing/python_testing/data_write.cprof data_write/data_write.py; bash;"
    start_usrp_driver="sleep 0.001s; source mode release; usrp_driver > usrp_output.txt ; read -p 'press enter' "
    start_dsp="sleep 0.001s; source mode release; signal_processing; bash;"
    start_rt="sleep 0.001s; source borealisrt_env/bin/activate; python3 -O realtime/realtime.py; bash;"
elif [ "$3" = "debug" ] || [ "$3" = "engineeringdebug" ]; then
    start_brian="python3 brian/brian.py; bash"
    start_exphan="sleep 0.001s; python3 experiment_handler/experiment_handler.py "$1" "$2" ; bash"
    start_radctrl="sleep 0.001s; python3 radar_control/radar_control.py; bash"
#    start_datawrite="sleep 0.001s; python3 data_write/data_write.py --enable-bfiq --enable-pre-bfiq --enable-tx --enable-raw-rf; bash"
    start_datawrite="sleep 0.001s; python3 data_write/data_write.py --enable-antenna-iq --enable-raw-rf --enable-raw-acfs; bash"
    start_usrp_driver="sleep 0.001s; source mode "$3" ; gdb -ex start usrp_driver 2>usrp_output.txt; bash"
#    start_dsp="sleep 0.001s; source mode "$2"; /usr/local/cuda/bin/cuda-gdb -ex start signal_processing; bash"
    start_dsp="sleep 0.001s; source mode release; signal_processing; bash;"
    start_rt="sleep 0.001s; source borealisrt_env/bin/activate; python3 -O realtime/realtime.py; bash;"
else
    echo "Mode '$3' is unknown, exiting without running Borealis"
    exit -1
fi

# Modify screen rc file
sed -i.bak "s#START_BRIAN#$start_brian#; \
            s#START_EXPHAN#$start_exphan#; \
            s#START_RADCTRL#$start_radctrl#; \
            s#START_DATAWRITE#$start_datawrite#; \
            s#START_USRP_DRIVER#$start_usrp_driver#; \
            s#START_DSP#$start_dsp#; \
            s#START_RT#$start_rt#;" $BOREALISPATH/borealisscreenrc

# Launch a detached screen with editted layout.
screen -S borealis -c $BOREALISPATH/borealisscreenrc
# Return the original config file
mv $BOREALISPATH/borealisscreenrc.bak $BOREALISPATH/borealisscreenrc
