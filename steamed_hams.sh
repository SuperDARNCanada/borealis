#!/bin/bash

# PROJECT BOREALIS BOOTER
# 12-JUNE-2018
# Adam Lozinsky

# Title Header.

# ARGS:
# $1 : experiment module to run, ex. normalscan
# $2 : run-type, including release, debug, and python-profiling

echo ""
echo "Project Borealis Booter"
echo "v2.3-Alpha Season 1 Episode 3"
echo "-----------------------------------------------------------------------------------"

# These are the commands to in each window.
if [ "$2" = "release" ]; then
    command1="echo Initialize brian.py; python3 -O brian/brian.py; bash"
    command2="echo Initialize experiment_handler.py; sleep 0.001s; python3 experiment_handler/experiment_handler.py "$1" ; bash;"
    command3="echo Initialize radar_control.py; sleep 0.001s; python3 -O radar_control/radar_control.py; bash;"
    command4="echo Initialize data_write.py; sleep 0.001s; python3 -O data_write/data_write.py; bash;"
    command5="echo Initialize n200_driver; sleep 0.001s; source mode "$2"; n200_driver > n200_output.txt; read -p 'press enter' "
    command6="echo Initialize signal_processing; sleep 0.001s; source mode "$2"; signal_processing; bash;"
    command7="echo Initialize tid setting; sleep 0.001s; python3 -O usrp_drivers/n200/set_affinity.py; bash;"
elif [ "$2" = "python-profiling" ]; then  # uses source mode release for C code.
    command1="echo Initialize brian.py; python3 -O -m cProfile -o testing/python_testing/brian.cprof brian/brian.py; bash"
    command2="echo Initialize experiment_handler.py; sleep 0.001s; python3 -O -m cProfile -o testing/python_testing/experiment_handler.cprof experiment_handler/experiment_handler.py "$1" ; bash;"
    command3="echo Initialize radar_control.py; sleep 0.001s; python3 -O -m cProfile -o testing/python_testing/radar_control.cprof radar_control/radar_control.py; bash;"
    command4="echo Initialize data_write.py; sleep 0.001s; python3 -O -m cProfile -o testing/python_testing/data_write.cprof data_write/data_write.py; bash;"
    command5="echo Initialize n200_driver; sleep 0.001s; source mode release; n200_driver > n200_output.txt ; read -p 'press enter' "
    command6="echo Initialize signal_processing; sleep 0.001s; source mode release; signal_processing; bash;"
    command7="echo Initialize tid setting; sleep 0.001s; python3 -O usrp_drivers/n200/set_affinity.py; bash;" 
elif [ "$2" = "debug" ]; then    
    command1="echo Initialize brian.py; python3 brian/brian.py; bash"
    command2="echo Initialize experiment_handler.py; sleep 0.001s; python3 experiment_handler/experiment_handler.py "$1" ; bash"
    command3="echo Initialize radar_control.py; sleep 0.001s; python3 radar_control/radar_control.py; bash"
    command4="echo Initialize data_write.py; sleep 0.001s; python3 data_write/data_write.py; bash"
    command5="echo Initialize n200_driver; sleep 0.001s; source mode "$2" ; gdb -ex start n200_driver; bash"
    command6="echo Initialize signal_processing; sleep 0.001s; source mode "$2"; /usr/local/cuda/bin/cuda-gdb -ex start signal_processing; bash"
    command7="echo Initialize tid setting; sleep 0.001s; python3 usrp_drivers/n200/set_affinity.py; bash"
else
    echo "Unknown run type "$2", exiting without running borealis."
    exit
fi
# Modify terminator's config
sed -i.bak "s#COMMAND1#$command1#; s#COMMAND2#$command2#; s#COMMAND3#$command3#; s#COMMAND4#$command4#; s#COMMAND5#$command5#; s#COMMAND6#$command6#; s#COMMAND7#$command7#;" ~/.config/terminator/config
# Launch a terminator instance using the new layout
terminator -l Borealis
# Return the original config file
mv ~/.config/terminator/config.bak ~/.config/terminator/config
