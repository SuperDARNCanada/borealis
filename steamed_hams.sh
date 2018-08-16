#!/bin/bash

# PROJECT BOREALIS BOOTER
# 12-JUNE-2018
# Adam Lozinsky

# Title Header.
echo ""
echo "Project Borealis Booter"
echo "v2.3-Alpha Season 1 Episode 3"
echo "-----------------------------------------------------------------------------------"

# These are the commands to in each window.
if [ "$2" = "release" ]; then
    command1="echo Initialize brian.py; python -O brian/brian.py; bash"
    command2="echo Initialize experiment_handler.py; sleep 0.001s; python experiment_handler/experiment_handler.py "$1" ; bash;"
    command3="echo Initialize radar_control.py; sleep 0.001s; python -O radar_control/radar_control.py; bash;"
    command4="echo Initialize data_write.py; sleep 0.001s; python -O data_write/data_write.py; bash;"
    command5="echo Initialize n200_driver; sleep 0.001s; source mode "$2"; n200_driver > n200_output.txt; read -p 'press enter' "
    command6="echo Initialize signal_processing; sleep 0.001s; source mode "$2"; signal_processing; bash;"
    command7="echo Initialize tid setting; sleep 0.001s; python -O usrp_drivers/n200/set_affinity.py; bash;"
else
    command1="echo Initialize brian.py; python brian/brian.py; bash"
    command2="echo Initialize experiment_handler.py; sleep 0.001s; python experiment_handler/experiment_handler.py "$1" ; bash"
    command3="echo Initialize radar_control.py; sleep 0.001s; python radar_control/radar_control.py; bash"
    command4="echo Initialize data_write.py; sleep 0.001s; python data_write/data_write.py; bash"
    command5="echo Initialize n200_driver; sleep 0.001s; source mode "$2" ; gdb -ex start n200_driver; bash"
    command6="echo Initialize signal_processing; sleep 0.001s; source mode "$2"; /usr/local/cuda/bin/cuda-gdb -ex start signal_processing; bash"
    command7="echo Initialize tid setting; sleep 0.001s; python usrp_drivers/n200/set_affinity.py; bash"
fi
# Modify terminator's config
sed -i.bak "s#COMMAND1#$command1#; s#COMMAND2#$command2#; s#COMMAND3#$command3#; s#COMMAND4#$command4#; s#COMMAND5#$command5#; s#COMMAND6#$command6#; s#COMMAND7#$command7#;" ~/.config/terminator/config
# Launch a terminator instance using the new layout
terminator -l Borealis
# Return the original config file
mv ~/.config/terminator/config.bak ~/.config/terminator/config
