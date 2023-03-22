#!/usr/bin/python3

"""
    Borealis Start up script
    ~~~~~~~~~~~~~~~~~~~~~~~~

    The Simpsons, Season 7 Episode 21

    :copyright: 2020 SuperDARN Canada
    :author: Keith Kotyk
"""
import argparse
import sys
import subprocess as sp
import os
import time

PYTHON_VERSION = os.environ['PYTHON_VERSION']


def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.

    :returns:   the usage message
    :rtype:     str
    """

    usage_message = """ steamed_hams.py [-h] experiment_module run_mode scheduling_mode_type

    Pass the module containing the experiment to steamed_hams as a required
    argument. The experiment handler will search for the module in the BOREALISPATH/experiments
    directory. It will retrieve the class from within the module (your experiment).

    Pass the mode to specify module run options and data outputs. Available modes are 
    release, debug, testdata, engineeringdebug, and pythonprofiling. 
    Release should be most commonly used. Note that testdata and engineeringdebug modes
    produce very large rawrf data files and will severely limit the rate of the system 
    (evident by low sequences per integration period). It is recommended to only run these 
    modes for short test periods due to the quantity of data produced.

    Pass in the scheduling mode type, in general common, discretionary, or special.

    """

    return usage_message


BOREALISSCREENRC = """####THIS FILE IS GENERATED BY STEAMED_HAMS.PY####

chdir $BOREALISPATH

scrollback 10000
layout autosave on
layout new borealis

caption always "%{{=ub kR}}%n %t %C:%s%a %=%l"

hardstatus alwayslastline
hardstatus string '%{{= kG}}[%{{G}}%H%? %1`%?%{{g}}][%= %{{= kw}}%-w%{{+b yk}} %n*%t%?(%u)%? %{{-}}%+w %=%{{g}}][%{{B}}%m/%d %{{W}}%C%A%{{g}}]'

# 256 colors
attrcolor b ".I"
termcapinfo xterm 'Co#256:AB=\\E[48;5;%dm:AF=\\E[38;5;%dm'
defbce on

# mouse tracking allows to switch region focus by clicking
defmousetrack on

#ctrl-arrow keys to navagate windows
bindkey ^[[1;5D focus left
bindkey ^[[1;5C focus right
bindkey ^[[1;5A focus up
bindkey ^[[1;5B focus down

#Realtime produces no real useful output at this time so we have it in a hidden window. It can
#still be switched to within screen if needed.
screen -t "Brian" bash -c "{START_BRIAN}"
split

split -v
focus right
screen -t "N200 Driver" bash -c "{START_USRP_DRIVER}"

split -v
focus right
screen -t "Signal Processing" bash -c "{START_DSP}"

focus down
screen -t "Data Write" bash -c "{START_DATAWRITE}"

split -v
focus right
screen -t "Experiment Handler" bash -c "{START_EXPHAN}"

split -v
focus right
screen -t "Radar Control" bash -c "{START_RADCTRL}"

split -v
focus right
screen -t "Realtime" bash -c "{START_RT}"

detach
"""


def steamed_hams_parser():
    """
    Creates the parser.

    :returns:   parser, the argument parser for steamed_hams.
    :rtype:     argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("experiment_module", help="The name of the module in the experiments directory "
                                                  "that contains your Experiment class, "
                                                  "e.g. 'normalscan'")
    parser.add_argument("run_mode", help="The mode to run, switches scons builds and some arguments to "
                                         "modules based on this mode. Commonly 'release'.")
    parser.add_argument("scheduling_mode_type", help="The type of scheduling time for this experiment "
                                                     "run, e.g. 'common', 'special', or 'discretionary'.")
    parser.add_argument("--kwargs_string", default='', 
                        help="String of keyword arguments for the experiment.")

    return parser


parser = steamed_hams_parser()
args = parser.parse_args()

if args.run_mode == "release":
    # python optimized, no debug for regular operations
    python_opts = "-O -u"
    c_debug_opts = ""
    mode = "release"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-antenna-iq"
elif args.run_mode == "debug":
    # run all modules in debug with regular operations data outputs, for testing modules
    python_opts = "-u"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-antenna-iq"
elif args.run_mode == "pythonprofiling":
    # run all modules in debug with python profiling, for optimizing python modules
    python_opts = "-O -u -m cProfile -o testing/python_testing/{module}.cprof"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-antenna-iq"
elif args.run_mode == "testdata":
    # run in scons release with python debug for tx data and print raw rf, for verifying data
    python_opts = "-u"
    c_debug_opts = ""
    mode = "release"
    data_write_args = "--file-type=hdf5 --enable-tx --enable-raw-rf"
elif args.run_mode == "engineeringdebug":
    # run all modules in debug with tx and rawrf data - this mode is very slow
    python_opts = "-u"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-bfiq --enable-antenna-iq --enable-raw-rf --enable-tx;"
elif args.run_mode == "filterdata":
    # run all modules in debug with rawrf, antennas_iq, and filter stage data. 
    python_opts = "-u"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-raw-rf --enable-antenna-iq"
else:
    print(f"Mode {args.run_mode} is unknown. Exiting without running Borealis")
    sys.exit(-1)

# Configure python first
modules = {"brian": "",
           "experiment_handler": "",
           "radar_control": "",
           "data_write": "",
           "realtime": "",
           "rx_signal_processing": "",
           "usrp_driver": ""}

for mod in modules.keys():
    opts = python_opts.format(module=mod)
    modules[mod] = f"source borealis_env{PYTHON_VERSION}/bin/activate; python{PYTHON_VERSION} {opts} src/{mod}.py" \

modules['data_write'] = modules['data_write'] + " " + data_write_args
modules['usrp_driver'] = modules['usrp_driver'] + " " + f'{mode} --c_debug_opts="{c_debug_opts}"'

if args.kwargs_string:
    modules['experiment_handler'] = modules['experiment_handler'] + " " + args.experiment_module + " " + \
                                    args.scheduling_mode_type + " --kwargs_string " + args.kwargs_string
else:
    modules['experiment_handler'] = modules['experiment_handler'] + " " + args.experiment_module + " " + \
                                    args.scheduling_mode_type
    
# # Configure C prog
# modules['usrp_driver'] = f"source mode {mode}; {c_debug_opts} usrp_driver"

# Set up the screenrc file and populate it
screenrc = BOREALISSCREENRC.format(
    START_RT=modules['realtime'],
    START_BRIAN=modules['brian'],
    START_USRP_DRIVER=modules['usrp_driver'],
    START_DSP=modules['rx_signal_processing'],
    START_DATAWRITE=modules['data_write'],
    START_EXPHAN=modules['experiment_handler'],
    START_RADCTRL=modules['radar_control'],
)

screenrc_file = os.environ['BOREALISPATH'] + "/borealisscreenrc"
with open(screenrc_file, 'w') as f:
    f.write(screenrc)

# Clean up any residuals in shared memory and dead screens
sp.call("rm -r /dev/shm/*", shell=True)
sp.call("screen -X -S borealis quit", shell=True)

# Give the os a chance to free all previously used sockets, etc.
time.sleep(1)

# Lights, camera, action!
screen_launch = "screen -S borealis -c " + screenrc_file
sp.call(screen_launch, shell=True)
