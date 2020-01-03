#!/usr/bin/python3

"""
Copyright SuperDARN Canada 2020
Keith Kotyk

Borealis Start up script

The Simpsons, Season 1 Episode 3 - Steamed Hams
"""
import sys
import subprocess as sp
import datetime
import os
import time

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
screen -t "Realtime" bash -c "{START_RT}"
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

#split -v
#focus right
#screen -t "Realtime" bash -c "{START_RT}"

detach
"""

#TODO: System likely needs some updating to get rawrf working properly.
if sys.argv[2] == "release":
    python_opts = "-O -u"
    c_debug_opts = ""
    mode = "release"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-bfiq --enable-antenna-iq"
elif sys.argv[2] == "debug":
    python_opts = "-u"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-bfiq --enable-antenna-iq"
elif sys.argv[2] == "pythonprofiling":
    python_opts = "-O -u -m cProfile -o testing/python_testing/{module}.cprof"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-bfiq --enable-antenna-iq"
elif sys.argv[2] == "engineeringdebug":
    python_opts = "-u"
    c_debug_opts = "/usr/local/cuda/bin/cuda-gdb -ex start"
    mode = "debug"
    data_write_args = "--file-type=hdf5 --enable-bfiq --enable-antenna-iq --enable-tx --enable-raw-rf;"
elif sys.argv[2] == "rawrf":
    python_opts = "-O -u"
    c_debug_opts = ""
    mode = "release"
    data_write_args = "--file-type=hdf5 --enable-raw-acfs --enable-bfiq --enable-antenna-iq --enable-raw-rf"
else:
    print("Mode {} is unknown. Exiting without running Borealis".format(sys.argv[2]))
    sys.exit(-1)

#Configure python first
modules= {"brian" : "", "experiment_handler" : "",
                "radar_control" : "", "data_write" : "",
                "realtime" : ""}

for mod in modules:
    opts = python_opts.format(module=mod)
    modules[mod] = "python3 {opts} {module}/{module}.py".format(opts=opts, module=mod)

modules['realtime'] = "source borealisrt_env/bin/activate;" + modules['realtime']
modules['experiment_handler'] = modules['experiment_handler'] + " " +  sys.argv[1]
modules['data_write'] = modules['data_write'] + " " + data_write_args

#Configure C progs
c_progs = ['usrp_driver', 'signal_processing']
for cprg in c_progs:
    modules[cprg] = "source mode {}; {} {}".format(mode, c_debug_opts, cprg)

#Configure terminal output to also go to file.
logfile_timestamp = datetime.datetime.utcnow().strftime("%Y.%m.%d.%H:%M")
for mod in modules:
    basic_screen_cmd = modules[mod] + " 2>&1 | tee {path}/{timestamp}-{module}; bash"
    path = os.environ['BOREALISPATH']+"/logs"
    modules[mod] = basic_screen_cmd.format(path=path,timestamp=logfile_timestamp, module=mod)

screenrc = BOREALISSCREENRC.format(
    START_RT=modules['realtime'],
    START_BRIAN=modules['brian'],
    START_USRP_DRIVER=modules['usrp_driver'],
    START_DSP=modules['signal_processing'],
    START_DATAWRITE=modules['data_write'],
    START_EXPHAN=modules['experiment_handler'],
    START_RADCTRL=modules['radar_control'],
)

screenrc_file = os.environ['BOREALISPATH']+"/borealisscreenrc"
with open(screenrc_file, 'w') as f:
    f.write(screenrc)

sp.call("rm -r /dev/shm/*", shell=True)
sp.call("screen -X -S borealis quit", shell=True)

# Give the os a chance to free all previously used sockets, etc.
time.sleep(1)

screen_launch = "screen -S borealis -c " + screenrc_file
sp.call(screen_launch, shell=True)
