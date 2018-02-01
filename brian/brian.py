#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# brian.py
# 2018-01-30
# Communicate with all processes to administrate the borealis software

import sys
import os
import zmq
import time
from datetime import datetime, timedelta
sys.path.append(os.environ["BOREALISPATH"])

if __debug__:
	sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')  # TODO need to get this from scons environment, 'release' may be 'debug'
else:
	sys.path.append(os.environ["BOREALISPATH"]+ '/build/release/utils/protobuf')
import driverpacket_pb2
import sigprocpacket_pb2v
