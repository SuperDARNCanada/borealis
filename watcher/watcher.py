#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
# Author: Liam Graham
#
# watcher.py
# 2019-08-15
# Monitoring process to flag any problems related to rx/tx power


import inotify


def check_antennas_iq_file_power(iq_file):
	"""
	Checks that the power between antennas is reasonably close for each range in a record.
	If it is not, alert the squad.
	Args:
		iq_file:	The antenna iq file being checked.
	"""


