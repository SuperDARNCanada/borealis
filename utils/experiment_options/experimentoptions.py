#!/usr/bin/env python

# Copyright 2017 SuperDARN Canada

"""
To load the config options to be used by the experiment and radar_control blocks. 
Config data comes from the config.ini file and the hdw.dat file.
"""

import json
import string
import datetime
import os

python_path = os.environ['PYTHONPATH']
config_file = python_path + 'config.ini'  # TODO how to add this file
hdw_dat_file = python_path + 'hdw.dat.'

class ExperimentOptions:
    def __init__(self):
        """
        Create an object of config data for use in the experiment.
        """
        with open(config_file) as config_data:
            config = json.load(config_data)
        try:
            self.main_antenna_count = int(config['main_antenna_count'])
            self.interferometer_antenna_count = int(config['interferometer_antenna_count'])
            self.main_antenna_spacing = int(config['main_antenna_spacing'])
            self.interferometer_antenna_spacing = int(config['interferometer_antenna_spacing'])
            self.tx_sample_rate = float(config['tx_sample_rate'])
            self.rx_sample_rate = float(config['rx_sample_rate'])
            self.max_usrp_dac_amplitude = float(config['max_usrp_dac_amplitude'])
            self.pulse_ramp_time = float(config['pulse_ramp_time'])  # in seconds
            self.tr_window_time = float(config['tr_window_time'])
            self.output_sample_rate = float(config['third_stage_sample_rate'])  # should use to check iqdata samples
            # when adjusting the experiment during operations.
            self.filter_description = {'filter_cutoff': config['third_stage_filter_cutoff'],
                                       'filter_transition': config['third_stage_filter_transition']}
            self.site_id = config['site_id']
            # TODO add appropriate timing here after timing is changed - can use to check for pulse spacing minimums
        except ValueError:
            # TODO: error
            pass

        today = datetime.datetime.today()
        year_start = datetime.datetime(today.year,1,1,0,0,0,0) # start of the year
        year_timedelta = today - year_start

        with open(hdw_dat_file + self.site_id) as hdwdata:
            lines = hdwdata.readlines()
        lines[:] = [line for line in lines if line[0] != "#"]  # remove comments
        lines[:] = [line for line in lines if len(string.split(line)) != 0]  # remove blanks
        lines[:] = [line for line in lines if string.split(line)[1] > today.year or
                    (string.split(line)[1] == today.year and string.split(line)[2] >
                     year_timedelta.total_seconds())]  # only take present & future hdw data

        # there should only be one line left, however if there are more we will take the
        # one that expires first.
        if len(lines) > 1:
            times = [[string.split(line)[1], string.split(line)[2]] for line in lines]
            min_year = times[0][0]
            min_yrsec = times[0][1]
            hdw_index = 0
            for i in range(len(times)):
                year = times[i][0]
                yrsec = times[i][1]
                if year < min_year:
                    hdw_index = i
                elif year == min_year:
                    if yrsec < min_yrsec:
                        hdw_index = i
            hdw = lines[hdw_index]
        else:
            hdw = lines[0]
        # we now have the correct line of data.
        params = string.split(hdw)
        self.geo_lat = params[3]  # decimal degrees, S = negative
        self.geo_long = params[4]  # decimal degrees, W = negative
        self.altitude = params[5]  # metres
        self.boresight = params[6]  # degrees from geographic north, CCW = negative.
        self.beam_sep = params[7]  # degrees TODO is this necessary, or is this a min.
        self.velocity_sign = params[8]  # +1.0 or -1.0
        self.analog_rx_attenuator = params[9]  # dB
        self.tdiff = params[10]
        self.phase_sign = params[11]
        self.intf_offset = [params[12], params[13], params[14]]  # interferometer offset from
        # midpoint of main, metres [x, y, z] where x is along line of antennas, y is along array
        # normal and z is altitude difference.
        self.analog_rx_rise = params[15]  # us
        self.analog_atten_stages = params[16]  # number of stages
        self.max_range_gates = params[17]
        self.max_beams = params[18]  # so a beam number always points in a certain direction
                        # TODO Is this last one necessary - why don't we specify directions in angle.