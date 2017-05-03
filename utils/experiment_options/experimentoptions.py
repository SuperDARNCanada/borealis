#!/usr/bin/env python

# COPYRIGHT 2017 SUPERDARN CANADA
"""
To load the config options to be used by the experiment and radar_control blocks. 
"""

import json

config_file = '/home/marci/code/URSP/placeholderOS/config.ini'  # TODO how to add this file


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
            self.tx_sample_rate = float(config['tx_sample_rate'])
            self.rx_sample_rate = float(config['rx_sample_rate'])
            self.tr_window_time = float(config['tr_window_time'])
            self.output_sample_rate = float(config['third_stage_sample_rate'])  # should use to check iqdata samples
            # when adjusting the experiment during operations.
            self.filter_description = {'filter_cutoff': config['third_stage_filter_cutoff'],
                                       'filter_transition': config['third_stage_filter_transition']}
            # TODO add appropriate timing here after timing is changed - can use to check for pulse spacing minimums
        except ValueError:
            # TODO: error
            pass
