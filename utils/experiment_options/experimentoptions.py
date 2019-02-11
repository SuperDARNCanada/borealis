#!/usr/bin/env python

# Copyright 2017 SuperDARN Canada

"""
To load the config options to be used by the experiment and radar_control blocks.
Config data comes from the config.ini file, the hdw.dat file, and the restrict.dat file.
"""

import json
import datetime
import os

from experiment_prototype.experiment_exception import ExperimentException

borealis_path = os.environ['BOREALISPATH']
config_file = borealis_path + '/config.ini'
hdw_dat_file = borealis_path + '/hdw.dat.'
restricted_freq_file = borealis_path + '/restrict.dat.'

class ExperimentOptions:
    # TODO long init file, consider using multiple functions
    def __init__(self):
        """
        Create an object of necessary hardware and site parameters for use in the experiment.
        """
        try:
            with open(config_file) as config_data:
                config = json.load(config_data)
        except IOError:
            errmsg = 'Cannot open config file at {}'.format(config_file)
            raise ExperimentException(errmsg)
        try:
            self._main_antenna_count = int(config['main_antenna_count'])
            self._interferometer_antenna_count = int(config['interferometer_antenna_count'])
            self._main_antenna_spacing = float(config['main_antenna_spacing'])
            self._interferometer_antenna_spacing = float(config['interferometer_antenna_spacing'])
            self._max_tx_sample_rate = float(config['max_tx_sample_rate'])
            self._max_rx_sample_rate = float(config['max_rx_sample_rate'])
            self._max_usrp_dac_amplitude = float(config['max_usrp_dac_amplitude'])
            self._pulse_ramp_time = float(config['pulse_ramp_time'])  # in seconds
            self._tr_window_time = float(config['tr_window_time'])
            self._max_output_sample_rate = float(
                config['max_output_sample_rate'])  # should use to check iqdata samples
            # when adjusting the experiment during operations.
            self._number_of_filtering_stages = int(config['number_of_filtering_stages'])
            self._max_number_of_filter_taps_per_stage = int(config['max_number_of_filter_taps_per_stage'])
            self._site_id = config['site_id']
            self._max_freq = float(config['max_freq'])  # Hz
            self._min_freq = float(config['min_freq'])  # Hz
            self._minimum_pulse_length = float(config['minimum_pulse_length'])  # us
            self._minimum_mpinc_length = float(config['minimum_mpinc_length'])  # us
            # Minimum pulse separation is the minimum before the experiment treats it as a single
            # pulse (transmitting zeroes or no receiving between the pulses)
            # 125 us is approx two TX/RX times

            self._minimum_pulse_separation = float(config['minimum_pulse_separation'])  # us
            self._usrp_master_clock_rate = float(config['usrp_master_clock_rate']) # Hz
            self._router_address = config['router_address']
            self._radctrl_to_exphan_identity = str(config["radctrl_to_exphan_identity"])
            self._radctrl_to_dsp_identity = str(config["radctrl_to_dsp_identity"])
            self._radctrl_to_driver_identity = str(config["radctrl_to_driver_identity"])
            self._radctrl_to_brian_identity = str(config["radctrl_to_brian_identity"])
            self._radctrl_to_dw_identity = str(config["radctrl_to_dw_identity"])
            self._driver_to_radctrl_identity = str(config["driver_to_radctrl_identity"])
            self._driver_to_dsp_identity = str(config["driver_to_dsp_identity"])
            self._driver_to_brian_identity = str(config["driver_to_brian_identity"])
            self._exphan_to_radctrl_identity = str(config["exphan_to_radctrl_identity"])
            self._exphan_to_dsp_identity = str(config["exphan_to_dsp_identity"])
            self._dsp_to_radctrl_identity = str(config["dsp_to_radctrl_identity"])
            self._dsp_to_driver_identity = str(config["dsp_to_driver_identity"])
            self._dsp_to_exphan_identity = str(config["dsp_to_exphan_identity"])
            self._dsp_to_dw_identity = str(config["dsp_to_dw_identity"])
            self._dspbegin_to_brian_identity = str(config["dspbegin_to_brian_identity"])
            self._dspend_to_brian_identity = str(config["dspend_to_brian_identity"])
            self._dw_to_dsp_identity = str(config["dw_to_dsp_identity"])
            self._dw_to_radctrl_identity = str(config["dw_to_radctrl_identity"])
            self._brian_to_radctrl_identity = str(config["brian_to_radctrl_identity"])
            self._brian_to_driver_identity = str(config["brian_to_driver_identity"])
            self._brian_to_dspbegin_identity = str(config["brian_to_dspbegin_identity"])
            self._brian_to_dspend_identity = str(config["brian_to_dspend_identity"])

            # TODO add appropriate signal process maximum time here after timing is changed - can
            # use to check for pulse spacing minimums, pace the driver

        except ValueError as e:
            # TODO: error
            raise e

        today = datetime.datetime.today()
        year_start = datetime.datetime(today.year, 1, 1, 0, 0, 0, 0)  # start of the year
        year_timedelta = today - year_start

        try:
            with open(hdw_dat_file + self.site_id) as hdwdata:
                lines = hdwdata.readlines()
        except IOError:
            errmsg = 'Cannot open hdw.dat.{} file at {}'.format(self.site_id, (hdw_dat_file + self.site_id))
            raise ExperimentException(errmsg)

        lines[:] = [line for line in lines if line[0] != "#"]  # remove comments
        lines[:] = [line for line in lines if len(line.split()) != 0]  # remove blanks
        lines[:] = [line for line in lines if int(line.split()[1]) > today.year or
                    (int(line.split()[1]) == today.year and float(line.split()[2]) >
                     year_timedelta.total_seconds())]  # only take present & future hdw data

        # there should only be one line left, however if there are more we will take the
        # one that expires first.
        if len(lines) > 1:
            times = [[line.split()[1], line.split()[2]] for line in lines]
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
            try:
                hdw = lines[0]
            except IndexError:
                errmsg = 'Cannot find any valid lines for this time period in the hardware file ' \
                         '{}'.format((hdw_dat_file + self.site_id))
                raise ExperimentException(errmsg)
        # we now have the correct line of data.

        params = hdw.split()
        if len(params) != 19:
            errmsg = 'Found {} parameters in hardware file, expected 19'.format(len(params))
            raise ExperimentException(errmsg)

        self._geo_lat = params[3]  # decimal degrees, S = negative
        self._geo_long = params[4]  # decimal degrees, W = negative
        self._altitude = params[5]  # metres
        self._boresight = params[6]  # degrees from geographic north, CCW = negative.
        self._beam_sep = params[7]  # degrees TODO is this necessary, or is this a min. - for
        # post-processing software in RST? check with others.
        self._velocity_sign = params[8]  # +1.0 or -1.0
        self._analog_rx_attenuator = params[9]  # dB
        self._tdiff = params[10] # ns
        self._phase_sign = params[11]
        self._intf_offset = [float(params[12]), float(params[13]), float(params[14])]  #
        # interferometer offset from
        # midpoint of main, metres [x, y, z] where x is along line of antennas, y is along array
        # normal and z is altitude difference.
        self._analog_rx_rise = params[15]  # us
        self._analog_atten_stages = params[16]  # number of stages
        self._max_range_gates = params[17]
        self._max_beams = params[18]  # so a beam number always points in a certain direction
        # TODO Is this last one necessary - why don't we specify directions in angle. - also for post-processing so check if it applies to Borealis

        try:
            with open(restricted_freq_file + self.site_id) as restricted_freq_data:
                restricted = restricted_freq_data.readlines()
        except IOError:
            errmsg = 'Cannot open restrict.dat.{} file at {}'.format(self.site_id,
                                                                (restricted_freq_file + self.site_id))
            raise ExperimentException(errmsg)

        restricted[:] = [line for line in restricted if line[0] != "#"]  # remove comments
        restricted[:] = [line for line in restricted if len(line.split()) != 0]  # remove blanks

        for line in restricted:
            splitup = line.split("=")
            if len(splitup) == 2:
                if splitup[0] == 'default' or splitup[0] == 'default ':
                    self.__default_freq = int(splitup[1])  # kHz
                    restricted.remove(line)
                    break
        else: #no break
            raise Exception('No Default Frequency Found in Restrict.dat')

        self.__restricted_ranges = []
        for line in restricted:
            splitup = line.split()
            if len(splitup) != 2:
                raise Exception('Problem with Restricted Frequency: A Range Len != 2')
            try:
                splitup = [int(float(freq)) for freq in splitup]  # convert to ints
            except ValueError:
                raise ValueError('Error parsing Restrict.Dat Frequency Ranges, Invalid Literal')
            restricted_range = tuple(splitup)
            self.__restricted_ranges.append(restricted_range)

    def __repr__(self):
        return_str = """\n    main_antenna_count = {} \
                    \n    interferometer_antenna_count = {} \
                    \n    main_antenna_spacing = {} metres \
                    \n    interferometer_antenna_spacing = {} metres \
                    \n    max_tx_sample_rate = {} Hz (samples/sec)\
                    \n    max_rx_sample_rate = {} Hz (samples/sec)\
                    \n    max_usrp_dac_amplitude = {} : 1\
                    \n    pulse_ramp_time = {} s\
                    \n    tr_window_time = {} s\
                    \n    max_output_sample_rate = {} Hz\
                    \n    number_of_filtering_stages = {} \
                    \n    max_number_of_filter_taps_per_stage = {} \
                    \n    site_id = {} \
                    \n    geo_lat = {} degrees \
                    \n    geo_long = {} degrees\
                    \n    altitude = {} metres \
                    \n    boresight = {} degrees from geographic north, CCW = negative. \
                    \n    beam_sep = {} degrees\
                    \n    velocity_sign = {} \
                    \n    analog_rx_attenuator = {} dB \
                    \n    tdiff = {} us \
                    \n    phase_sign = {} \
                    \n    intf_offset = {} \
                    \n    analog_rx_rise = {} us \
                    \n    analog_atten_stages = {} \
                    \n    max_range_gates = {} \
                    \n    max_beams = {} \
                    \n    max_freq = {} \
                    \n    min_freq = {} \
                    \n    minimum_pulse_length = {} \
                    \n    minimum_mpinc_length = {} \
                    \n    minimum_pulse_separation = {} \
                    \n    tr_window_time = {} \
                    \n    atten_window_time_start = {} \
                    \n    atten_window_time_end = {} \
                    \n    default_freq = {} \
                    \n    restricted_ranges = {} \
                     """.format(self.main_antenna_count, self.interferometer_antenna_count,
                                self.main_antenna_spacing, self.interferometer_antenna_spacing,
                                self.max_tx_sample_rate, self.max_rx_sample_rate,
                                self.max_usrp_dac_amplitude, self.pulse_ramp_time,
                                self.tr_window_time, self.max_output_sample_rate,
                                self.number_of_filtering_stages, self.max_number_of_filter_taps_per_stage,
                                self.site_id, self.geo_lat, self.geo_long,
                                self.altitude, self.boresight, self.beam_sep, self.velocity_sign,
                                self.analog_rx_attenuator, self.tdiff, self.phase_sign,
                                self.intf_offset, self.analog_rx_rise, self.analog_atten_stages,
                                self.max_range_gates, self.max_beams, self.max_freq, self.min_freq,
                                self. minimum_pulse_length, self.minimum_mpinc_length,
                                self.minimum_pulse_separation, self.tr_window_time,
                                self.atten_window_time_start, self.atten_window_time_end,
                                self.default_freq, self.restricted_ranges)
        return return_str

    @property
    def main_antenna_count(self):
        return self._main_antenna_count

    @property
    def interferometer_antenna_count(self):
        return self._interferometer_antenna_count

    @property
    def main_antenna_spacing(self):
        return self._main_antenna_spacing

    @property
    def interferometer_antenna_spacing(self):
        return self._interferometer_antenna_spacing

    @property
    def max_tx_sample_rate(self):
        return self._max_tx_sample_rate

    @property
    def max_rx_sample_rate(self):
        return self._max_rx_sample_rate

    @property
    def max_usrp_dac_amplitude(self):
        return self._max_usrp_dac_amplitude

    @property
    def pulse_ramp_time(self):
        return self._pulse_ramp_time  # in seconds

    @property
    def tr_window_time(self):
        return self._tr_window_time  # in seconds

    @property
    def max_output_sample_rate(self):
        return self._max_output_sample_rate  # Hz

    @property
    def number_of_filtering_stages(self):
        return self._number_of_filtering_stages

    @property
    def max_number_of_filter_taps_per_stage(self):
        return self._max_number_of_filter_taps_per_stage
  
    @property
    def site_id(self):
        return self._site_id

    @property
    def max_freq(self):
        return self._max_freq  # Hz

    @property
    def min_freq(self):
        return self._min_freq  # Hz

    @property
    def minimum_pulse_length(self):
        return self._minimum_pulse_length  # us

    @property
    def minimum_mpinc_length(self):
        return self._minimum_mpinc_length  # us

    @property
    def minimum_pulse_separation(self):
        """
        Minimum pulse separation is the minimum before the experiment treats it as a single pulse
        (transmitting zeroes or no receiving between the pulses)
        """
        return self._minimum_pulse_separation  # us

    @property
    def usrp_master_clock_rate(self):
        return self._usrp_master_clock_rate

    @property
    def router_address(self):
        return self._router_address

    @property
    def radctrl_to_exphan_identity(self):
        return self._radctrl_to_exphan_identity

    @property
    def radctrl_to_dsp_identity(self):
        return self._radctrl_to_dsp_identity

    @property
    def radctrl_to_driver_identity(self):
        return self._radctrl_to_driver_identity

    @property
    def radctrl_to_brian_identity(self):
        return self._radctrl_to_brian_identity

    @property
    def radctrl_to_dw_identity(self):
        return self._radctrl_to_dw_identity

    @property
    def driver_to_radctrl_identity(self):
        return self._driver_to_radctrl_identity

    @property
    def driver_to_dsp_identity(self):
        return self._driver_to_dsp_identity

    @property
    def driver_to_brian_identity(self):
        return self._driver_to_brian_identity

    @property
    def exphan_to_radctrl_identity(self):
        return self._exphan_to_radctrl_identity

    @property
    def exphan_to_dsp_identity(self):
        return self._exphan_to_dsp_identity

    @property
    def dsp_to_radctrl_identity(self):
        return self._dsp_to_radctrl_identity

    @property
    def dsp_to_driver_identity(self):
        return self._dsp_to_driver_identity

    @property
    def dsp_to_exphan_identity(self):
        return self._dsp_to_exphan_identity

    @property
    def dsp_to_dw_identity(self):
        return self._dsp_to_dw_identity

    @property
    def dspbegin_to_brian_identity(self):
        return self._dspbegin_to_brian_identity

    @property
    def dspend_to_brian_identity(self):
        return self._dspend_to_brian_identity

    @property
    def dw_to_dsp_identity(self):
        return self._dw_to_dsp_identity

    @property
    def dw_to_radctrl_identity(self):
        return self._dw_to_radctrl_identity

    @property
    def brian_to_radctrl_identity(self):
        return self._brian_to_radctrl_identity

    @property
    def brian_to_driver_identity(self):
        return self._brian_to_driver_identity

    @property
    def brian_to_dspbegin_identity(self):
        return self._brian_to_dspbegin_identity

    @property
    def brian_to_dspend_identity(self):
        return self._brian_to_dspend_identity

    @property
    def geo_lat(self):
        return self._geo_lat  # decimal degrees, S = negative

    @property
    def geo_long(self):
        return self._geo_long  # decimal degrees, W = negative

    @property
    def altitude(self):
        return self._altitude  # metres

    @property
    def boresight(self):
        return self._boresight  # degrees from geographic north, CCW = negative.

    @property
    def beam_sep(self):
        return self._beam_sep  # degrees TODO is this necessary, or is this a min. - for
                        # post-processing software in RST? check with others.

    @property
    def velocity_sign(self):
        return self._velocity_sign   # +1.0 or -1.0

    @property
    def analog_rx_attenuator(self):
        return self._analog_rx_attenuator  # dB

    @property
    def tdiff(self):
        return self._tdiff  # ns

    @property
    def phase_sign(self):
        return self._phase_sign

    @property
    def intf_offset(self):
        return self._intf_offset # interferometer offset from
    # midpoint of main, metres [x, y, z] where x is along line of antennas, y is along array
    # normal and z is altitude difference.

    @property
    def analog_rx_rise(self):
        return self._analog_rx_rise  # us

    @property
    def analog_atten_stages(self):
        return self._analog_atten_stages  # number of stages

    @property
    def max_range_gates(self):
        return self._max_range_gates

    @property
    def max_beams(self):
        return self._max_beams  # so a beam number always points in a certain direction

    @property
    def default_freq(self):
        return self.__default_freq  # kHz

    @property
    def restricted_ranges(self):
        """
        given in tuples of kHz
        """
        return self.__restricted_ranges

