#!/usr/bin/env python

"""
    experiment options
    ~~~~~~~~~~~~~~~~~~

    To load the config options to be used by the experiment and radar_control blocks.
    Config data comes from the config.ini file, the hdw.dat file, and the restrict.dat file.

    :copyright: 2017 SuperDARN Canada
"""

import json
import os
from experiment_prototype.experiment_exception import ExperimentException


class ExperimentOptions:
    # TODO long init file, consider using multiple functions
    def __init__(self):
        """
        Create an object of necessary hardware and site parameters for use in the experiment.
        """

        # Gather the borealis configuration information
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_CODE']:
            raise ValueError('RADAR_CODE env variable not set')
        config_path = f'{os.environ["BOREALISPATH"]}/config/' \
                      f'{os.environ["RADAR_CODE"]}/' \
                      f'{os.environ["RADAR_CODE"]}_config.ini'
        restricted_path = f'{os.environ["BOREALISPATH"]}/config/' \
                          f'{os.environ["RADAR_CODE"]}/' \
                          f'restrict.dat.{os.environ["RADAR_CODE"]}'
        try:         # Try to open config file
            with open(config_path, 'r') as config_data:
                raw_config = json.load(config_data)
        except IOError:
            errmsg = f'Cannot open config file at {config_path}'
            raise IOError(errmsg)
        try:         # Try to open restricted.dat file
            with open(restricted_path) as restricted_freq_data:
                restricted = restricted_freq_data.readlines()
        except IOError:
            errmsg = f'Cannot open restrict.dat file at {restricted_path}'
            raise ExperimentException(errmsg)


        try:
            self._main_antenna_count = int(raw_config["main_antenna_count"])
            self._interferometer_antenna_count = int(raw_config["interferometer_antenna_count"])

            # Parse N200 array and calculate main and intf antennas operating
            self._main_antennas = []
            self._interferometer_antennas = []
            for n200 in raw_config["n200s"]:
                rx = bool(n200["rx"])
                tx = bool(n200["tx"])
                rx_int = bool(n200["rx_int"])
                if rx or tx:
                    main_antenna_num = int(n200["main_antenna"])
                    self._main_antennas.append(main_antenna_num)
                if rx_int:
                    intf_antenna_num = int(n200["interferometer_antenna"])
                    self._interferometer_antennas.append(intf_antenna_num)
            self._main_antennas.sort()
            self._interferometer_antennas.sort()

            self._main_antenna_spacing = float(raw_config['main_antenna_spacing'])
            self._interferometer_antenna_spacing = float(raw_config['interferometer_antenna_spacing'])
            self._max_tx_sample_rate = float(raw_config['max_tx_sample_rate'])
            self._max_rx_sample_rate = float(raw_config['max_rx_sample_rate'])
            self._max_usrp_dac_amplitude = float(raw_config['max_usrp_dac_amplitude'])
            self._pulse_ramp_time = float(raw_config['pulse_ramp_time'])  # in seconds
            self._tr_window_time = float(raw_config['tr_window_time'])
            self._max_output_sample_rate = float(
                raw_config['max_output_sample_rate'])  # should use to check iqdata samples
            # when adjusting the experiment during operations.
            self._max_number_of_filtering_stages = int(raw_config['max_number_of_filtering_stages'])
            self._max_number_of_filter_taps_per_stage = int(raw_config['max_number_of_filter_taps_per_stage'])
            self._site_id = raw_config['site_id']
            self._max_freq = float(raw_config['max_freq'])  # Hz
            self._min_freq = float(raw_config['min_freq'])  # Hz
            self._minimum_pulse_length = float(raw_config['minimum_pulse_length'])  # us
            self._minimum_tau_spacing_length = float(raw_config['minimum_tau_spacing_length'])  # us
            # Minimum pulse separation is the minimum before the experiment treats it as a single
            # pulse (transmitting zeroes or no receiving between the pulses)
            # 125 us is approx two TX/RX times

            self._minimum_pulse_separation = float(raw_config['minimum_pulse_separation'])  # us
            self._usrp_master_clock_rate = float(raw_config['usrp_master_clock_rate']) # Hz
            self._router_address = raw_config['router_address']
            self._radctrl_to_exphan_identity = str(raw_config["radctrl_to_exphan_identity"])
            self._radctrl_to_dsp_identity = str(raw_config["radctrl_to_dsp_identity"])
            self._radctrl_to_driver_identity = str(raw_config["radctrl_to_driver_identity"])
            self._radctrl_to_brian_identity = str(raw_config["radctrl_to_brian_identity"])
            self._radctrl_to_dw_identity = str(raw_config["radctrl_to_dw_identity"])
            self._driver_to_radctrl_identity = str(raw_config["driver_to_radctrl_identity"])
            self._driver_to_dsp_identity = str(raw_config["driver_to_dsp_identity"])
            self._driver_to_brian_identity = str(raw_config["driver_to_brian_identity"])
            self._exphan_to_radctrl_identity = str(raw_config["exphan_to_radctrl_identity"])
            self._exphan_to_dsp_identity = str(raw_config["exphan_to_dsp_identity"])
            self._dsp_to_radctrl_identity = str(raw_config["dsp_to_radctrl_identity"])
            self._dsp_to_driver_identity = str(raw_config["dsp_to_driver_identity"])
            self._dsp_to_exphan_identity = str(raw_config["dsp_to_exphan_identity"])
            self._dsp_to_dw_identity = str(raw_config["dsp_to_dw_identity"])
            self._dspbegin_to_brian_identity = str(raw_config["dspbegin_to_brian_identity"])
            self._dspend_to_brian_identity = str(raw_config["dspend_to_brian_identity"])
            self._dw_to_dsp_identity = str(raw_config["dw_to_dsp_identity"])
            self._dw_to_radctrl_identity = str(raw_config["dw_to_radctrl_identity"])
            self._brian_to_radctrl_identity = str(raw_config["brian_to_radctrl_identity"])
            self._brian_to_driver_identity = str(raw_config["brian_to_driver_identity"])
            self._brian_to_dspbegin_identity = str(raw_config["brian_to_dspbegin_identity"])
            self._brian_to_dspend_identity = str(raw_config["brian_to_dspend_identity"])

            if len(self.main_antennas) > 0:
                if min(self.main_antennas) < 0 or max(self.main_antennas) >= self.main_antenna_count:
                    errmsg = 'main_antennas and main_antenna_count have incompatible values in'\
                            f' {config_file}'
                    raise ExperimentException(errmsg)
            if len(self.interferometer_antennas) > 0:
                if min(self.interferometer_antennas) < 0 or \
                        max(self.interferometer_antennas) >= self.interferometer_antenna_count:
                    errmsg = 'interferometer_antennas and interferometer_antenna_count have'\
                            f' incompatible values in {config_file}'
                    raise ExperimentException(errmsg)

            hdw_path = str(raw_config["hdw_path"])

            # TODO add appropriate signal process maximum time here after timing is changed - can
            # use to check for pulse spacing minimums, pace the driver

        except ValueError as e:
            # TODO: error
            raise e

        hdw_dat_file = f'{hdw_path}/hdw.dat.{os.environ["RADAR_CODE"]}'

        try:
            with open(hdw_dat_file) as hdwdata:
                lines = hdwdata.readlines()
        except IOError:
            errmsg = f'Cannot open hdw.dat file at {hdw_dat_file}'
            raise ExperimentException(errmsg)

        lines[:] = [line for line in lines if line[0] != "#"]  # remove comments
        lines[:] = [line for line in lines if len(line.split()) != 0]  # remove blanks

        # Take the final line
        try:
            hdw = lines[-1]
        except IndexError:
            errmsg = f'Cannot find any valid lines in the hardware file: {hdw_dat_file}'
            raise ExperimentException(errmsg)
        # we now have the correct line of data.

        params = hdw.split()
        if len(params) != 22:
            errmsg = f'Found {len(params)} parameters in hardware file, expected 22'
            raise ExperimentException(errmsg)

        self._status = params[1]  # 1 operational, -1 offline
        self._geo_lat = params[4]  # decimal degrees, S = negative
        self._geo_long = params[5]  # decimal degrees, W = negative
        self._altitude = params[6]  # metres
        self._boresight = params[7]  # degrees from geographic north, CCW = negative.
        self._boresight_shift = params[8]  # degrees from physical boresight. nominal 0.0 degrees
        self._beam_sep = params[9]  # degrees, nominal 3.24 degrees
        self._velocity_sign = params[10]  # +1.0 or -1.0
        self._phase_sign = params[11]  # +1 indicates correct interferometry phase, -1 indicates 180
        self._tdiff_a = params[12]  # us for channel A.
        self._tdiff_b = params[13]  # us for channel B.

        self._intf_offset = [float(params[14]), float(params[15]), float(params[16])]
        # interferometer offset from
        # midpoint of main, metres [x, y, z] where x is along line of antennas, y is along array
        # normal and z is altitude difference, in m.
        self._analog_rx_rise = params[17]  # us
        self._analog_rx_attenuator = params[18]  # dB
        self._analog_atten_stages = params[19]  # number of stages
        self._max_range_gates = params[20]
        self._max_beams = params[21]  # so a beam number always points in a certain direction

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
        return_str = f"""\n    main_antennas = {self.main_antennas} \
                    \n    main_antenna_count = {self.main_antenna_count} \
                    \n    interferometer_antennas = {self.interferometer_antennas} \
                    \n    interferometer_antenna_count = {self.interferometer_antenna_count} \
                    \n    main_antenna_spacing = {self.main_antenna_spacing} metres \
                    \n    interferometer_antenna_spacing = {self.interferometer_antenna_spacing} metres \
                    \n    max_tx_sample_rate = {self.max_tx_sample_rate} Hz (samples/sec)\
                    \n    max_rx_sample_rate = {self.max_rx_sample_rate} Hz (samples/sec)\
                    \n    max_usrp_dac_amplitude = {self.max_usrp_dac_amplitude} : 1\
                    \n    pulse_ramp_time = {self.pulse_ramp_time} s\
                    \n    tr_window_time = {self.tr_window_time} s\
                    \n    max_output_sample_rate = {self.max_output_sample_rate} Hz\
                    \n    max_number_of_filtering_stages = {self.max_number_of_filtering_stages} \
                    \n    max_number_of_filter_taps_per_stage = {self.max_number_of_filter_taps_per_stage} \
                    \n    site_id = {self.site_id} \
                    \n    geo_lat = {self.geo_lat} degrees \
                    \n    geo_long = {self.geo_long} degrees\
                    \n    altitude = {self.altitude} metres \
                    \n    boresight = {self.boresight} degrees from geographic north, CCW = negative. \
                    \n    boresight_shift = {self.boresight_shift} degrees. \
                    \n    beam_sep = {self.beam_sep} degrees\
                    \n    velocity_sign = {self.velocity_sign} \
                    \n    tdiff_a = {self.tdiff_a} us \
                    \n    tdiff_b = {self.tdiff_b} us \
                    \n    phase_sign = {self.phase_sign} \
                    \n    intf_offset = {self.intf_offset} \
                    \n    analog_rx_rise = {self.analog_rx_rise} us \
                    \n    analog_rx_attenuator = {self.analog_rx_attenuator} dB \
                    \n    analog_atten_stages = {self.analog_atten_stages} \
                    \n    max_range_gates = {self.max_range_gates} \
                    \n    max_beams = {self.max_beams} \
                    \n    max_freq = {self.max_freq} \
                    \n    min_freq = {self.min_freq} \
                    \n    minimum_pulse_length = {self. minimum_pulse_length} \
                    \n    minimum_tau_spacing_length = {self.minimum_tau_spacing_length} \
                    \n    minimum_pulse_separation = {self.minimum_pulse_separation} \
                    \n    tr_window_time = {self.tr_window_time} \
                    \n    default_freq = {self.default_freq} \
                    \n    restricted_ranges = {self.restricted_ranges} \
                     """
        return return_str

    @property
    def main_antennas(self):
        return self._main_antennas

    @property
    def main_antenna_count(self):
        return self._main_antenna_count

    @property
    def interferometer_antennas(self):
        return self._interferometer_antennas

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
    def max_number_of_filtering_stages(self):
        return self._max_number_of_filtering_stages

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
    def minimum_tau_spacing_length(self):
        return self._minimum_tau_spacing_length  # us

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
    def boresight_shift(self):
        return self._boresight_shift  # degrees

    @property
    def beam_sep(self):
        return self._beam_sep  # degrees

    @property
    def velocity_sign(self):
        return self._velocity_sign   # +1.0 or -1.0

    @property
    def tdiff_a(self):
        return self._tdiff_a  # us

    @property
    def tdiff_b(self):
        return self._tdiff_b  # us

    @property
    def analog_rx_attenuator(self):
        return self._analog_rx_attenuator  # dB

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

