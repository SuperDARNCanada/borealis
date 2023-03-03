#!/usr/bin/python

"""
    options
    ~~~~~~~

    Parse all configuration options from the config.ini file for the current site.

    :copyright: 2023 SuperDARN Canada
    :author: Theodore Kolkman
"""

from dataclasses import dataclass, field
import os
import json

@dataclass(frozen=True)
class Options:
    def __post_init__(self):
        # Read in config.ini file for current site
        raw_config = self.load_config()

        self.site_id = raw_config['site_id']
        self.data_directory = raw_config["data_directory"]

        self.max_usrp_dac_amplitude = float(raw_config["max_usrp_dac_amplitude"])
        self.pulse_ramp_time = float(raw_config["pulse_ramp_time"])
        self.tr_window_time = float(raw_config["tr_window_time"])
        self.router_address = raw_config["router_address"]
        self.rt_address = raw_config["realtime_address"]
        self.ringbuffer_name = raw_config["ringbuffer_name"]

        self.main_antenna_count = int(raw_config["main_antenna_count"])
        self.interferometer_antenna_count = int(raw_config["interferometer_antenna_count"])

        # Parse N200 array and calculate main and intf antennas operating
        self.main_antennas = []
        self.interferometer_antennas = []
        for n200 in raw_config["n200s"]:
            rx = bool(n200["rx"])
            tx = bool(n200["tx"])
            rx_int = bool(n200["rx_int"])
            if rx or tx:
                main_antenna_num = int(n200["main_antenna"])
                self.main_antennas.append(main_antenna_num)
            if rx_int:
                intf_antenna_num = int(n200["interferometer_antenna"])
                self.interferometer_antennas.append(intf_antenna_num)
        self.main_antennas.sort()
        self.interferometer_antennas.sort()

        self.main_antenna_spacing = float(raw_config['main_antenna_spacing'])
        self.interferometer_antenna_spacing = float(raw_config['interferometer_antenna_spacing'])
        self.max_tx_sample_rate = float(raw_config['max_tx_sample_rate'])
        self.max_rx_sample_rate = float(raw_config['max_rx_sample_rate'])
        self.max_usrp_dac_amplitude = float(raw_config['max_usrp_dac_amplitude'])
        self.pulse_ramp_time = float(raw_config['pulse_ramp_time'])  # in seconds
        self.tr_window_time = float(raw_config['tr_window_time'])
        self.max_output_sample_rate = float(raw_config['max_output_sample_rate'])  
        # should use to check iqdata samples when adjusting the experiment during operations.
        self.max_number_of_filtering_stages = int(raw_config['max_number_of_filtering_stages'])
        self.max_number_of_filter_taps_per_stage = int(raw_config['max_number_of_filter_taps_per_stage'])
        self.max_freq = float(raw_config['max_freq'])  # Hz
        self.min_freq = float(raw_config['min_freq'])  # Hz
        self.minimum_pulse_length = float(raw_config['minimum_pulse_length'])  # us
        self.minimum_tau_spacing_length = float(raw_config['minimum_tau_spacing_length'])  # us
        # Minimum pulse separation is the minimum before the experiment treats it as a single
        # pulse (transmitting zeroes or no receiving between the pulses)
        # 125 us is approx two TX/RX times

        self.minimum_pulse_separation = float(raw_config['minimum_pulse_separation'])  # us
        self.usrp_master_clock_rate = float(raw_config['usrp_master_clock_rate']) # Hz
        self.router_address = raw_config['router_address']

        self.error_check()


        hdw_path = str(raw_config["hdw_path"])

        # Load information from the hardware files
        hdw_dat_file = f'{hdw_path}/hdw.dat.{os.environ["RADAR_ID"]}'

        try:
            with open(hdw_dat_file) as hdwdata:
                lines = hdwdata.readlines()
        except IOError:
            errmsg = f'Cannot open hdw.dat file at {hdw_dat_file}'
            raise ValueError(errmsg)

        lines[:] = [line for line in lines if line[0] != "#"]  # remove comments
        lines[:] = [line for line in lines if len(line.split()) != 0]  # remove blanks

        # Take the final line
        try:
            hdw = lines[-1]
        except IndexError:
            errmsg = f'Cannot find any valid lines in the hardware file: {hdw_dat_file}'
            raise ValueError(errmsg)
        # we now have the correct line of data.

        params = hdw.split()
        if len(params) != 22:
            errmsg = f'Found {len(params)} parameters in hardware file, expected 22'
            raise ValueError(errmsg)

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

        # Read in restrict.dat
        restricted = self.load_restrict()

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



    # ZMQ Identities
    radctrl_to_exphan_identity: str = "RADCTRL_EXPHAN_IDEN"
    radctrl_to_dsp_identity: str = "RADCTRL_DSP_IDEN"
    radctrl_to_driver_identity: str = "RADCTRL_DRIVER_IDEN"
    radctrl_to_brian_identity:str = "RADCTRL_BRIAN_IDEN"
    radctrl_to_dw_identity: str = "RADCTRL_DW_IDEN"
    driver_to_radctrl_identity: str = "DRIVER_RADCTRL_IDEN"
    driver_to_dsp_identity: str = "DRIVER_DSP_IDEN"
    driver_to_brian_identity: str = "DRIVER_BRIAN_IDEN"
    driver_to_mainaffinity_identity: str = "DRIVER_MAINAFFINITY_IDEN"
    driver_to_txaffinity_identity: str = "DRIVER_TXAFFINITY_IDEN"
    driver_to_rxaffinity_identity: str = "DRIVER_RXAFFINITY_IDEN"
    mainaffinity_to_driver_identity: str = "MAINAFFINITY_DRIVER_IDEN"
    txaffinity_to_driver_identity: str = "TXAFFINITY_DRIVER_IDEN"
    rxaffinity_to_driver_identity: str = "RXAFFINITY_DRIVER_IDEN"
    exphan_to_radctrl_identity: str = "EXPHAN_RADCTRL_IDEN"
    exphan_to_dsp_identity: str = "EXPHAN_DSP_IDEN"
    dsp_to_radctrl_identity: str = "DSP_RADCTRL_IDEN"
    dsp_to_driver_identity: str = "DSP_DRIVER_IDEN"
    dsp_to_exphan_identity: str = "DSP_EXPHAN_IDEN"
    dsp_to_dw_identity: str = "DSP_DW_IDEN"
    dspbegin_to_brian_identity: str = "DSPBEGIN_BRIAN_IDEN"
    dspend_to_brian_identity: str = "DSPEND_BRIAN_IDEN"
    dw_to_dsp_identity: str = "DW_DSP_IDEN"
    dw_to_radctrl_identity: str = "DW_RADCTRL_IDEN"
    rt_to_dw_identity: str = "DW_RT_IDEN"
    dw_to_rt_identity: str = "RT_DW_IDEN"
    brian_to_radctrl_identity: str = "BRIAN_RADCTRL_IDEN"
    brian_to_driver_identity: str = "BRIAN_DRIVER_IDEN"
    brian_to_dspbegin_identity: str = "BRIAN_DSPBEGIN_IDEN"
    brian_to_dspend_identity: str = "BRIAN_DSPEND_IDEN"

    def load_config():
        # Gather the borealis configuration information
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_ID']:
            raise ValueError('RADAR_ID env variable not set')
        path = f'{os.environ["BOREALISPATH"]}/config/' \
            f'{os.environ["RADAR_ID"]}/' \
            f'{os.environ["RADAR_ID"]}_config.ini'
        try:
            with open(path, 'r') as data:
                config = json.load(data)
        except IOError:
            print(f'IOError on config file at {path}')
            raise

        return config

    def load_restrict():
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_ID']:
            raise ValueError('RADAR_ID env variable not set')
        path = f'{os.environ["BOREALISPATH"]}/config/' \
            f'{os.environ["RADAR_ID"]}/' \
            f'restrict.dat.{os.environ["RADAR_ID"]}'
        try:
            with open(path) as data:
                restrict = data.readlines()
        except IOError:
            print(f'IOError on restrict.dat file at {path}')
            raise

        return restrict

    def error_check(self):
        if len(self.main_antennas) > 0:
            if min(self.main_antennas) < 0 or max(self.main_antennas) >= self.main_antenna_count:
                errmsg = 'main_antennas and main_antenna_count have incompatible values in'
                raise ValueError(errmsg)
        if len(self.interferometer_antennas) > 0:
            if min(self.interferometer_antennas) < 0 or \
                    max(self.interferometer_antennas) >= self.interferometer_antenna_count:
                errmsg = 'interferometer_antennas and interferometer_antenna_count have incompatible values'
                raise ValueError(errmsg)
    

