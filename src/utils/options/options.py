#!/usr/bin/python

"""
    options
    ~~~~~~~

    Parse all configuration options from the config.ini file for the current site.

    :copyright: 2023 SuperDARN Canada
    :author: Theodore Kolkman
"""

import os
import json
from dataclasses import dataclass

@dataclass
class Options:
    def __post_init__(self):
        self.parse_config()     # Parse info from config file
        self.parse_hdw()        # Parse info from hdw file
        self.parse_restrict()   # Parse info from restrict.dat file
        self.verify_options()   # Check that all parsed values are valid


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
    dw_to_rt_identity: str = "RT_DW_IDEN"
    rt_to_dw_identity: str = "DW_RT_IDEN"
    brian_to_radctrl_identity: str = "BRIAN_RADCTRL_IDEN"
    brian_to_driver_identity: str = "BRIAN_DRIVER_IDEN"
    brian_to_dspbegin_identity: str = "BRIAN_DSPBEGIN_IDEN"
    brian_to_dspend_identity: str = "BRIAN_DSPEND_IDEN"
    
    def parse_config(self):
        # Read in config.ini file for current site
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_ID']:
            raise ValueError('RADAR_ID env variable not set')
        path = f'{os.environ["BOREALISPATH"]}/config/' \
            f'{os.environ["RADAR_ID"]}/' \
            f'{os.environ["RADAR_ID"]}_config.ini'
        try:
            with open(path, 'r') as data:
                raw_config = json.load(data)
        except IOError:
            print(f'IOError on config file at {path}')
            raise

        # Initialize all options from config file
        self.site_id = raw_config['site_id']
        self.main_antenna_count = int(raw_config["main_antenna_count"])
        self.intf_antenna_count = int(raw_config["intf_antenna_count"])

        # Parse N200 array and calculate which main and intf antennas operating
        self.num_n200s = 0
        self.n200_addrs = []
        self.main_antennas = []
        self.intf_antennas = []
        for n200 in raw_config["n200s"]:
            rx = bool(n200["rx"])
            tx = bool(n200["tx"])
            rx_int = bool(n200["rx_int"])
            if rx or tx:
                main_antenna_num = int(n200["main_antenna"])
                self.main_antennas.append(main_antenna_num)
            if rx_int:
                intf_antenna_num = int(n200["intf_antenna"])
                self.intf_antennas.append(intf_antenna_num)
            if rx or tx or rx_int:
                self.n200_addrs.append(n200["addr"])
                self.num_n200s += 1
        self.main_antennas.sort()
        self.intf_antennas.sort()

        self.main_antenna_spacing = float(raw_config['main_antenna_spacing'])  # m
        self.intf_antenna_spacing = float(raw_config['intf_antenna_spacing'])  # m
        self.min_freq = float(raw_config['min_freq'])  # Hz
        self.max_freq = float(raw_config['max_freq'])  # Hz
        self.minimum_pulse_length = float(raw_config['minimum_pulse_length'])  # us
        self.minimum_tau_spacing_length = float(raw_config['minimum_tau_spacing_length'])  # us
        # Minimum pulse separation is the minimum before the experiment treats it as a single pulse
        # (transmitting zeroes or no receiving between the pulses) 125 us is approx two TX/RX times
        self.minimum_pulse_separation = float(raw_config['minimum_pulse_separation'])  # us

        self.max_tx_sample_rate = float(raw_config['max_tx_sample_rate'])
        self.max_rx_sample_rate = float(raw_config['max_rx_sample_rate'])

        self.max_usrp_dac_amplitude = float(raw_config["max_usrp_dac_amplitude"])
        self.pulse_ramp_time = float(raw_config["pulse_ramp_time"])
        self.tr_window_time = float(raw_config["tr_window_time"])

        self.usrp_master_clock_rate = float(raw_config['usrp_master_clock_rate']) # Hz
        self.max_output_sample_rate = float(raw_config['max_output_sample_rate'])  
        # should use to check iqdata samples when adjusting the experiment during operations.
        self.max_filtering_stages = int(raw_config['max_filtering_stages'])
        self.max_filter_taps_per_stage = int(raw_config['max_filter_taps_per_stage'])

        self.router_address = raw_config["router_address"]
        self.realtime_address = raw_config["realtime_address"]
        self.ringbuffer_name = raw_config["ringbuffer_name"]

        self.data_directory = raw_config["data_directory"]
        self.log_directory = raw_config["log_directory"]
        self.log_level = raw_config["log_level"]
        self.log_console_bool = raw_config["log_handlers"]["console"]
        self.log_logfile_bool = raw_config["log_handlers"]["logfile"]
        self.log_aggregator_bool = raw_config["log_handlers"]["aggregator"]
        self.log_aggregator_addr = raw_config["log_aggregator_addr"]
        self.log_aggregator_port = int(raw_config["log_aggregator_port"])
        self.hdw_path = raw_config['hdw_path']


    def parse_hdw(self):
        # Load information from the hardware file
        hdw_dat_file = f'{self.hdw_path}/hdw.dat.{os.environ["RADAR_ID"]}'

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

        self.status = params[1]         # 1 operational, -1 offline
        self.geo_lat = params[4]        # decimal degrees, S = negative
        self.geo_long = params[5]       # decimal degrees, W = negative
        self.altitude = params[6]       # metres
        self.boresight = params[7]      # degrees from geographic north, CCW = negative.
        self.boresight_shift = params[8]  # degrees from physical boresight. nominal 0.0 degrees
        self.beam_sep = params[9]       # degrees, nominal 3.24 degrees
        self.velocity_sign = params[10] # +1.0 or -1.0
        self.phase_sign = params[11]    # +1 indicates correct interferometry phase, -1 indicates 180
        self.tdiff_a = params[12]       # us for channel A.
        self.tdiff_b = params[13]       # us for channel B.
        # interferometer offset from midpoint of main, metres [x, y, z] where x is along line of
        # antennas, y is along array normal and z is altitude difference, in m.
        self.intf_offset = [float(params[14]), float(params[15]), float(params[16])]
        self.analog_rx_rise = params[17]        # us
        self.analog_rx_attenuator = params[18]  # dB
        self.analog_atten_stages = params[19]   # number of stages
        self.max_range_gates = params[20]
        self.max_beams = params[21]  # so a beam number always points in a certain direction


    def parse_restrict(self):
        # Read in restrict.dat
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ['RADAR_ID']:
            raise ValueError('RADAR_ID env variable not set')
        path = f'{os.environ["BOREALISPATH"]}/config/' \
            f'{os.environ["RADAR_ID"]}/' \
            f'restrict.dat.{os.environ["RADAR_ID"]}'
        try:
            with open(path) as data:
                restricted = data.readlines()
        except IOError:
            print(f'IOError on restrict.dat file at {path}')
            raise

        restricted[:] = [line for line in restricted if line[0] != "#"]  # remove comments
        restricted[:] = [line for line in restricted if len(line.split()) != 0]  # remove blanks

        for line in restricted:
            splitup = line.split("=")
            if len(splitup) == 2:
                if splitup[0] == 'default' or splitup[0] == 'default ':
                    self.default_freq = int(splitup[1])  # kHz
                    restricted.remove(line)
                    break
        else: #no break
            raise Exception('No Default Frequency Found in Restrict.dat')

        self.restricted_ranges = []
        for line in restricted:
            splitup = line.split()
            if len(splitup) != 2:
                raise Exception('Problem with Restricted Frequency: A Range Len != 2')
            try:
                splitup = [int(float(freq)) for freq in splitup]  # convert to ints
            except ValueError:
                raise ValueError('Error parsing Restrict.Dat Frequency Ranges, Invalid Literal')
            restricted_range = tuple(splitup)
            self.restricted_ranges.append(restricted_range)


    def verify_options(self):
        if self.site_id != os.environ["RADAR_ID"]:
            errmsg = f'site_id {self.site_id} is different from RADAR_ID {os.environ["RADAR_ID"]}'
            raise ValueError(errmsg)

        print(self.n200_addrs)
        if len(self.n200_addrs) != len(set(self.n200_addrs)):
            errmsg = 'Two or more n200s have identical IP addresses'
            raise ValueError(errmsg)

        if len(self.main_antennas) > 0:
            if min(self.main_antennas) < 0 or max(self.main_antennas) >= self.main_antenna_count:
                errmsg = 'main_antennas and main_antenna_count are not consistent'
                raise ValueError(errmsg)
        if len(self.intf_antennas) > 0:
            if min(self.intf_antennas) < 0 or \
                    max(self.intf_antennas) >= self.intf_antenna_count:
                errmsg = 'intf_antennas and intf_antenna_count are not consistent'
                raise ValueError(errmsg)
        if len(self.main_antennas) != len(set(self.main_antennas)):
            errmsg = 'main_antennas contains duplicate values'
            raise ValueError(errmsg)
        if len(self.intf_antennas) != len(set(self.intf_antennas)):
            errmsg = 'intf_antennas contains duplicate values'
            raise ValueError(errmsg)

        # TODO: Test that realtime_address and router_address are valid addresses

        if not os.path.exists(self.data_directory):
            errmsg = f'data_directory {self.data_directory} does not exist'
            raise ValueError(errmsg)
        if not os.path.exists(self.log_directory):
            errmsg = f'log_directory {self.log_directory} does not exist'
            raise ValueError(errmsg)
        if not os.path.exists(self.hdw_path):
            errmsg = f'hdw_path directory {self.hdw_path} does not exist'
            raise ValueError(errmsg)


    def __repr__(self):
        return_str = f"""\n    main_antennas = {self.main_antennas} \
                    \n    main_antenna_count = {self.main_antenna_count} \
                    \n    intf_antennas = {self.intf_antennas} \
                    \n    intf_antenna_count = {self.intf_antenna_count} \
                    \n    main_antenna_spacing = {self.main_antenna_spacing} metres \
                    \n    intf_antenna_spacing = {self.intf_antenna_spacing} metres \
                    \n    max_tx_sample_rate = {self.max_tx_sample_rate} Hz (samples/sec)\
                    \n    max_rx_sample_rate = {self.max_rx_sample_rate} Hz (samples/sec)\
                    \n    max_usrp_dac_amplitude = {self.max_usrp_dac_amplitude} : 1\
                    \n    pulse_ramp_time = {self.pulse_ramp_time} s\
                    \n    tr_window_time = {self.tr_window_time} s\
                    \n    max_output_sample_rate = {self.max_output_sample_rate} Hz\
                    \n    max_filtering_stages = {self.max_filtering_stages} \
                    \n    max_filter_taps_per_stage = {self.max_filter_taps_per_stage} \
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