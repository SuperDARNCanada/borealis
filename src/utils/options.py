"""
options
~~~~~~~

Parse all configuration options from the config.ini file for the current site.
Additionally, parse the hdw.dat and restrict.dat files of the current site for other
configuration information.

See https://borealis.readthedocs.io/en/latest/source/config_options.html for
detailed descriptions of each configuration option.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Options:
    """
    Parses all configuration options from the ``config.ini``, ``hdw.dat``, and ``restrict.dat`` files for the
    current site.

    .. note:

        Reads ``BOREALISPATH`` and ``RADAR_ID`` from the environment to determine the paths to the configuration files.
        If these are not found in the environment, instantiation will fail.

    Raises errors if invalid configuration options are detected, e.g. two TX channels connected to the same antenna,
    or two USRPs have the same IP address.
    """

    # config.ini options
    data_directory: str = field(
        init=False
    )  #: Location of output data files, e.g. ``"/data/borealis_data"``
    hdw_path: str = field(
        init=False
    )  #: Path to SuperDARN hardware files, e.g. ``"/usr/local/hdw"``
    rx_intf_antennas: list[int] = field(
        init=False
    )  #: Interferometer antennas connected to USRP RX channels.
    intf_antenna_locations: np.ndarray = field(init=False)
    """
    ``[x, y, z]`` coordinates for each antenna in the interferometer array, relative to the center-point of the main
    array, in meters. Example::

        {
            "0": [-22.86, 0.0, 0.0],
            "1": [-7.62, 0.0, 0.0],
            "2": [7.62, 0.0, 0.0],
            "3": [22.86, 0.0, 0.0]
        }
    """
    intf_antenna_count: int = field(
        init=False
    )  #: Total number of interferometer array antennas
    intf_antenna_spacing: float = field(
        init=False
    )  #: Distance between adjacent antennas in interferometer array [m]
    log_aggregator_addr: str = field(
        init=False
    )  #: Network address of remote log aggregator
    log_aggregator_bool: bool = field(
        init=False
    )  #: Flag to enable remote log aggregation
    log_aggregator_port: int = field(init=False)  #: Port of remote log aggregator
    log_console_bool: bool = field(init=False)  #: Flag to enable console logging
    log_directory: str = field(init=False)
    """Directory in which to store JSON log files, e.g. ``"/data/borealis_logs"``"""
    console_log_level: str = field(init=False)
    """Level for console logging.

    Supported levels are ``DEBUG``, ``VERBOSE``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``, and ``NOTSET``,
    or a numeric value between 0 and 50.
    """
    logfile_log_level: str = field(init=False)
    """Level for JSON logfile logging.

    Supported levels are ``DEBUG``, ``VERBOSE``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``, and ``NOTSET``,
    or a numeric value between 0 and 50.
    """
    aggregator_log_level: str = field(init=False)
    """Level for remote aggregator logging.

    Supported levels are ``DEBUG``, ``VERBOSE``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``, and ``NOTSET``,
    or a numeric value between 0 and 50.
    """
    log_logfile_bool: bool = field(init=False)  #: Flag to enable JSON file logging
    rx_main_antennas: list[int] = field(
        init=False
    )  #: Main antennas connected to RX channels
    tx_main_antennas: list[int] = field(
        init=False
    )  #: Main antennas connected to TX channels
    main_antenna_locations: np.ndarray = field(init=False)
    """
    ``[x, y, z]`` coordinates for each antenna in the main array, relative to the center-point of the main
    array, in meters. Example::

        {
            "0": [-114.3, 0.0, 0.0],
            "1": [-99.06, 0.0, 0.0],
            ...,
            "14": [99.06, 0.0, 0.0],
            "15": [114.3, 0.0, 0.0]
        }
    """
    main_antenna_count: int = field(init=False)  #: Total number of main array antennas
    main_antenna_spacing: float = field(
        init=False
    )  #: Distance between adjacent antennas in main array [m]
    max_filtering_stages: int = field(init=False)
    """Maximum number of filter stages allowed in a :class:`DecimationScheme`"""
    max_filter_taps_per_stage: int = field(init=False)
    """Maximum number of filter taps allowed in any :class:`DecimationStage` of a :class:`DecimationScheme`"""
    max_freq: float = field(init=False)  #: Maximum operating frequency [Hz]
    max_output_sample_rate: float = field(
        init=False
    )  #: Maximum output sampling rate [Hz]
    max_rx_sample_rate: float = field(init=False)  #: Maximum USRP RX sampling rate [Hz]
    max_tx_sample_rate: float = field(init=False)  #: Maximum USRP TX sample rate [Hz]
    max_usrp_dac_amplitude: float = field(
        init=False
    )  #: Max amplitude of USRP TX samples [V]
    min_freq: float = field(init=False)  #: Minimum operating frequency [Hz]
    min_pulse_length: float = field(
        init=False
    )  #: Minimum pulse duration [μs]. Dependent on AGC hold time
    min_pulse_separation: float = field(
        init=False
    )  #: Minimum separation before pulses are digitally combined [μs]
    min_tau_spacing_length: float = field(
        init=False
    )  #: Minimum duration between pulses in a pulse sequence [μs]
    num_beams: int = field(init=False)  #: Default number of beam directions to scan
    num_ranges: int = field(init=False)  #: Default number of range gates
    scan_direction: str = field(
        init=False
    )  #: Scan direction, "clockwise" or "counterclockwise"

    n200_addrs: list[str] = field(init=False)
    n200_count: int = field(init=False)
    pulse_ramp_time: float = field(init=False)  #: Linear ramp time for the pulse [s]
    rawacf_format: str = field(
        init=False
    )  #: Output format for rawacf files. Either ``"hdf5"`` or ``"dmap"``
    realtime_address: str = field(
        init=False
    )  #: Network address to serve DMAP data over, e.g. ``tcp://eth0:9696``
    ringbuffer_name: str = field(
        init=False
    )  #: Shared memory name for the RX sample buffer
    pulse_buffer_name: str = field(
        init=False
    )  #: Shared memory name for the TX sample buffer
    pulse_buffer_size: int = field(
        init=False
    )  #: Size of TX pulse buffer, per TX channel [bytes]
    router_address: str = field(
        init=False
    )  #: Internal network address used for IPC, e.g. ``tcp://127.0.0.1:9696``
    site_id: str = field(
        init=False
    )  #: Standard 3-letter ID of the radar, e.g. ``"sas"``
    standard_antenna_positions: bool = field(
        init=False
    )  #: Flag for standard (uniformly spaced) antennas
    tr_window_time: float = field(
        init=False
    )  #: Duration to window on either side of TX pulse for T/R signal [s]
    usrp_master_clock_rate: float = field(init=False)  #: E.g. ``"1.00E+08"``
    default_freqs: dict = field(
         init = False
    )  #: Common-mode and Sounding operating frequencies [kHz]

    # hdw.dat options
    altitude: float = field(init=False)
    analog_atten_stages: int = field(init=False)
    analog_rx_attenuator: float = field(init=False)
    analog_rx_rise: float = field(init=False)
    beam_sep: float = field(init=False)
    boresight: float = field(init=False)
    boresight_shift: float = field(init=False)
    geo_lat: float = field(init=False)
    geo_long: float = field(init=False)
    intf_offset: list[float] = field(init=False)
    max_beams: int = field(init=False)
    max_range_gates: int = field(init=False)
    phase_sign: int = field(init=False)
    status: int = field(init=False)
    tdiff_a: float = field(init=False)
    tdiff_b: float = field(init=False)
    velocity_sign: int = field(init=False)

    # restrict.dat options
    default_freq: int = field(init=False)
    restricted_ranges: list[tuple[int, int]] = field(init=False)

    # ZMQ Identities
    brian_to_driver_identity: str = "BRIAN_DRIVER_IDEN"
    brian_to_dspbegin_identity: str = "BRIAN_DSPBEGIN_IDEN"
    brian_to_dspend_identity: str = "BRIAN_DSPEND_IDEN"
    brian_to_radctrl_identity: str = "BRIAN_RADCTRL_IDEN"
    driver_to_brian_identity: str = "DRIVER_BRIAN_IDEN"
    driver_to_dsp_identity: str = "DRIVER_DSP_IDEN"
    driver_to_mainaffinity_identity: str = "DRIVER_MAINAFFINITY_IDEN"
    driver_to_radctrl_identity: str = "DRIVER_RADCTRL_IDEN"
    driver_to_rxaffinity_identity: str = "DRIVER_RXAFFINITY_IDEN"
    driver_to_txaffinity_identity: str = "DRIVER_TXAFFINITY_IDEN"
    dspbegin_to_brian_identity: str = "DSPBEGIN_BRIAN_IDEN"
    dspend_to_brian_identity: str = "DSPEND_BRIAN_IDEN"
    dsp_to_driver_identity: str = "DSP_DRIVER_IDEN"
    dsp_to_dw_identity: str = "DSP_DW_IDEN"
    dsp_to_exphan_identity: str = "DSP_EXPHAN_IDEN"
    dsp_to_radctrl_identity: str = "DSP_RADCTRL_IDEN"
    dsp_cfs_identity: str = "DSP_CFS_IDEN"
    dw_to_dsp_identity: str = "DW_DSP_IDEN"
    dw_to_radctrl_identity: str = "DW_RADCTRL_IDEN"
    dw_to_rt_identity: str = "DW_RT_IDEN"
    dw_cfs_identity: str = "DW_CFS_IDEN"
    exphan_to_dsp_identity: str = "EXPHAN_DSP_IDEN"
    exphan_to_radctrl_identity: str = "EXPHAN_RADCTRL_IDEN"
    mainaffinity_to_driver_identity: str = "MAINAFFINITY_DRIVER_IDEN"
    radctrl_to_brian_identity: str = "RADCTRL_BRIAN_IDEN"
    radctrl_to_driver_identity: str = "RADCTRL_DRIVER_IDEN"
    radctrl_to_dsp_identity: str = "RADCTRL_DSP_IDEN"
    radctrl_cfs_identity: str = "RADCTRL_CFS_IDEN"
    radctrl_to_dw_identity: str = "RADCTRL_DW_IDEN"
    radctrl_to_exphan_identity: str = "RADCTRL_EXPHAN_IDEN"
    rt_to_dw_identity: str = "RT_DW_IDEN"
    rxaffinity_to_driver_identity: str = "RXAFFINITY_DRIVER_IDEN"
    txaffinity_to_driver_identity: str = "TXAFFINITY_DRIVER_IDEN"

    def __post_init__(self):
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")
        if not os.environ["RADAR_ID"]:
            raise ValueError("RADAR_ID env variable not set")
        self._parse_config()  # Parse info from config file
        self._parse_hdw()  # Parse info from hdw file
        self._parse_restrict()  # Parse info from restrict.dat file
        self._verify_options()  # Check that all parsed values are valid

    def _parse_config(self):
        # Read in config.ini file for current site
        path = (
            f"{os.environ['BOREALISPATH']}/config/"
            f"{os.environ['RADAR_ID']}/"
            f"{os.environ['RADAR_ID']}_config.ini"
        )
        try:
            with open(path, "r") as data:
                raw_config = json.load(data)
        except OSError:
            errmsg = f"Cannot open config file at {path}"
            raise ValueError(errmsg)

        # Initialize all options from config file
        self.site_id = raw_config["site_id"]
        antenna_fields = raw_config["antennas"]

        # Parse N200 array and calculate which main and intf antennas operating
        self.n200_count = 0
        self.n200_addrs = []  # Used for checking IPs of N200s
        self.rx_main_antennas = []
        self.rx_intf_antennas = []
        self.tx_main_antennas = []

        def parse_channel(channel_str: str, rx: bool):
            """Parse the antenna number and which array it belongs to"""
            try:
                antenna_num = int(channel_str[1:])
            except ValueError:
                problem = "channel[1:] must be an integer"
                raise ValueError(problem)
            if channel_str[0] == "m":
                if channel_str[1:] not in antenna_fields["main_locations"].keys():
                    raise ValueError(
                        f"channel {channel_str[1:]} not in main antenna list {list(antenna_fields['main_locations'].keys())}"
                    )
                if rx:
                    self.rx_main_antennas.append(antenna_num)
                else:
                    self.tx_main_antennas.append(antenna_num)
            elif channel_str[0] == "i":
                if not rx:
                    raise ValueError(
                        "Cannot connect tx channel to interferometer array"
                    )
                if channel_str[1:] not in antenna_fields["intf_locations"].keys():
                    raise ValueError(
                        f"channel {channel_str[1:]} not in intf antenna list {list(antenna_fields['intf_locations'].keys())}"
                    )
                self.rx_intf_antennas.append(antenna_num)
            else:
                problem = "channel must start with either 'm' or 'i' for main or interferometer array"
                raise ValueError(problem)

        for n200 in raw_config["n200s"]:
            rx_channel_0 = n200["rx_channel_0"]
            rx_channel_1 = n200["rx_channel_1"]
            tx_channel_0 = n200["tx_channel_0"]
            n200_in_use = False

            if rx_channel_0 != "":
                try:
                    parse_channel(rx_channel_0, True)
                    n200_in_use = True
                except ValueError as err:
                    msg = f"; N200 {n200['addr']} rx_channel_0"
                    raise ValueError(str(err) + msg)
            if rx_channel_1 != "":
                try:
                    parse_channel(rx_channel_1, True)
                    n200_in_use = True
                except ValueError as err:
                    msg = f"; N200 {n200['addr']} rx_channel_1"
                    raise ValueError(str(err) + msg)
            if tx_channel_0 != "":
                try:
                    parse_channel(tx_channel_0, False)
                    n200_in_use = True
                except ValueError as err:
                    msg = f"; N200 {n200['addr']} tx_channel_0"
                    raise ValueError(str(err) + msg)

            if n200_in_use:
                self.n200_addrs.append(n200["addr"])
                self.n200_count += 1
        self.rx_main_antennas.sort()
        self.rx_intf_antennas.sort()
        self.tx_main_antennas.sort()

        # Parse the antenna parameters
        self.main_antenna_count = int(antenna_fields["main_antenna_count"])
        self.intf_antenna_count = int(antenna_fields["intf_antenna_count"])
        self.main_antenna_spacing = float(antenna_fields["main_antenna_spacing"])
        self.intf_antenna_spacing = float(antenna_fields["intf_antenna_spacing"])
        self.standard_antenna_positions = antenna_fields["standard_positions"]

        def parse_antennas(array_name: str):
            """Read the antenna_fields for an array and verify congruency of all fields"""
            config_locations = antenna_fields[f"{array_name}_locations"]
            antenna_locations = list()
            antenna_names = list()
            antenna_count = int(antenna_fields[f"{array_name}_antenna_count"])
            if len(config_locations.keys()) != antenna_count:
                raise ValueError(
                    f"Number of specified antenna locations != {array_name}_antenna_count ({len(config_locations.keys())} != {antenna_count})"
                )
            for name, location in config_locations.items():
                if len(location) != 3:
                    raise ValueError(
                        f"Antenna {name} has invalid location {location} (expected [x, y, z])"
                    )
                if int(name) < 0 or int(name) >= antenna_count:
                    raise ValueError(
                        f"Antenna {name} lies outside range [0, {array_name}_antenna_count) => [0, {antenna_count})"
                    )
                antenna_locations.append(location)
                antenna_names.append(int(name))
            sorted_antenna_indices = sorted(
                range(len(antenna_names)), key=lambda k: antenna_names[k]
            )
            sorted_locations = np.array(antenna_locations, dtype=np.float32)[
                sorted_antenna_indices
            ]
            if antenna_fields["standard_positions"] and len(sorted_locations) > 1:
                expected_diffs = np.zeros_like(sorted_locations[1:])
                expected_diffs[:, 0] = antenna_fields[f"{array_name}_antenna_spacing"]
                if not np.allclose(
                    sorted_locations[1:] - sorted_locations[:-1], expected_diffs
                ):
                    raise ValueError(
                        f"{array_name} locations not in line parallel to x-axis and equally spaced by {antenna_fields[f'{array_name}_antenna_spacing']} m"
                    )

            return sorted_locations

        self.main_antenna_locations = parse_antennas("main")
        self.intf_antenna_locations = parse_antennas("intf")

        self.min_freq = float(raw_config["min_freq"])  # Hz
        self.max_freq = float(raw_config["max_freq"])  # Hz
        self.default_freqs = raw_config["default_freqs"] # {"common": [...], "sounding": [...]} kHz
        self.min_pulse_length = float(raw_config["min_pulse_length"])  # us
        self.min_tau_spacing_length = float(raw_config["min_tau_spacing_length"])  # us
        self.num_beams = int(raw_config["num_beams"])
        self.num_ranges = int(raw_config["num_ranges"])
        self.scan_direction = raw_config["scan_direction"]

        # Minimum pulse separation is the minimum before the experiment treats it as a single pulse
        # (transmitting zeroes or no receiving between the pulses) 125 us is approx two TX/RX times
        self.min_pulse_separation = float(raw_config["min_pulse_separation"])  # us

        self.max_tx_sample_rate = float(raw_config["max_tx_sample_rate"])  # sps
        self.max_rx_sample_rate = float(raw_config["max_rx_sample_rate"])  # sps

        self.max_usrp_dac_amplitude = float(raw_config["max_usrp_dac_amplitude"])  # V
        self.pulse_ramp_time = float(raw_config["pulse_ramp_time"])  # s
        self.tr_window_time = float(raw_config["tr_window_time"])  # s

        self.usrp_master_clock_rate = float(raw_config["usrp_master_clock_rate"])  # sps
        self.max_output_sample_rate = float(raw_config["max_output_sample_rate"])  # sps
        self.max_filtering_stages = int(raw_config["max_filtering_stages"])
        self.max_filter_taps_per_stage = int(raw_config["max_filter_taps_per_stage"])

        self.router_address = raw_config["router_address"]
        self.realtime_address = raw_config["realtime_address"]
        self.ringbuffer_name = raw_config["ringbuffer_name"]
        self.pulse_buffer_name = raw_config["pulse_buffer_name"]
        self.pulse_buffer_size = int(raw_config["pulse_buffer_size"])

        self.data_directory = raw_config["data_directory"]
        self.rawacf_format = raw_config["rawacf_format"]
        self.log_directory = raw_config["log_handlers"]["logfile"]["directory"]
        self.hdw_path = raw_config["hdw_path"]

        self.console_log_level = raw_config["log_handlers"]["console"]["level"]
        self.logfile_log_level = raw_config["log_handlers"]["logfile"]["level"]
        self.aggregator_log_level = raw_config["log_handlers"]["aggregator"]["level"]
        self.log_console_bool = raw_config["log_handlers"]["console"]["enable"]
        self.log_logfile_bool = raw_config["log_handlers"]["logfile"]["enable"]
        self.log_aggregator_bool = raw_config["log_handlers"]["aggregator"]["enable"]
        self.log_aggregator_addr = raw_config["log_handlers"]["aggregator"]["addr"]
        self.log_aggregator_port = int(raw_config["log_handlers"]["aggregator"]["port"])

    def _parse_hdw(self):
        # Load information from the hardware file
        hdw_dat_file = f"{self.hdw_path}/hdw.dat.{os.environ['RADAR_ID']}"

        try:
            with open(hdw_dat_file) as hdwdata:
                lines = hdwdata.readlines()
        except OSError:
            errmsg = f"Cannot open hdw.dat file at {hdw_dat_file}"
            raise ValueError(errmsg)

        lines[:] = [line for line in lines if line[0] != "#"]  # remove comments
        lines[:] = [line for line in lines if len(line.split()) != 0]  # remove blanks

        # Take the final line
        try:
            hdw = lines[-1]
        except IndexError:
            errmsg = f"Cannot find any valid lines in the hardware file: {hdw_dat_file}"
            raise ValueError(errmsg)
        # we now have the correct line of data.

        params = hdw.split()
        if len(params) != 22:
            errmsg = f"Found {len(params)} parameters in hardware file, expected 22"
            raise ValueError(errmsg)

        self.status = int(params[1])  # 1 operational, -1 offline
        self.geo_lat = float(params[4])  # decimal degrees, S = negative
        self.geo_long = float(params[5])  # decimal degrees, W = negative
        self.altitude = float(params[6])  # metres
        self.boresight = float(
            params[7]
        )  # degrees from geographic north, CCW = negative.
        self.boresight_shift = float(
            params[8]
        )  # degrees from physical boresight. nominal 0.0 degrees
        self.beam_sep = float(params[9])  # degrees, nominal 3.24 degrees
        self.velocity_sign = int(params[10])  # +1 or -1
        self.phase_sign = int(
            params[11]
        )  # +1 indicates correct interferometry phase, -1 indicates 180
        self.tdiff_a = float(params[12])  # us for channel A.
        self.tdiff_b = float(params[13])  # us for channel B.
        # interferometer offset from midpoint of main, metres [x, y, z] where x is along line of
        # antennas, y is along array normal and z is altitude difference, in m.
        self.intf_offset = [float(params[14]), float(params[15]), float(params[16])]
        self.analog_rx_rise = float(params[17])  # us
        self.analog_rx_attenuator = float(params[18])  # dB
        self.analog_atten_stages = int(params[19])  # number of stages
        self.max_range_gates = int(params[20])
        self.max_beams = int(
            params[21]
        )  # so a beam number always points in a certain direction

    def _parse_restrict(self):
        # Read in restrict.dat
        path = (
            f"{os.environ['BOREALISPATH']}/config/"
            f"{os.environ['RADAR_ID']}/"
            f"restrict.dat.{os.environ['RADAR_ID']}"
        )
        try:
            with open(path) as data:
                restricted = data.readlines()
        except IOError:
            print(f"IOError on restrict.dat file at {path}")
            raise

        restricted[:] = [
            line for line in restricted if line[0] != "#"
        ]  # remove comments
        restricted[:] = [
            line for line in restricted if len(line.split()) != 0
        ]  # remove blanks

        for line in restricted:
            splitup = line.split("=")
            if len(splitup) == 2:
                if splitup[0].strip() == "default":
                    self.default_freq = int(splitup[1])  # kHz
                    restricted.remove(line)
                    break
        else:  # no break
            raise ValueError("No default frequency found in restrict.dat")

        self.restricted_ranges = []
        for line in restricted:
            splitup = line.split()
            if len(splitup) != 2:
                errmsg = "Error reading restricted frequency: more than two frequencies listed"
                raise ValueError(errmsg)
            try:
                splitup = [int(float(freq)) for freq in splitup]  # convert to ints
            except ValueError:
                errmsg = "Error parsing restrict.dat: frequencies must be valid numbers"
                raise ValueError(errmsg)
            restricted_range = tuple(splitup)
            self.restricted_ranges.append(restricted_range)

    def _verify_options(self):
        if self.site_id != os.environ["RADAR_ID"]:
            errmsg = f"site_id {self.site_id} is different from RADAR_ID {os.environ['RADAR_ID']}"
            raise ValueError(errmsg)

        if len(self.n200_addrs) != len(set(self.n200_addrs)):
            raise ValueError("Two or more n200s have identical IP addresses")

        if len(self.rx_main_antennas) > 0:
            if len(self.rx_main_antennas) != len(set(self.rx_main_antennas)):
                raise ValueError("rx_main_antennas has duplicate values")
        if len(self.tx_main_antennas) > 0:
            if len(self.tx_main_antennas) != len(set(self.tx_main_antennas)):
                raise ValueError("tx_main_antennas has duplicate values")
        if len(self.rx_intf_antennas) > 0:
            if len(self.rx_intf_antennas) != len(set(self.rx_intf_antennas)):
                raise ValueError("rx_intf_antennas has duplicate values")

        if self.num_beams <= 0:
            raise ValueError("num_beams must be > 0")
        if self.num_ranges <= 0:
            raise ValueError("num_ranges must be > 0")
        if self.scan_direction not in ["clockwise", "counterclockwise"]:
            raise ValueError(
                "scan_direction must be either `clockwise` or `counterclockwise`"
            )

        # TODO: Test that realtime_address and router_address are valid addresses

    def __str__(self):
        return_str = f"""    site_id = {self.site_id} \
                       \n    rx_main_antennas = {self.rx_main_antennas} \
                       \n    tx_main_antennas = {self.tx_main_antennas} \
                       \n    main_antenna_count = {self.main_antenna_count} \
                       \n    rx_intf_antennas = {self.rx_intf_antennas} \
                       \n    intf_antenna_count = {self.intf_antenna_count} \
                       \n    main_antenna_spacing = {self.main_antenna_spacing} metres \
                       \n    intf_antenna_spacing = {self.intf_antenna_spacing} metres \
                       \n    min_freq = {self.min_freq} Hz\
                       \n    max_freq = {self.max_freq} Hz\
                       \n    min_pulse_length = {self.min_pulse_length} us \
                       \n    min_tau_spacing_length = {self.min_tau_spacing_length} us \
                       \n    min_pulse_separation = {self.min_pulse_separation} us \
                       \n    max_tx_sample_rate = {self.max_tx_sample_rate} Hz (samples/sec)\
                       \n    max_rx_sample_rate = {self.max_rx_sample_rate} Hz (samples/sec)\
                       \n    max_usrp_dac_amplitude = {self.max_usrp_dac_amplitude} V\
                       \n    pulse_ramp_time = {self.pulse_ramp_time} s\
                       \n    tr_window_time = {self.tr_window_time} s\
                       \n    usrp_master_clock_rate = {self.usrp_master_clock_rate} Hz (samples/sec)\
                       \n    max_output_sample_rate = {self.max_output_sample_rate} Hz (samples/sec)\
                       \n    max_filtering_stages = {self.max_filtering_stages} \
                       \n    max_filter_taps_per_stage = {self.max_filter_taps_per_stage} \
                       \n    hdw_path = {self.hdw_path} \
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
                       \n    intf_offset = {self.intf_offset} m \
                       \n    analog_rx_rise = {self.analog_rx_rise} us \
                       \n    analog_rx_attenuator = {self.analog_rx_attenuator} dB \
                       \n    analog_atten_stages = {self.analog_atten_stages} \
                       \n    max_range_gates = {self.max_range_gates} \
                       \n    max_beams = {self.max_beams} \
                       \n    default_freq = {self.default_freq} kHz \
                       \n    restricted_ranges = {self.restricted_ranges} kHz \
                       \n    rawacf_format = {self.rawacf_format}
                       \n"""
        return return_str
