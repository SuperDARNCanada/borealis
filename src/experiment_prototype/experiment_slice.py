"""
    experiment_slice
    ~~~~~~~~~~~~~~~~~~~~
    This module contains the class for experiment slices.

    :copyright: 2023 SuperDARN Canada
    :author: Remington Rohel
"""
# built-in
from dataclasses import InitVar
import itertools
import math

# third-party
import numpy as np
from pydantic.dataclasses import dataclass, Field
from pydantic import (
    validator, root_validator, conlist, conint, confloat, StrictBool, StrictInt, PositiveInt, PositiveFloat,
)
from scipy.constants import speed_of_light
from typing import Optional, Union, Literal, Callable

# local
from utils.options import Options
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.decimation_scheme.decimation_scheme import DecimationScheme
from experiment_prototype import list_tests

options = Options()

slice_key_set = frozenset([
    "acf",
    "acfint",
    "align_sequences",
    "averaging_method",
    "beam_angle",
    "clrfrqrange",
    "comment",
    "cpid",
    "first_range",
    "freq",
    "intn",
    "intt",
    "lag_table",
    "num_ranges",
    "pulse_len",
    "pulse_phase_offset",
    "pulse_sequence",
    "range_sep",
    "rx_beam_order",
    "rx_int_antennas",
    "rx_main_antennas",
    "rxonly",
    "scanbound",
    "seqoffset",
    "slice_id",
    "tau_spacing",
    "tx_antennas",
    "tx_antenna_pattern",
    "tx_beam_order",
    "wait_for_first_scanbound",
    "xcf",
    ])
hidden_key_set = frozenset(['clrfrqflag', 'slice_interfacing'])
"""
These are used by the build_scans method (called from the experiment_handler every time the
experiment is run). If set by the user, the values will be overwritten and therefore ignored.
"""


def default_callable():
    """This function does nothing, and exists only as a default value for Callable fields in ExperimentSlice"""
    pass


class SliceConfig:
    validate_assignment = True
    validate_all = True
    allow_mutation = False
    extra = 'forbid'
    arbitrary_types_allowed = True


freq_float_hz = confloat(ge=options.min_freq, le=options.max_freq)
freq_float_khz = confloat(ge=options.min_freq / 1e3, le=options.max_freq / 1e3)
freq_int_hz = conint(ge=options.min_freq, le=options.max_freq)
freq_int_khz = conint(ge=options.min_freq / 1e3, le=options.max_freq / 1e3)


@dataclass(config=SliceConfig)
class ExperimentSlice:
    """
    These are the keys that are set by the user when initializing a slice. Some are required, some can
    be defaulted, and some are set by the experiment and are read-only.

    **Slice Keys Required by the User**

    beam_angle *required*
        list of beam directions, in degrees off azimuth. Positive is E of N. The beam_angle list length
        = number of beams. Traditionally beams have been 3.24 degrees separated but we don't refer to
        them as beam -19.64 degrees, we refer as beam 1, beam 2. Beam 0 will be the 0th element in the
        list, beam 1 will be the 1st, etc. These beam numbers are needed to write the [rx|tx]_beam_order
        list. This is like a mapping of beam number (list index) to beam direction off boresight.
    clrfrqrange *required or freq required*
        range for clear frequency search, should be a list of length = 2, [min_freq, max_freq] in kHz.
        **Not currently supported.**
    first_range *required*
        first range gate, in km
    freq *required or clrfrqrange required*
        transmit/receive frequency, in kHz. Note if you specify clrfrqrange it won't be used.
    intt *required or intn required*
        duration of an averaging period (integration), in ms. (maximum)
    intn *required or intt required*
        number of averages to make a single averaging period (integration), only used if intt = None
    num_ranges *required*
        Number of range gates to receive for. Range gate time is equal to pulse_len and range gate
        distance is the range_sep, calculated from pulse_len.
    pulse_len *required*
        length of pulse in us. Range gate size is also determined by this.
    pulse_sequence *required*
        The pulse sequence timing, given in quantities of tau_spacing, for example normalscan = [0, 14,
        22, 24, 27, 31, 42, 43].
    rx_beam_order *required*
        beam numbers written in order of preference, one element in this list corresponds to one
        averaging period. Can have lists within the list, resulting in multiple beams running
        simultaneously in the averaging period, so imaging. A beam number of 0 in this list gives us the
        direction of the 0th element in the beam_angle list. It is up to the writer to ensure their beam
        pattern makes sense. Typically rx_beam_order is just in order (scanning W to E or E to W, ie.
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]. You can list numbers multiple times in
        the rx_beam_order list, for example [0, 1, 1, 2, 1] or use multiple beam numbers in a single
        averaging period (example [[0, 1], [3, 4]], which would trigger an imaging integration. When we
        do imaging we will still have to quantize the directions we are looking in to certain beam
        directions. It is up to the user to ensure that this field works well with the specified
        tx_beam_order or tx_antenna_pattern.
    rxonly *read-only*
        A boolean flag to indicate that the slice doesn't transmit, only receives.
    tau_spacing *required*
        multi-pulse increment (mpinc) in us, Defines minimum space between pulses.

    **Defaultable Slice Keys**

    acf *defaults*
        flag for rawacf generation. The default is False. If True, the following fields are also used: -
        averaging_method (default 'mean') - xcf (default True if acf is True) - acfint (default True if
        acf is True) - lagtable (default built based on all possible pulse combos) - range_sep (will be
        built by pulse_len to verify any provided value)
    acfint *defaults*
        flag for interferometer autocorrelation data. The default is True if acf is True, otherwise
        False.
    align_sequences *defaults*
        flag for aligning the start of the first pulse in each sequence to tenths of a second. Default
        False.
    averaging_method *defaults*
        a string defining the type of averaging to be done. Current methods are 'mean' or 'median'. The
        default is 'mean'.
    comment *defaults*
        a comment string that will be placed in the borealis files describing the slice. Defaults to
        empty string.
    lag_table *defaults*
        used in acf calculations. It is a list of lags. Example of a lag: [24, 27] from 8-pulse
        normalscan. This defaults to a lagtable built by the pulse sequence provided. All combinations
        of pulses will be calculated, with both the first pulses and last pulses used for lag-0.
    pulse_phase_offset *defaults*
        a handle to a function that will be used to generate one phase per each pulse in the sequence.
        If a function is supplied, the beam iterator, sequence number, and number of pulses in the
        sequence are passed as arguments that can be used in this function. The default is None if no
        function handle is supplied.

        encode_fn(beam_iter, sequence_num, num_pulses):
            return np.ones(size=(num_pulses))

        The return value must be numpy array of num_pulses in size. The result is a single phase shift
        for each pulse, in degrees.

        Result is expected to be real and in degrees and will be converted to complex radians.
    range_sep *defaults*
        a calculated value from pulse_len. If already set, it will be overwritten to be the correct
        value determined by the pulse_len. Used for acfs. This is the range gate separation, in the
        radial direction (away from the radar), in km.
    rx_int_antennas *defaults*
        The antennas to receive on in interferometer array, default is all antennas given max number
        from config.
    rx_main_antennas *defaults*
        The antennas to receive on in main array, default is all antennas given max number from config.
    scanbound *defaults*
        A list of seconds past the minute for averaging periods in a scan to align to. Defaults to None,
        not required. If one slice in an experiment has a scanbound, they all must.
    seqoffset *defaults*
        offset in us that this slice's sequence will begin at, after the start of the sequence. This is
        intended for CONCURRENT interfacing, when you want multiple slice's pulses in one sequence you
        can offset one slice's sequence from the other by a certain time value so as to not run both
        frequencies in the same pulse, etc. Default is 0 offset.
    tx_antennas *defaults*
        The antennas to transmit on, default is all main antennas given max number from config.
    tx_antenna_pattern *defaults*
        experiment-defined function which returns a complex weighting factor of magnitude <= 1 for each
        tx antenna used in the experiment. The return value of the function must be an array of size
        [num_beams, main_antenna_count] with all elements having magnitude <= 1. This function is
        analogous to the beam_angle field in that it defines the transmission pattern for the array, and
        the tx_beam_order field specifies which "beam" to use in a given averaging period.
    tx_beam_order *defaults, but required if tx_antenna_pattern given*
        beam numbers written in order of preference, one element in this list corresponds to one
        averaging period. A beam number of 0 in this list gives us the direction of the 0th element in
        the beam_angle list. It is up to the writer to ensure their beam pattern makes sense. Typically
        tx_beam_order is just in order (scanning W to E or E to W, i.e. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15]. You can list numbers multiple times in the tx_beam_order list, for
        example [0, 1, 1, 2, 1], but unlike rx_beam_order, you CANNOT use multiple beam numbers in a
        single averaging period. In other words, this field MUST be a list of integers, as opposed to
        rx_beam_order, which can be a list of lists of integers. The length of this list must be equal
        to the length of the rx_beam_order list. If tx_antenna_pattern is given, the items in
        tx_beam_order specify which row of the return from tx_antenna_pattern to use to beamform a given
        transmission. Default is None, i.e. rxonly slice.
    wait_for_first_scanbound *defaults*
        A boolean flag to determine when an experiment starts running. True (default) means an
        experiment will wait until the first averaging period in a scan to start transmitting. False
        means an experiment will not wait for the first averaging period, but will instead start
        transmitting at the nearest averaging period. Note: for multi-slice experiments, the first slice
        is the only one impacted by this parameter.
    xcf *defaults*
        flag for cross-correlation data. The default is True if acf is True, otherwise False.

    **Read-only Slice Keys**

    clrfrqflag *read-only*
        A boolean flag to indicate that a clear frequency search will be done. **Not currently
        supported.**
    cpid *read-only*
        The ID of the experiment, consistent with existing radar control programs. This is actually an
        experiment-wide attribute but is stored within the slice as well. This is provided by the user
        but not within the slice, instead when the experiment is initialized.
    slice_id *read-only*
        The ID of this slice object. An experiment can have multiple slices. This is not set by the user
        but instead set by the experiment when the slice is added. Each slice id within an experiment is
        unique. When experiments start, the first slice_id will be 0 and incremented from there.
    slice_interfacing *read-only*
        A dictionary of slice_id : interface_type for each sibling slice in the experiment at any given
        time.
    """
    # NOTE: The order of fields matters, as the validation is done according to the order defined here. Validation
    # of some fields assumes that other fields have already been validated, so be careful and test if making any changes
    # to the order below.

    # These fields are for checking the validity of the user-specified fields, to ensure the slice is
    # compatible with the experiment settings.
    tx_minfreq: freq_float_hz
    tx_maxfreq: freq_float_hz
    rx_minfreq: freq_float_hz
    rx_maxfreq: freq_float_hz
    txctrfreq: freq_float_khz
    rxctrfreq: freq_float_khz
    tx_bandwidth: float
    rx_bandwidth: float
    output_rx_rate: float
    transition_bandwidth: float

    # These fields can be specified in exp_slice_dict, subject to some conditions. Some may have dynamic default values.
    beam_angle: conlist(float, unique_items=True)
    cpid: StrictInt
    decimation_scheme: DecimationScheme
    first_range: Union[PositiveFloat, PositiveInt]
    num_ranges: conint(gt=0, le=options.max_range_gates)
    tau_spacing: conint(ge=options.min_tau_spacing_length)
    pulse_len: conint(ge=options.min_pulse_length)
    pulse_sequence: conlist(conint(ge=0), unique_items=True)
    rx_beam_order: conlist(Union[list[conint(ge=0)], conint(ge=0)])
    rxonly: StrictBool
    slice_id: conint(ge=0)
    freq: Union[freq_float_khz, freq_int_khz]

    # These fields have default values. Some have specification requirements in conjunction with each other
    # e.g. one of intt or intn must be specified.
    tx_antennas: Optional[conlist(conint(ge=0, lt=options.main_antenna_count),
                                  max_items=options.main_antenna_count,
                                  unique_items=True)] = Field(default_factory=list)
    rx_main_antennas: Optional[conlist(conint(ge=0, lt=options.main_antenna_count),
                                       max_items=options.main_antenna_count,
                                       unique_items=True)] = Field(default_factory=list)
    rx_int_antennas: Optional[conlist(conint(ge=0, lt=options.intf_antenna_count),
                                      max_items=options.intf_antenna_count,
                                      unique_items=True)] = Field(default_factory=list)
    tx_antenna_pattern: Optional[Callable] = default_callable
    tx_beam_order: Optional[list[Union[conint(ge=0), list[conint(ge=0)]]]] = Field(default_factory=list)
    intt: Optional[float] = -1
    scanbound: Optional[list[PositiveFloat]] = Field(default_factory=list)
    pulse_phase_offset: Optional[Callable] = default_callable
    clrfrqrange: Optional[conlist(freq_float_khz, max_items=2)] = Field(default_factory=list)
    clrfrqflag: StrictBool = Field(init=False)

    acf: Optional[StrictBool] = False
    acfint: Optional[StrictBool] = False
    align_sequences: Optional[StrictBool] = False
    averaging_method: Optional[Literal['mean', 'median']] = 'mean'
    comment: Optional[str] = ''
    intn: Optional[int] = -1
    lag_table: Optional[list[list[StrictInt]]] = Field(default_factory=list)
    range_sep: Optional[PositiveFloat] = Field(init=False)
    seqoffset: Optional[conint(ge=0)] = 0
    wait_for_first_scanbound: Optional[StrictBool] = False
    xcf: Optional[bool] = False

    # Validators which check that all mutually exclusive sets of fields have one option set

    @root_validator(pre=True)
    def check_tx_specifier(cls, values):
        if 'tx_antenna_pattern' not in values and 'tx_beam_order' in values:
            errmsg = f"tx_beam_order must be specified if tx_antenna_pattern specified. Slice: {values['slice_id']}"
            raise ExperimentException(errmsg)
        elif 'tx_beam_order' in values and 'rxonly' in values and values['rxonly'] is True:
            errmsg = f"rxonly specified as True but tx_beam_order specified. Slice: {values['slice_id']}"
            raise ExperimentException(errmsg)
        elif 'tx_beam_order' not in values and 'rxonly' in values and values['rxonly'] is False:
            errmsg = f"rxonly specified as False but tx_beam_order not given. Slice: {values['slice_id']}"
            raise ExperimentException(errmsg)
        return values

    @root_validator(pre=True)
    def check_intt_intn(cls, values):
        if 'intt' not in values and 'intn' not in values:
            errmsg = f"Slice must specify either an intn (unitless) or intt in ms. Slice: {values['slice_id']}"
            raise ExperimentException(errmsg)
        if 'intt' in values and 'intn' in values:
            print('intn is set in experiment slice but will not be used due to intt')
            values.pop('intn')
            # TODO Log warning intn will not be used
        return values

    @root_validator(pre=True)
    def check_freq_clrfrqrange(cls, values):
        if 'clrfrqrange' in values:
            values['clrfrqflag'] = True

            if 'freq' in values and values['freq'] not in range(values['clrfrqrange'][0], values['clrfrqrange'][1]):
                # TODO: Log this appropriately
                print("Slice parameter 'freq' removed as 'clrfrqrange' takes precedence. If this is not desired,"
                      "remove 'clrfrqrange' parameter from experiment.")
        elif 'freq' in values:
            values['clrfrqflag'] = False
        else:
            errmsg = f"A freq or clrfrqrange must be specified in a slice. Slice: {values['slice_id']}"
            raise ExperimentException(errmsg)
        return values

    # Validate that a list is increasing

    @validator('pulse_sequence', 'beam_angle')
    def check_list_increasing(cls, v_list):
        if not list_tests.is_increasing(v_list):
            raise ValueError(f"not increasing: {v_list}")
        return v_list

    # Validators that depend on other previously-validated fields

    @validator('tau_spacing')
    def check_tau_spacing(cls, tau_spacing, values):
        # TODO : tau_spacing needs to be an integer multiple of pulse_len in ros - is there a max ratio
        #  allowed for pulse_len/tau_spacing ? Add this check and add check for each slice's tx duty-cycle
        #  and make sure we aren't transmitting the entire time after combination with all slices
        if not math.isclose((tau_spacing * values['output_rx_rate'] % 1.0), 0.0, abs_tol=0.0001):
            raise ValueError(f"Slice {values['slice_id']} correlation lags will be off because tau_spacing "
                             f"{tau_spacing} us is not a multiple of the output rx sampling period "
                             f"(1/output_rx_rate {values['output_rx_rate']} Hz).")
        return tau_spacing

    @validator('pulse_len')
    def check_pulse_len(cls, pulse_len, values):
        if pulse_len > values['tau_spacing']:
            raise ValueError(f"Slice {values['slice_id']} pulse length greater than tau_spacing")
        if pulse_len <= 2 * options.pulse_ramp_time * 1.0e6:
            raise ValueError(f"Slice {values['slice_id']} pulse length too small")
        if 'acf' in values and values['acf']:
            # The below check is an assumption that is made during acf calculation
            # (1 output received sample = 1 range separation)
            if not math.isclose(pulse_len * 1.0e-6, (1 / values['output_rx_rate']), abs_tol=0.000001):
                errmsg = "For an experiment slice with real-time acfs, pulse length must be equal " \
                         f"(within 1 us) to 1/output_rx_rate to make acfs valid. Current pulse length is " \
                         f"{pulse_len} us, output rate is {values['output_rx_rate']} Hz."
                raise ExperimentException(errmsg)
        return pulse_len

    @validator('intt')
    def check_intt(cls, intt, values):
        if intt is -1:  # Not provided
            return

        # check intn and intt make sense given tau_spacing, and pulse_sequence.
        # Sequence length is length of pulse sequence plus the scope sync delay time.
        # TODO: this is an old check and seqtime now set in sequences class, update.
        seq_len = values['tau_spacing'] * (values['pulse_sequence'][-1]) + \
                  (values['num_ranges'] + 19 + 10) * values['pulse_len']  # us
        if seq_len > (intt * 1000):  # seq_len in us, intt in ms
            raise ExperimentException(f"Slice {values['slice_id']}: pulse sequence is too long for integration time "
                                      f"given")
        return intt

    @validator('tx_antennas')
    def check_tx_antennas(cls, tx_antennas, values):
        if not tx_antennas:
            return [i for i in options.main_antennas]
        return tx_antennas

    @validator('rx_main_antennas')
    def check_rx_main_antennas(cls, rx_main_antennas, values):
        if not rx_main_antennas:
            return [i for i in options.main_antennas]
        return rx_main_antennas

    @validator('rx_int_antennas')
    def check_rx_int_antennas(cls, rx_int_antennas, values):
        if not rx_int_antennas:
            return [i for i in options.intf_antennas]
        return rx_int_antennas

    @validator('tx_antenna_pattern')
    def check_tx_antenna_pattern(cls, tx_antenna_pattern, values):
        if tx_antenna_pattern is default_callable:  # No value given
            return

        antenna_pattern = tx_antenna_pattern(values['freq'], values['tx_antennas'],
                                             options.main_antenna_spacing)
        if not isinstance(antenna_pattern, np.ndarray):
            raise ExperimentException(f"Slice {values['slice_id']} tx antenna pattern return is not a numpy array")
        else:
            if len(antenna_pattern.shape) != 2:
                raise ExperimentException(f"Slice {values['slice_id']} tx antenna pattern return shape "
                                          f"{antenna_pattern.shape} must be 2-dimensional")
            elif antenna_pattern.shape[1] != options.main_antenna_count:
                raise ExperimentException(f"Slice {values['slice_id']} tx antenna pattern return 2nd dimension "
                                          f"({antenna_pattern.shape[1]}) must be equal to number of main antennas "
                                          f"({options.main_antenna_count})")
            antenna_pattern_mag = np.abs(antenna_pattern)
            if np.argwhere(antenna_pattern_mag > 1.0).size > 0:
                raise ExperimentException(f"Slice {values['slice_id']} tx antenna pattern return must not have any "
                                          f"values with a magnitude greater than 1")
        return tx_antenna_pattern

    @validator('rx_beam_order', each_item=True)
    def check_rx_beam_order(cls, rx_beam, values):
        if isinstance(rx_beam, list):
            for beamnum in rx_beam:
                if beamnum >= len(values['beam_angle']):
                    errmsg = f"Beam number {beamnum} could not index in beam_angle list of " \
                             f"length {len(values['beam_angle'])}. Slice: {values['slice_id']}"
                    raise ExperimentException(errmsg)
        else:
            if rx_beam >= len(values['beam_angle']):
                errmsg = f"Beam number {rx_beam} could not index in beam_angle list of length " \
                         f"{len(values['beam_angle'])}. Slice: {values['slice_id']}"
                raise ExperimentException(errmsg)
        return rx_beam

    @validator('tx_beam_order')
    def check_tx_beam_order(cls, tx_beam_order, values):
        if not tx_beam_order:   # Empty list, was not specified
            return

        if len(tx_beam_order) != len(values['rx_beam_order']):
            errmsg = f"tx_beam_order does not have same length as rx_beam_order. Slice: {values['slice_id']}"
            raise ExperimentException(errmsg)
        for element in tx_beam_order:
            if element >= len(values['beam_angle']) and \
                    ('tx_antenna_pattern' not in values or not values['tx_antenna_pattern']):
                errmsg = f"Beam number {element} in tx_beam_order could not index in beam_angle " \
                         f"list of length {len(values['beam_angle'])}. Slice: {values['slice_id']}"
                raise ExperimentException(errmsg)

        num_beams = len(values['beam_angle'])
        if 'tx_antenna_pattern' in values and values['tx_antenna_pattern']:
            antenna_pattern = values['tx_antenna_pattern'](values['freq'], values['tx_antennas'],
                                                           options.main_antenna_spacing)
            if isinstance(antenna_pattern, np.ndarray):
                num_beams = antenna_pattern.shape[0]
        for bmnum in tx_beam_order:
            if bmnum >= num_beams:
                raise ExperimentException(f"Slice {values['slice_id']} scan tx beam number {bmnum} DNE")

        return tx_beam_order

    @validator('scanbound')
    def check_scanbound(cls, scanbound, values):
        if not scanbound:   # No scanbound defined
            return

        if 'intt' not in values:
            raise ExperimentException(f"Slice {values['slice_id']} must have intt enabled to use scanbound")
        elif any(i < 0 for i in scanbound):
            raise ExperimentException(f"Slice {values['slice_id']} scanbound times must be non-negative")
        elif len(scanbound) > 1 and not all(i < j for i, j in zip(scanbound, scanbound[1:])):
            raise ExperimentException(f"Slice {values['slice_id']} scanbound times must be increasing")
        else:
            # Check if any scanbound times are shorter than the intt.
            tolerance = 1e-9
            if len(scanbound) == 1:
                if values['intt'] > (scanbound[0] * 1000 + tolerance):
                    raise ExperimentException(f"Slice {values['slice_id']} intt {values['intt']}ms longer than "
                                              f"scanbound time {scanbound[0]}s")
            else:
                for i in range(len(scanbound) - 1):
                    beam_time = (scanbound[i + 1] - scanbound[i]) * 1000
                    if values['intt'] > beam_time + tolerance:
                        raise ExperimentException(f"Slice {values['slice_id']} intt {values['intt']}ms longer than "
                                                  f"one of the scanbound times")
        return scanbound

    @validator('pulse_phase_offset')
    def check_pulse_phase_offset(cls, ppo, values):
        if ppo is default_callable:     # No value given
            return

        # Test the encoding fn with beam iterator of 0 and sequence num of 0. test the user's
        # phase encoding function on first beam (beam_iterator = 0) and first sequence
        # (sequence_number = 0)
        phase_encoding = ppo(0, 0, len(values['pulse_sequence']))
        if not isinstance(phase_encoding, np.ndarray):
            raise ExperimentException(f"Slice {values['slice_id']} Phase encoding return is not numpy array")
        else:
            if len(phase_encoding.shape) > 1:
                raise ExperimentException(f"Slice {values['slice_id']} Phase encoding return must be 1 dimensional")
            else:
                if phase_encoding.shape[0] != len(values['pulse_sequence']):
                    raise ExperimentException(f"Slice {values['slice_id']} Phase encoding return dimension must be "
                                              f"equal to number of pulses")
        return ppo

    @validator('clrfrqrange')
    def check_clrfrqrange(cls, clrfrqrange, values):
        if not clrfrqrange:
            return

        if len(clrfrqrange) != 2:
            raise ExperimentException('clrfrqrange must be an integer list of length = 2')

        errmsg = f"clrfrqrange must be between min and max tx frequencies " \
                 f"{(values['tx_minfreq'], values['tx_maxfreq'])} and rx frequencies " \
                 f"{(values['rx_minfreq'], values['rx_maxfreq'])} according to license and/or center frequencies / " \
                 f"sampling rates / transition bands, and must have lower frequency first."
        if clrfrqrange[0] >= clrfrqrange[1]:
            raise ExperimentException(errmsg)
        if (clrfrqrange[1] * 1000) >= values['tx_maxfreq'] or (clrfrqrange[1] * 1000) >= values['rx_maxfreq']:
            raise ExperimentException(errmsg)
        if (clrfrqrange[0] * 1000) <= values['tx_minfreq'] or (clrfrqrange[0] * 1000) <= values['rx_minfreq']:
            raise ExperimentException(errmsg)

        still_checking = True
        while still_checking:
            for freq_range in options.restricted_ranges:
                if freq_range[0] <= clrfrqrange[0] <= freq_range[1]:
                    if freq_range[0] <= clrfrqrange[1] <= freq_range[1]:
                        # the range is entirely within the restricted range.
                        raise ExperimentException(f'clrfrqrange is entirely within restricted range {freq_range}')
                    else:
                        print('Clrfrqrange will be modified because it is partially in a restricted range.')
                        # TODO Log warning, changing clrfrqrange because lower portion is in a restricted
                        #  frequency range.
                        clrfrqrange[0] = freq_range[1] + 1
                        # outside of restricted range now.
                        break  # we have changed the 'clrfrqrange' - must restart the
                        # check in case it's in another range.
                else:
                    # lower end is not in restricted frequency range.
                    if freq_range[0] <= clrfrqrange[1] <= freq_range[1]:
                        print('Clrfrqrange will be modified because it is partially in a restricted range.')
                        # TODO Log warning, changing clrfrqrange because upper portion is in a
                        #  restricted frequency range.
                        clrfrqrange[1] = freq_range[0] - 1
                        # outside of restricted range now.
                        break  # we have changed the 'clrfrqrange' - must restart the for loop
                        # checking in case it's in another range.
                    else:  # neither end of clrfrqrange is inside the restricted range but
                        # we should check if the range is inside the clrfrqrange.
                        if clrfrqrange[0] <= freq_range[0] <= clrfrqrange[1]:
                            print('There is a restricted range within the clrfrqrange - STOP.')
                            # TODO Log a warning that there is a restricted range in the middle
                            #  of the clrfrqrange that will be avoided OR could make this an
                            #  Error. Still need to implement clear frequency searching.
            else:  # no break, so no changes to the clrfrqrange
                still_checking = False

        return clrfrqrange

    @validator('freq')
    def check_freq(cls, freq, values):
        if values['rxonly']:  # RX only mode.
            # In this mode, freq is required.
            if (freq * 1000) >= values['rx_maxfreq'] or (freq * 1000) <= values['rx_minfreq']:
                errmsg = "freq must be a number (kHz) between rx min and max frequencies " \
                         f"{(values['rx_minfreq'] / 1.0e3, values['rx_maxfreq'] / 1.0e3)} for the radar license " \
                         f"and be within range given center frequency {values['rxctrfreq']} kHz, " \
                         f"sampling rate {values['rx_bandwidth'] / 1.0e3} kHz, and transition band " \
                         f"{values['transition_bandwidth'] / 1.0e3} kHz."
                raise ExperimentException(errmsg)

        else:  # TX-specific mode, without a clear frequency search.
            # In this mode, freq is required along with the other requirements.
            freq_error = False
            if (freq * 1000) >= values['tx_maxfreq'] or (freq * 1000) >= values['rx_maxfreq']:
                freq_error = True
            elif (freq * 1000) <= values['tx_minfreq'] or (freq * 1000) <= values['rx_minfreq']:
                freq_error = True

            if freq_error:
                errmsg = "freq must be a number (kHz) between tx min and max frequencies " \
                         f"{(values['tx_minfreq'] / 1.0e3, values['tx_maxfreq'] / 1.0e3)} and rx min and max " \
                         f"frequencies {(values['rx_minfreq'] / 1.0e3, values['rx_maxfreq'] / 1.0e3)} for the " \
                         f"radar license and be within range given center frequencies " \
                         f"(tx: {values['txctrfreq']} kHz, rx: {values['rxctrfreq']} kHz), sampling rates " \
                         f"(tx: {values['tx_bandwidth'] / 1.0e3} kHz, rx: {values['rx_bandwidth'] / 1.0e3} kHz), " \
                         f"and transition band ({values['transition_bandwidth'] / 1.0e3} kHz)."
                raise ExperimentException(errmsg)

            for freq_range in options.restricted_ranges:
                if freq_range[0] <= freq <= freq_range[1]:
                    raise ExperimentException(f"freq is within a restricted frequency range {freq_range}")

        return freq

    # Validators that set dynamic default values (depends on user-specified fields)

    @validator('xcf', always=True)
    def check_xcf(cls, xcf, values):
        if 'acf' in values and values['acf']:
            return xcf
        else:   # TODO log that no xcf will happen if acfs are not set.
            return False

    @validator('acfint', always=True)
    def check_acfint(cls, acfint, values):
        if 'acf' in values and values['acf']:
            return acfint
        else:   # TODO log that no acfint will happen if acfs are not set.
            return False

    @validator('range_sep', always=True)
    def check_range_sep(cls, range_sep, values):
        # This is the distance travelled by the wave in the length of the pulse, divided by
        # two because it's an echo (travels there and back). In km.
        correct_range_sep = values['pulse_len'] * 1.0e-9 * speed_of_light / 2.0  # km
        if 'acf' in values and values['acf'] and range_sep is not None:
            if not math.isclose(range_sep, correct_range_sep, abs_tol=0.01):
                errmsg = f"range_sep = {range_sep} was set incorrectly. range_sep will be overwritten " \
                         f"based on pulse_len, which must be equal to 1/rx_rate. The new range_sep = " \
                         f"{correct_range_sep}"
                # TODO change to logging
                print(errmsg)
        else:
            # TODO: log range_sep will not be used
            pass
        return correct_range_sep

    @validator('averaging_method', always=True)
    def check_averaging_method(cls, averaging_method, values):
        if 'acf' in values and values['acf']:
            return averaging_method or 'mean'
        else:   # TODO: log averaging_method will not be used
            return None

    @validator('lag_table', always=True)
    def check_lag_table(cls, lag_table, values):
        if 'acf' in values and values['acf']:
            if lag_table:
                # Check that lags are valid
                for lag in lag_table:
                    if not set(np.array(lag).flatten()).issubset(set(values['pulse_sequence'])):
                        errmsg = f"Lag {lag} not valid; One of the pulses does not exist in the sequence. Slice " \
                                 f"{values['slice_id']}"
                        raise ExperimentException(errmsg)
            else:
                # build lag table from pulse_sequence
                lag_table = list(itertools.combinations(values['pulse_sequence'], 2))
                lag_table.append([values['pulse_sequence'][0], values['pulse_sequence'][0]])  # lag 0
                # sort by lag number
                lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
                lag_table.append([values['pulse_sequence'][-1], values['pulse_sequence'][-1]])  # alternate lag 0
        else:   # TODO: log lag_table will not be used
            lag_table = []
        return lag_table
