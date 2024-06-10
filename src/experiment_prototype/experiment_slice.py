"""
    experiment_slice
    ~~~~~~~~~~~~~~~~~~~~
    This module contains the class for experiment slices, the base unit of a Borealis experiment.
    Each field of a slice has allowed types, and some have limits on the values they can take.
    The class also defines methods for complex validation of a slice, to confirm that all values
    make sense in the context of SuperDARN operations.

    :copyright: 2023 SuperDARN Canada
    :author: Remington Rohel
"""

# built-in
import copy
import inspect
import itertools
import math
from pathlib import Path

# third-party
import numpy as np
from pydantic.dataclasses import dataclass, Field
from pydantic import (
    validator,
    root_validator,
    conlist,
    conint,
    confloat,
    StrictBool,
    StrictInt,
    PositiveFloat,
)
from scipy.constants import speed_of_light
import structlog
from typing import Optional, Union, Literal, Callable

# local
from utils.options import Options
from experiment_prototype.experiment_utils.decimation_scheme import (
    DecimationScheme,
    create_default_scheme,
)

# Obtain the module name that imported this log_config
caller = Path(inspect.stack()[-1].filename)
module_name = caller.name.split(".")[0]
log = structlog.getLogger(module_name)

options = Options()

slice_key_set = frozenset(
    [
        "acf",
        "acfint",
        "align_sequences",
        "averaging_method",
        "beam_angle",
        "clrfrqrange",
        "comment",
        "cpid",
        "first_range",
        "txctrfreq",
        "rxctrfreq",
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
        "rx_intf_antennas",
        "rx_main_antennas",
        "rxonly",
        "rx_antenna_pattern",
        "scanbound",
        "seqoffset",
        "slice_id",
        "tau_spacing",
        "tx_antennas",
        "tx_antenna_pattern",
        "tx_beam_order",
        "wait_for_first_scanbound",
        "xcf",
    ]
)
hidden_key_set = frozenset(["clrfrqflag", "slice_interfacing"])
"""
These are used by the build_scans method (called from the experiment_handler every time the
experiment is run). If set by the user, the values will be overwritten and therefore ignored.
"""


def default_callable():
    """This function does nothing, and exists only as a default value for Callable fields in ExperimentSlice"""
    return


class SliceConfig:
    """
    This class configures pydantic options for ExperimentSlice.

    validate_assignment: Whether to run all validators for a field whenever field is changed (init or after init)
    validate_all: Whether to validate default fields
    extra: Whether to allow extra fields not defined when instantiating
    arbitrary_types_allowed: Whether to allow arbitrary types like user-defined classes (e.g. Options, DecimationScheme)
    """

    validate_assignment = True
    validate_all = True
    extra = "forbid"
    arbitrary_types_allowed = True


freq_hz = confloat(ge=options.min_freq, le=options.max_freq)
freq_khz = confloat(ge=options.min_freq / 1e3, le=options.max_freq / 1e3)
freq_float_hz = confloat(ge=options.min_freq, le=options.max_freq, strict=True)
freq_float_khz = confloat(
    ge=options.min_freq / 1e3, le=options.max_freq / 1e3, strict=True
)
freq_int_hz = conint(ge=options.min_freq, le=options.max_freq, strict=True)
freq_int_khz = conint(ge=options.min_freq / 1e3, le=options.max_freq / 1e3, strict=True)
beam_order_type = list[conint(ge=0, strict=True)]


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
        pattern makes sense. Typically rx_beam_order is just in order (scanning W to E or E to W), ie.
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
    rxctrfreq *defaults*
        Center frequency, in kHz, used to mix to baseband.
        Since this requires tuning time to set, it is the user's responsibility to ensure that the
        re-tuning time does not detract from the experiment implementation. Tuning time is set in
        the usrp_driver.cpp script and changes to the time will require recompiling of the code.
    rx_intf_antennas *defaults*
        The antennas to receive on in interferometer array, default is all antennas given max number
        from config.
    rx_main_antennas *defaults*
        The antennas to receive on in main array, default is all antennas given max number from config.
    rx_antenna_pattern *defaults*
        Experiment-defined function which returns a complex weighting factor of magnitude <= 1 for each
        beam direction scanned in the experiment. The return value of the function must be an array of
        size [beam_angle, antenna_num]. This function allows for custom beamforming of the receive
        antennas for borealis processing of antenna iq to rawacf.
    scanbound *defaults*
        A list of seconds past the minute for averaging periods in a scan to align to. Defaults to None,
        not required. If one slice in an experiment has a scanbound, they all must.
    seqoffset *defaults*
        offset in us that this slice's sequence will begin at, after the start of the sequence. This is
        intended for CONCURRENT interfacing, when you want multiple slice's pulses in one sequence you
        can offset one slice's sequence from the other by a certain time value so as to not run both
        frequencies in the same pulse, etc. Default is 0 offset.
    txctrfreq *defaults*
        Center frequency, in kHz, for the USRP to mix the samples with.
        Since this requires tuning time to set, it is the user's responsibility to ensure that the
        re-tuning time does not detract from the experiment implementation. Tuning time is set in
        the usrp_driver.cpp script and changes to the time will require recompiling of the code.
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
    tx_bandwidth: float
    rx_bandwidth: float
    transition_bandwidth: float

    # These fields can be specified in exp_slice_dict, subject to some conditions. Some may have dynamic default values.
    slice_id: conint(ge=0, strict=True)
    beam_angle: conlist(
        Union[confloat(strict=True), conint(strict=True)], unique_items=True
    )
    cpid: StrictInt
    first_range: Union[confloat(ge=0), conint(ge=0)]
    num_ranges: conint(gt=0, strict=True)
    tau_spacing: conint(ge=options.min_tau_spacing_length, strict=True)
    pulse_len: conint(ge=options.min_pulse_length, strict=True)
    pulse_sequence: conlist(conint(ge=0, strict=True), unique_items=True)
    rx_beam_order: list[Union[beam_order_type, conint(ge=0, strict=True)]]

    # Frequency rx and tx limits are dependent on the tx and rx center frequencies. Since the center freq
    # parameter is defined by slice, the max and min rx frequencies must be determined after center freq validation
    txctrfreq: Optional[freq_khz] = None
    rxctrfreq: Optional[freq_khz] = None
    freq: Optional[freq_khz] = None

    # These fields have default values. Some have specification requirements in conjunction with each other
    # e.g. one of intt or intn must be specified.
    rxonly: Optional[StrictBool] = False
    tx_antennas: Optional[
        conlist(
            conint(ge=0, lt=options.main_antenna_count, strict=True),
            max_items=options.main_antenna_count,
            unique_items=True,
        )
    ] = None
    rx_main_antennas: Optional[
        conlist(
            conint(ge=0, lt=options.main_antenna_count, strict=True),
            max_items=options.main_antenna_count,
            unique_items=True,
        )
    ] = None
    rx_intf_antennas: Optional[
        conlist(
            conint(ge=0, lt=options.intf_antenna_count, strict=True),
            max_items=options.intf_antenna_count,
            unique_items=True,
        )
    ] = None
    tx_antenna_pattern: Optional[Callable] = default_callable
    rx_antenna_pattern: Optional[Callable] = default_callable
    tx_beam_order: Optional[beam_order_type] = Field(default_factory=list)
    intt: Optional[confloat(ge=0)] = None
    scanbound: Optional[list[confloat(ge=0)]] = Field(default_factory=list)
    pulse_phase_offset: Optional[Callable] = default_callable
    clrfrqrange: Optional[conlist(freq_int_khz, min_items=2, max_items=2)] = None
    clrfrqflag: StrictBool = Field(init=False)
    decimation_scheme: DecimationScheme = Field(default_factory=create_default_scheme)

    acf: Optional[StrictBool] = False
    acfint: Optional[StrictBool] = False
    align_sequences: Optional[StrictBool] = False
    averaging_method: Optional[Literal["mean", "median"]] = "mean"
    comment: Optional[str] = ""
    intn: Optional[conint(ge=0, strict=True)] = None
    lag_table: Optional[list[list[StrictInt]]] = Field(default_factory=list)
    range_sep: Optional[PositiveFloat] = Field(init=False)
    seqoffset: Optional[conint(ge=0, strict=True)] = 0
    wait_for_first_scanbound: Optional[StrictBool] = False
    xcf: Optional[bool] = False

    # Validators which check that all mutually exclusive sets of fields have one option set

    @root_validator(pre=True)
    def check_tx_specifier(cls, values):
        if "tx_antenna_pattern" not in values and "tx_beam_order" in values:
            raise ValueError(
                f"tx_beam_order must be specified if tx_antenna_pattern specified. Slice: "
                f"{values['slice_id']}"
            )
        elif "tx_beam_order" in values and "rxonly" in values and values["rxonly"]:
            raise ValueError(
                f"rxonly specified as True but tx_beam_order specified. Slice: {values['slice_id']}"
            )
        elif (
            "tx_beam_order" not in values
            and "rxonly" in values
            and values["rxonly"] is False
        ):
            raise ValueError(
                f"rxonly specified as False but tx_beam_order not given. Slice: {values['slice_id']}"
            )
        return values

    @root_validator(pre=True)
    def check_intt_intn(cls, values):
        if not values["intt"] and not values["intn"]:
            raise ValueError(
                f"Slice must specify either an intn (unitless) or intt in ms. Slice: {values['slice_id']}"
            )
        elif not values["intt"] and not values["intn"]:
            raise ValueError(
                f"intn is set in experiment slice but will not be used due to intt. Slice: "
                f"{values['slice_id']}"
            )
        return values

    @root_validator(pre=True)
    def check_freq_clrfrqrange(cls, values):
        if "clrfrqrange" in values and values["clrfrqrange"]:
            values["clrfrqflag"] = True
            if "freq" in values and values["freq"]:
                log.info(
                    f"Slice parameter 'freq' removed as 'clrfrqrange' takes precedence. If this is not desired,"
                    f"remove 'clrfrqrange' parameter from experiment. Slice: {values['slice_id']}"
                )
        elif "freq" in values and values["freq"]:
            values["clrfrqflag"] = False
        else:
            raise ValueError(
                f"A freq or clrfrqrange must be specified in a slice. Slice: {values['slice_id']}"
            )
        return values

    # Validate that a list is increasing

    @validator("pulse_sequence", "beam_angle")
    def check_list_increasing(cls, v_list):
        if not all(x < y for x, y in zip(v_list, v_list[1:])):
            raise ValueError(f"not increasing: {v_list}")
        return v_list

    # Validators that depend on other previously-validated fields

    @validator("intt")
    def check_intt(cls, intt, values):
        if not intt:  # Not provided
            return

        # check intn and intt make sense given tau_spacing, and pulse_sequence.
        # Sequence length is length of pulse sequence plus the scope sync delay time.
        # TODO: this is an old check and seqtime now set in sequences class, update.
        if (
            "tau_spacing" in values
            and "pulse_sequence" in values
            and "num_ranges" in values
            and "pulse_len" in values
        ):
            seq_len = (
                values["tau_spacing"] * (values["pulse_sequence"][-1])
                + (values["num_ranges"] + 19 + 10) * values["pulse_len"]
            )  # us
            if seq_len > (intt * 1000):  # seq_len in us, intt in ms
                raise ValueError(
                    f"Slice {values['slice_id']}: pulse sequence is too long for integration time given"
                )
        return intt

    @validator("tx_antennas")
    def check_tx_antennas(cls, tx_antennas):
        if tx_antennas is None:
            tx_antennas = [i for i in options.tx_main_antennas]
        for ant in tx_antennas:
            if ant not in options.tx_main_antennas:
                raise ValueError(f"TX antenna {ant} not specified in config file")
        tx_antennas.sort()
        return tx_antennas

    @validator("rx_main_antennas")
    def check_rx_main_antennas(cls, rx_main_antennas):
        if rx_main_antennas is None:
            rx_main_antennas = [i for i in options.rx_main_antennas]
        if len(rx_main_antennas) == 0:
            raise ValueError("Must have at least one main antenna for RX")
        for ant in rx_main_antennas:
            if ant not in options.rx_main_antennas:
                raise ValueError(f"RX main antenna {ant} not specified in config file")
        rx_main_antennas.sort()
        return rx_main_antennas

    @validator("rx_intf_antennas")
    def check_rx_intf_antennas(cls, rx_intf_antennas):
        if rx_intf_antennas is None:
            return [i for i in options.rx_intf_antennas]
        for ant in rx_intf_antennas:
            if ant not in options.rx_intf_antennas:
                raise ValueError(f"RX intf antenna {ant} not specified in config file")
        rx_intf_antennas.sort()
        return rx_intf_antennas

    @validator("tx_antenna_pattern")
    def check_tx_antenna_pattern(cls, tx_antenna_pattern, values):
        if tx_antenna_pattern is default_callable:  # No value given
            return

        antenna_pattern = tx_antenna_pattern(
            values["freq"], values["tx_antennas"], options.main_antenna_spacing
        )
        if not isinstance(antenna_pattern, np.ndarray):
            raise ValueError(
                f"Slice {values['slice_id']} tx antenna pattern return is not a numpy array"
            )
        else:
            if len(antenna_pattern.shape) != 2:
                raise ValueError(
                    f"Slice {values['slice_id']} tx antenna pattern return shape "
                    f"{antenna_pattern.shape} must be 2-dimensional"
                )
            elif antenna_pattern.shape[1] != options.main_antenna_count:
                raise ValueError(
                    f"Slice {values['slice_id']} tx antenna pattern return 2nd dimension "
                    f"({antenna_pattern.shape[1]}) must be equal to number of main antennas "
                    f"({options.main_antenna_count})"
                )
            antenna_pattern_mag = np.abs(antenna_pattern)
            if np.argwhere(antenna_pattern_mag > 1.0).size > 0:
                raise ValueError(
                    f"Slice {values['slice_id']} tx antenna pattern return must not have any "
                    f"values with a magnitude greater than 1"
                )
        return tx_antenna_pattern

    @validator("rx_antenna_pattern")
    def check_rx_antenna_pattern(cls, rx_antenna_pattern, values):
        if rx_antenna_pattern is default_callable:  # No value given
            return

        # Main and interferometer patterns
        antenna_pattern = [
            rx_antenna_pattern(
                values["beam_angle"],
                values["freq"],
                options.main_antenna_count,
                options.main_antenna_spacing,
            ),
            rx_antenna_pattern(
                values["beam_angle"],
                values["freq"],
                options.intf_antenna_count,
                options.intf_antenna_spacing,
                offset=-100,
            ),
        ]
        for index in range(0, len(antenna_pattern)):
            if index == 0:
                pattern = "main"
                antenna_num = len(options.rx_main_antennas)
            else:
                pattern = "interferometer"
                antenna_num = len(options.rx_intf_antennas)
            if not isinstance(antenna_pattern[index], np.ndarray):
                raise ValueError(
                    f"Slice {values['slice_id']} {pattern} array rx antenna pattern return is "
                    f"not a numpy array"
                )
            else:
                if antenna_pattern[index].shape != (
                    len(values["beam_angle"]),
                    antenna_num,
                ):
                    raise ValueError(
                        f"Slice {values['slice_id']} {pattern} array must be the same shape as"
                        f" ([beam angle], [antenna_count])"
                    )
            antenna_pattern_mag = np.abs(antenna_pattern[index])
            if np.argwhere(antenna_pattern_mag > 1.0).size > 0:
                raise ValueError(
                    f"Slice {values['slice_id']} {pattern} array rx antenna pattern return must not have "
                    f"any values with a magnitude greater than 1"
                )
        return rx_antenna_pattern

    @validator("rx_beam_order", each_item=True)
    def check_rx_beam_order(cls, rx_beam, values):
        if "beam_angle" in values:
            if isinstance(rx_beam, list):
                for beamnum in rx_beam:
                    if beamnum >= len(values["beam_angle"]):
                        raise ValueError(
                            f"Beam number {beamnum} could not index in beam_angle list of length "
                            f"{len(values['beam_angle'])}. Slice: {values['slice_id']}"
                        )
            else:
                if rx_beam >= len(values["beam_angle"]):
                    raise ValueError(
                        f"Beam number {rx_beam} could not index in beam_angle list of length "
                        f"{len(values['beam_angle'])}. Slice: {values['slice_id']}"
                    )
        return rx_beam

    @validator("tx_beam_order")
    def check_tx_beam_order(cls, tx_beam_order, values):
        if not tx_beam_order:  # Empty list, was not specified
            return

        if "rx_beam_order" in values and len(tx_beam_order) != len(
            values["rx_beam_order"]
        ):
            raise ValueError(
                f"tx_beam_order does not have same length as rx_beam_order. Slice: {values['slice_id']}"
            )
        for element in tx_beam_order:
            if (
                "beam_angle" in values
                and element >= len(values["beam_angle"])
                and (
                    "tx_antenna_pattern" not in values
                    or not values["tx_antenna_pattern"]
                )
            ):
                raise ValueError(
                    f"Beam number {element} in tx_beam_order could not index in beam_angle list of "
                    f"length {len(values['beam_angle'])}. Slice: {values['slice_id']}"
                )

        num_beams = None
        if "tx_antenna_pattern" in values and values["tx_antenna_pattern"]:
            antenna_pattern = values["tx_antenna_pattern"](
                values["freq"], values["tx_antennas"], options.main_antenna_spacing
            )
            if isinstance(antenna_pattern, np.ndarray):
                num_beams = antenna_pattern.shape[0]
        elif "beam_angle" in values:
            num_beams = len(values["beam_angle"])
        if num_beams:
            for bmnum in tx_beam_order:
                if bmnum >= num_beams:
                    raise ValueError(
                        f"Slice {values['slice_id']} scan tx beam number {bmnum} DNE"
                    )
        if "tx_antennas" in values and len(values["tx_antennas"]) == 0:
            raise ValueError(
                "Must have TX antennas specified if tx_beam_order specified"
            )

        return tx_beam_order

    @validator("scanbound")
    def check_scanbound(cls, scanbound, values):
        if not scanbound:  # No scanbound defined
            return

        if "intt" not in values or not values["intt"]:
            raise ValueError(
                f"Slice {values['slice_id']} must have intt enabled to use scanbound"
            )
        elif any(i < 0 for i in scanbound):
            raise ValueError(
                f"Slice {values['slice_id']} scanbound times must be non-negative"
            )
        elif len(scanbound) > 1 and not all(
            i < j for i, j in zip(scanbound, scanbound[1:])
        ):
            raise ValueError(
                f"Slice {values['slice_id']} scanbound times must be increasing"
            )
        elif "intt" in values and values["intt"]:
            # Check if any scanbound times are shorter than the intt.
            tolerance = 1e-9
            if len(scanbound) == 1:
                if values["intt"] > (scanbound[0] * 1000 + tolerance):
                    raise ValueError(
                        f"Slice {values['slice_id']} intt {values['intt']}ms longer than "
                        f"scanbound time {scanbound[0]}s"
                    )
            else:
                for i in range(len(scanbound) - 1):
                    beam_time = (scanbound[i + 1] - scanbound[i]) * 1000
                    if values["intt"] > beam_time + tolerance:
                        raise ValueError(
                            f"Slice {values['slice_id']} intt {values['intt']}ms longer than "
                            f"one of the scanbound times"
                        )
        return scanbound

    @validator("pulse_phase_offset")
    def check_pulse_phase_offset(cls, ppo, values):
        if ppo is default_callable:  # No value given
            return

        # Test the encoding fn with beam iterator of 0 and sequence num of 0. test the user's
        # phase encoding function on first beam (beam_iterator = 0) and first sequence
        # (sequence_number = 0)
        phase_encoding = ppo(0, 0, len(values["pulse_sequence"]))
        if not isinstance(phase_encoding, np.ndarray):
            raise ValueError(
                f"Slice {values['slice_id']} Phase encoding return is not numpy array"
            )
        else:
            if len(phase_encoding.shape) > 1:
                raise ValueError(
                    f"Slice {values['slice_id']} Phase encoding return must be 1 dimensional"
                )
            else:
                if phase_encoding.shape[0] != len(values["pulse_sequence"]):
                    raise ValueError(
                        f"Slice {values['slice_id']} Phase encoding return dimension must be equal to "
                        f"number of pulses"
                    )
        return ppo

    @validator("clrfrqrange")
    def check_clrfrqrange(cls, clrfrqrange, values):
        if not clrfrqrange:
            return clrfrqrange

        if clrfrqrange[0] >= clrfrqrange[1]:
            raise ValueError(
                f"Slice {values['slice_id']} clrfrqrange must be between min and max tx frequencies "
                f"and rx frequencies according to license and/or center "
                f"frequencies / sampling rates / transition bands, and must have lower frequency first."
            )

        still_checking = True
        while still_checking:
            for freq_range in options.restricted_ranges:
                if freq_range[0] <= clrfrqrange[0] <= freq_range[1]:
                    if freq_range[0] <= clrfrqrange[1] <= freq_range[1]:
                        # the range is entirely within the restricted range.
                        raise ValueError(
                            f"clrfrqrange is entirely within restricted range {freq_range}. Slice: "
                            f"{values['slice_id']}"
                        )
                    else:
                        log.warning(
                            f"Slice: {values['slice_id']} clrfrqrange will be modified because it is partially "
                            f"in a restricted range."
                        )
                        clrfrqrange[0] = freq_range[1] + 1
                        # outside of restricted range now.
                        break  # we have changed the 'clrfrqrange' - must restart the
                        # check in case it's in another range.
                else:
                    # lower end is not in restricted frequency range.
                    if freq_range[0] <= clrfrqrange[1] <= freq_range[1]:
                        log.warning(
                            f"Slice: {values['slice_id']} clrfrqrange will be modified because it is partially "
                            f"in a restricted range."
                        )
                        clrfrqrange[1] = freq_range[0] - 1
                        # outside of restricted range now.
                        break  # we have changed the 'clrfrqrange' - must restart the for loop
                        # checking in case it's in another range.
                    else:  # neither end of clrfrqrange is inside the restricted range but
                        # we should check if the range is inside the clrfrqrange.
                        if clrfrqrange[0] <= freq_range[0] <= clrfrqrange[1]:
                            log.warning(
                                f"There is a restricted range within the clrfrqrange - STOP. Slice: "
                                f"{values['slice_id']}"
                            )
                            # TODO: Error. Still need to implement clear frequency searching.
            else:  # no break, so no changes to the clrfrqrange
                still_checking = False

        return values

    @validator("txctrfreq", always=True)
    def check_txctrfreq(cls, txctrfreq):
        if not txctrfreq:
            txctrfreq = 12000.0  # Default value when not set

        # Note - txctrfreq set here and modify the actual center frequency to a
        # multiple of the clock divider that is possible by the USRP - this default value set
        # here is not exact (center freq is never exactly 12 MHz).

        # convert from kHz to Hz to get correct clock divider. Return the result back in kHz.
        clock_multiples = options.usrp_master_clock_rate / 2**32
        clock_divider = math.ceil(txctrfreq * 1e3 / clock_multiples)
        txctrfreq = (clock_divider * clock_multiples) / 1e3

        return txctrfreq

    @validator("rxctrfreq", always=True)
    def check_rxctrfreq(cls, rxctrfreq):
        if not rxctrfreq:
            rxctrfreq = 12000.0  # Default value when not set

        # Note - rxctrfreq set here and modify the actual center frequency to a
        # multiple of the clock divider that is possible by the USRP - this default value set
        # here is not exact (center freq is never exactly 12 MHz).

        # convert from kHz to Hz to get correct clock divider. Return the result back in kHz.
        clock_multiples = options.usrp_master_clock_rate / 2**32
        clock_divider = math.ceil(rxctrfreq * 1e3 / clock_multiples)
        rxctrfreq = (clock_divider * clock_multiples) / 1e3

        return rxctrfreq

    @validator("freq")
    def check_freq(cls, freq, values):
        for freq_range in options.restricted_ranges:
            if freq_range[0] <= freq <= freq_range[1]:
                raise ValueError(
                    f"freq is within a restricted frequency range {freq_range}"
                )

        # max frequency is defined as [center freq] + [bandwidth / 2] - [bandwidth * 0.15]
        # min frequency is defined as [center freq] - [bandwidth / 2] + [bandwidth * 0.15]
        # [bandwidth * 0.15] is the transition bandwidth. This was set a 750 kHz originally
        # but for smaller bandwidth this value is too large. For the typical operating
        # bandwidth of 5 MHz, the calculated transition bandwidth here will be 750 kHz

        # TODO review issue #195

        if "rxctrfreq" in values:
            rx_maxfreq = (
                values["rxctrfreq"] * 1000
                + (values["rx_bandwidth"] / 2.0)
                - (values["rx_bandwidth"] * 0.15)
            )
            rx_minfreq = (
                values["rxctrfreq"] * 1000
                - (values["rx_bandwidth"] / 2.0)
                + (values["rx_bandwidth"] * 0.15)
            )
            rx_center = values["rxctrfreq"]
        else:
            rx_maxfreq = options.max_freq
            rx_minfreq = options.min_freq
            rx_center = 0

        if "txctrfreq" in values:
            tx_maxfreq = (
                values["txctrfreq"] * 1000
                + (values["tx_bandwidth"] / 2.0)
                - (values["tx_bandwidth"] * 0.15)
            )
            tx_minfreq = (
                values["txctrfreq"] * 1000
                - (values["tx_bandwidth"] / 2.0)
                + (values["tx_bandwidth"] * 0.15)
            )
            tx_center = values["txctrfreq"]
        else:
            tx_maxfreq = options.max_freq
            tx_minfreq = options.min_freq
            tx_center = 0

        # Frequency must be withing bandwidth of rx and tx center frequency
        if (freq > rx_maxfreq / 1000) or (freq < rx_minfreq / 1000):
            raise ValueError(
                f"Slice frequency is outside bandwidth around rx center frequency {int(rx_center)}"
            )
        if (freq > tx_maxfreq / 1000) or (freq < tx_minfreq / 1000):
            raise ValueError(
                f"Slice frequency is outside bandwidth around tx center frequency {int(tx_center)}"
            )

        # Frequency cannot be set to the rx or tx center frequency (100kHz bandwidth around center freqs)
        if abs(freq - rx_center) < 50:
            raise ValueError(
                f"Slice frequency cannot be within 50kHz of rx center frequency {int(rx_center)}"
            )
        if abs(freq - tx_center) < 50:
            raise ValueError(
                f"Slice frequency cannot be within 50kHz of tx center frequency {int(tx_center)}"
            )

        return freq

    @validator("decimation_scheme")
    def check_decimation_rates(cls, decimation_scheme, values):
        # check that number of stages is not too large
        if len(decimation_scheme.stages) > options.max_filtering_stages:
            errmsg = (
                f"Number of decimation stages ({len(decimation_scheme.stages)}) is greater than max"
                f" available {options.max_filtering_stages}"
            )
            raise ValueError(errmsg)

        # Check that the rx_bandwidth matches input rate of the DecimationScheme
        input_rate = decimation_scheme.input_rates[0]
        if input_rate != values["rx_bandwidth"]:
            raise ValueError(
                f"decimation_scheme input data rate {input_rate} does not match rx_bandwidth "
                f"{values['rx_bandwidth']}"
            )

        return decimation_scheme

    # Validators that set dynamic default values (depends on user-specified fields)

    @validator("xcf", always=True)
    def check_xcf(cls, xcf, values):
        if "acf" not in values or not values["acf"]:
            xcf = False
            log.info(
                f"XCF defaulted to False as ACF not set. Slice: {values['slice_id']}"
            )
            return False
        if (
            xcf
            and "rx_intf_antennas" in values
            and len(values["rx_intf_antennas"]) == 0
        ):
            raise ValueError("XCF set to True but no interferometer antennas present")
        return xcf

    @validator("acfint", always=True)
    def check_acfint(cls, acfint, values):
        if "acf" not in values or not values["acf"]:
            acfint = False
            log.info(
                f"ACFINT defaulted to False as ACF not set. Slice: {values['slice_id']}"
            )
        if (
            acfint
            and "rx_intf_antennas" in values
            and len(values["rx_intf_antennas"]) == 0
        ):
            raise ValueError(
                "ACFINT set to True but no interferometer antennas present"
            )
        return acfint

    @validator("range_sep", always=True)
    def check_range_sep(cls, range_sep, values):
        if "pulse_len" in values:
            # This is the distance travelled by the wave in the length of the pulse, divided by
            # two because it's an echo (travels there and back). In km.
            correct_range_sep = (
                values["pulse_len"] * 1.0e-9 * speed_of_light / 2.0
            )  # km
            if "acf" in values and values["acf"] and range_sep is not None:
                if not math.isclose(range_sep, correct_range_sep, abs_tol=0.01):
                    errmsg = (
                        f"range_sep = {range_sep} was set incorrectly. range_sep will be overwritten "
                        f"based on pulse_len, which must be equal to 1/rx_rate. The new range_sep = "
                        f"{correct_range_sep}"
                    )
                    log.warning(errmsg)
            range_sep = correct_range_sep
        return range_sep

    @validator("averaging_method", always=True)
    def check_averaging_method(cls, averaging_method, values):
        if "acf" in values and values["acf"]:
            return averaging_method or "mean"
        else:
            log.info(
                f"Averaging method unset as ACF not set. Slice: {values['slice_id']}"
            )
            return None

    @validator("lag_table", always=True)
    def check_lag_table(cls, lag_table, values):
        if "acf" in values and values["acf"] and "pulse_sequence" in values:
            if lag_table:
                # Check that lags are valid
                for lag in lag_table:
                    if not set(np.array(lag).flatten()).issubset(
                        set(values["pulse_sequence"])
                    ):
                        raise ValueError(
                            f"Lag {lag} not valid; One of the pulses does not exist in the sequence. "
                            f"Slice: {values['slice_id']}"
                        )
            else:
                # build lag table from pulse_sequence
                lag_table = list(itertools.combinations(values["pulse_sequence"], 2))
                lag_table.append(
                    [values["pulse_sequence"][0], values["pulse_sequence"][0]]
                )  # lag 0
                # sort by lag number
                lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
                lag_table.append(
                    [values["pulse_sequence"][-1], values["pulse_sequence"][-1]]
                )  # alternate lag 0
        else:
            log.info(f"Lag table unused as ACF not set. Slice: {values['slice_id']}")
            lag_table = []
        return lag_table

    # Validators for when a check requires that an early-validated field and later-validated field have both been
    # validated. E.g. could not validate pulse_len fully off the bat because it depends on acf, which gets validated
    # later.

    @root_validator
    def check_tau_spacing(cls, values):
        # TODO : tau_spacing needs to be an integer multiple of pulse_len in ros - is there a max ratio
        #  allowed for pulse_len/tau_spacing ? Add this check and add check for each slice's tx duty-cycle
        #  and make sure we aren't transmitting the entire time after combination with all slices
        if "tau_spacing" not in values or "decimation_scheme" not in values:
            return values

        tau_spacing = values["tau_spacing"]
        filter_scheme = values["decimation_scheme"]
        dm_rate = 1
        for stage in filter_scheme.stages:
            dm_rate *= stage.dm_rate
        output_rx_rate = values["rx_bandwidth"] / dm_rate
        if not math.isclose((tau_spacing * output_rx_rate % 1.0), 0.0, abs_tol=0.0001):
            raise ValueError(
                f"Slice {values['slice_id']} correlation lags will be off because tau_spacing "
                f"{tau_spacing} us is not a multiple of the output rx sampling period "
                f"(1/output_rx_rate {output_rx_rate:.3f} Hz)."
            )
        return values

    @root_validator
    def check_pulse_len(cls, values):
        if "pulse_len" not in values:
            return values

        pulse_len = values["pulse_len"]
        if "tau_spacing" in values and pulse_len > values["tau_spacing"]:
            raise ValueError(
                f"Slice {values['slice_id']} pulse length greater than tau_spacing"
            )
        if pulse_len <= 2 * options.pulse_ramp_time * 1.0e6:
            raise ValueError(f"Slice {values['slice_id']} pulse length too small")

        if "acf" in values and values["acf"] and "decimation_scheme" in values:
            filter_scheme = values["decimation_scheme"]
            dm_rate = 1
            for stage in filter_scheme.stages:
                dm_rate *= stage.dm_rate
            output_rx_rate = values["rx_bandwidth"] / dm_rate
            # The below check is an assumption that is made during acf calculation
            # (1 output received sample = 1 range separation)
            if not math.isclose(
                values["pulse_len"],
                (1 / output_rx_rate * 1e6),
                abs_tol=0.0000001,
            ):
                raise ValueError(
                    f"For an experiment slice with real-time acfs, pulse length must be equal (within 1 "
                    f"us) to 1/output_rx_rate to make acfs valid. Current pulse length is "
                    f"{values['pulse_len']} us, output rate is {output_rx_rate:.3f} Hz. "
                    f"Slice: {values['slice_id']}"
                )
        return values

    # Post-initialization validator

    def check_slice(self):
        """
        Checks and verifies all fields at any time after instantiation.
        """
        new_exp = copy.deepcopy(self.__dict__)
        to_pop = [k for k, v in new_exp.items() if v is None]
        to_pop.extend(
            ["__pydantic_initialised__", "slice_interfacing"]
        )  # Remove fields that we expect to have now
        for k in to_pop:
            new_exp.pop(
                k, None
            )  # default None just in case slice_interfacing isn't set yet
        ExperimentSlice(**new_exp)  # Will raise exception if something is amiss
