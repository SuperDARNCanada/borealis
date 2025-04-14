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
import inspect
import itertools
import math
from pathlib import Path

# third-party
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import (
    field_validator,
    model_validator,
    Field,
    conlist,
    Strict,
    StrictBool,
    StrictInt,
    PositiveFloat,
    NonNegativeFloat,
    StrictFloat,
    AfterValidator,
)
from scipy.constants import speed_of_light
import structlog
from typing import List, Optional, Union, Literal, Callable, Hashable, TypeVar

# local
from utils.options import Options
from experiment_prototype.experiment_utils.decimation_scheme import (
    DecimationScheme,
    create_default_scheme,
    create_default_cfs_scheme,
)
from typing_extensions import Annotated

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
        "cfs_range",
        "cfs_fft_n",
        "cfs_freq_res",
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
hidden_key_set = frozenset(
    ["cfs_flag", "slice_interfacing", "tx_freq_bounds", "rx_freq_bounds"]
)
"""
These are used by the build_scans method (called from the experiment_handler every time the
experiment is run). If set by the user, the values will be overwritten and therefore ignored.
"""


class SliceConfig:
    """
    This class configures pydantic options for ExperimentSlice.

    validate_assignment: Whether to run all validators for a field whenever field is changed (init or after init)
    validate_all: Whether to validate default fields
    extra: Whether to allow extra fields not defined when instantiating
    arbitrary_types_allowed: Whether to allow arbitrary types like user-defined classes (e.g. Options, DecimationScheme)
    """

    validate_assignment = True
    validate_default = True
    extra = "forbid"
    arbitrary_types_allowed = True


T = TypeVar("T", bound=Hashable)


def _validate_unique_list(v: list[T]) -> list[T]:
    """Validates that a list contains unique items"""
    if len(v) != len(set(v)):
        raise ValueError("List must be unique")
    return v


def check_list_increasing(v: list[T]):
    """Validates that a list has increasing entries"""
    if not all(x < y for x, y in zip(v, v[1:])):
        raise ValueError("List must have increasing values")
    return v


def check_list_sorted(v: list[T]):
    """Validates that a list is sorted"""
    if not all(x <= y for x, y in zip(v, v[1:])):
        raise ValueError("List must be sorted in increasing order")
    return v


freq_hz = Annotated[float, Field(ge=options.min_freq, le=options.max_freq)]
freq_khz = Annotated[float, Field(ge=options.min_freq / 1e3, le=options.max_freq / 1e3)]
freq_float_hz = Annotated[
    float, Field(ge=options.min_freq, le=options.max_freq, strict=True)
]
freq_float_khz = Annotated[
    float, Field(ge=options.min_freq / 1e3, le=options.max_freq / 1e3, strict=True)
]
freq_int_hz = Annotated[
    int, Field(ge=options.min_freq, le=options.max_freq, strict=True)
]
freq_int_khz = Annotated[
    int, Field(ge=options.min_freq / 1e3, le=options.max_freq / 1e3, strict=True)
]

positive_int = Annotated[int, Field(gt=0), Strict()]
non_neg_int = Annotated[int, Field(ge=0), Strict()]
positive_float = Annotated[float, Field(gt=0), Strict()]
non_neg_float = Annotated[float, Field(ge=0), Strict()]


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
    cfs_range *required or freq required*
        range for clear frequency search, should be a list of length = 2, [min_freq, max_freq] in kHz.
    first_range *required*
        first range gate, in km
    freq *required or cfs_range required*
        transmit/receive frequency, in kHz. Note if you specify cfs_range it won't be used.
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
    cfs_duration *defaults*
        Amount of time a clear frequency search will listen for in ms.
    cfs_scheme *defaults*
        Decimation scheme to be used in analyzing data collected during a clear frequency search when
        determining transmit frequencies for CFS experiment slices
    cfs_stable_time *defaults*
        Amount of time in seconds clear frequency search will not change the slice frequencies for.
        Ensures a minimum stable time, but does not force a change in frequency once the stable time
        has elapsed.
    cfs_pwr_threshold *defaults*
         Difference in power (in dB) that clear frequency search must see in the measured frequencies
         before it changes the cfs slice frequencies. This can be trigered either when another
         frequency in the cfs range is found to have a lower power than the currently set frequency, or
         when the currently set frequency has increased in power by the threshold value since it was
         selected as an operating frequency.
    cfs_fft_n *defaults*
        Sets the number of elements used in the fft when processing clear frequency search results.
        Determines the frequency resolution of the processing following the formula;
        frequency resolution = (rx rate / total decimation rate) / cfs fft n value
    cfs_freq_res *defaults*
        Defines the desired frequency resolution of clear frequency search results. The frequency
        resolution is used to calculate the number of elements used in the fft when processing and
        sets that value to cfs_fft_n. Note the actual frequency resolution set will differ based on
        the nearest integer value of n corresponding to the requested resolution.
    cfs_always_run *defaults*
        If true always run the cfs sequence, otherwise only run after cfs_stable_time has expired
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

    cfs_flag *read-only*
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
    tx_bandwidth: positive_float
    rx_bandwidth: positive_float
    transition_bandwidth: positive_float

    # These fields can be specified in exp_slice_dict, subject to some conditions. Some may have dynamic default values.
    slice_id: non_neg_int
    beam_angle: Annotated[List[StrictFloat], AfterValidator(check_list_increasing)]
    cpid: StrictInt
    first_range: NonNegativeFloat
    num_ranges: non_neg_int
    tau_spacing: Annotated[int, Field(ge=options.min_tau_spacing_length, strict=True)]
    pulse_len: Annotated[int, Field(ge=options.min_pulse_length, strict=True)]
    pulse_sequence: Annotated[
        List[Annotated[int, Field(ge=0, strict=True)]],
        AfterValidator(check_list_increasing),
    ]
    rx_beam_order: List[Union[List[non_neg_int], non_neg_int]]

    # Frequency rx and tx limits are dependent on the tx and rx center frequencies. Since the center freq
    # parameter is defined by slice, the max and min rx frequencies must be determined after center freq validation
    txctrfreq: Optional[freq_khz] = None
    rxctrfreq: Optional[freq_khz] = None
    tx_freq_bounds: Optional[tuple] = (options.min_freq / 1000, options.max_freq / 1000)
    rx_freq_bounds: Optional[tuple] = (options.min_freq / 1000, options.max_freq / 1000)
    rxonly: Optional[StrictBool] = False
    freq: Optional[freq_khz] = None

    acf: Optional[StrictBool] = False
    align_sequences: Optional[StrictBool] = False
    averaging_method: Optional[Literal["mean", "median"]] = "mean"
    comment: Optional[str] = ""
    intn: Optional[non_neg_int] = None
    seqoffset: Optional[non_neg_int] = 0
    wait_for_first_scanbound: Optional[StrictBool] = False

    # These fields have default values. Some have specification requirements in conjunction with each other
    # e.g. one of intt or intn must be specified.
    tx_antennas: Optional[
        Annotated[
            conlist(
                Annotated[int, Field(ge=0, lt=options.main_antenna_count, strict=True)],
                max_length=options.main_antenna_count,
            ),
            AfterValidator(_validate_unique_list),
        ]
    ] = None
    rx_main_antennas: Optional[
        Annotated[
            conlist(
                Annotated[int, Field(ge=0, lt=options.main_antenna_count, strict=True)],
                max_length=options.main_antenna_count,
            ),
            AfterValidator(_validate_unique_list),
        ]
    ] = None
    rx_intf_antennas: Optional[
        Annotated[
            conlist(
                Annotated[int, Field(ge=0, lt=options.main_antenna_count, strict=True)],
                max_length=options.intf_antenna_count,
            ),
            AfterValidator(_validate_unique_list),
        ]
    ] = None
    tx_antenna_pattern: Optional[Callable] = None
    rx_antenna_pattern: Optional[Callable] = None
    tx_beam_order: Optional[List[Union[List[non_neg_int], non_neg_int]]] = None
    intt: Optional[non_neg_float] = None
    scanbound: Optional[
        Annotated[List[non_neg_float], AfterValidator(check_list_increasing)]
    ] = None
    pulse_phase_offset: Optional[Callable] = None
    decimation_scheme: DecimationScheme = Field(default_factory=create_default_scheme)

    cfs_range: Optional[
        Annotated[
            conlist(freq_int_khz, min_length=2, max_length=2),
            AfterValidator(check_list_increasing),
        ]
    ] = None
    cfs_flag: StrictBool = False
    cfs_duration: Optional[non_neg_int] = 90  # ms
    cfs_scheme: DecimationScheme = Field(default_factory=create_default_cfs_scheme)
    cfs_stable_time: Optional[non_neg_int] = 0  # seconds
    cfs_pwr_threshold: Optional[NonNegativeFloat] = None  # dB
    cfs_fft_n: Optional[non_neg_int] = 512
    cfs_freq_res: Optional[NonNegativeFloat] = None  # Hz
    cfs_always_run: Optional[StrictBool] = False

    acfint: Optional[StrictBool] = False
    xcf: Optional[StrictBool] = False
    lag_table: Optional[
        List[Annotated[List[non_neg_int], AfterValidator(check_list_sorted)]]
    ] = Field(default_factory=list)
    range_sep: Optional[PositiveFloat] = None
    slice_interfacing: Optional[dict[non_neg_int, str]] = Field(default_factory=dict)

    # Validators which check that all mutually exclusive sets of fields have one option set
    @model_validator(mode="before")
    @classmethod
    def check_tx_specifier(cls, values):
        if isinstance(values, dict):
            data = values
        else:
            data = values.kwargs
        if "tx_antenna_pattern" in data and "tx_beam_order" not in data:
            raise ValueError(
                f"tx_beam_order must be specified if tx_antenna_pattern specified. Slice: "
                f"{data['slice_id']}"
            )
        elif data.get("tx_beam_order", None) is not None and data.get("rxonly", False):
            raise ValueError(
                f"rxonly specified as True but tx_beam_order specified. Slice: {data['slice_id']}"
            )
        elif (
            data.get("tx_beam_order", None) is None
            and data.get("rxonly", False) is False
        ):
            raise ValueError(
                f"rxonly specified as False but tx_beam_order not given. Slice: {data['slice_id']}"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def check_intt_intn(cls, values):
        if isinstance(values, dict):
            data = values
        else:
            data = values.kwargs
        if data.get("intt", None) is None and data.get("intn", None) is None:
            raise ValueError(
                f"Slice must specify either an intn (unitless) or intt in ms. Slice: {data['slice_id']}"
            )
        elif data.get("intt", None) is not None and data.get("intn", None) is not None:
            raise ValueError(
                f"intn is set in experiment slice but will not be used due to intt. Slice: "
                f"{data['slice_id']}"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def check_freq_cfs_range(cls, values):
        if isinstance(values, dict):
            data = values
        else:
            data = values.kwargs
        if data.get("cfs_range", False):
            if ("cfs_flag" not in data or not data.get("cfs_flag", False)) and data.get(
                "freq", False
            ):
                log.info(
                    f"Slice parameter 'freq' ignored as 'cfs_range' takes precedence. If this is not desired, "
                    f"remove 'cfs_range' parameter from experiment. Slice: {data['slice_id']}"
                )
            data["cfs_flag"] = True
        elif data.get("freq", False):
            data["cfs_flag"] = False
        else:
            raise ValueError(
                f"A freq or cfs_range must be specified in a slice. Slice: {data['slice_id']}"
            )
        return values

    # Validators that depend on other previously-validated fields

    @model_validator(mode="after")
    def check_intt(self):
        if self.intt is None:  # Not provided
            return self

        # check intn and intt make sense given tau_spacing, and pulse_sequence.
        # Sequence length is length of pulse sequence plus the scope sync delay time.
        # TODO: this is an old check and seqtime now set in sequences class, update.
        seq_len = (self.tau_spacing * self.pulse_sequence[-1]) + (
            self.num_ranges + 19 + 10
        ) * self.pulse_len  # us
        if seq_len > (self.intt * 1000):  # seq_len in us, intt in ms
            raise ValueError(
                f"Slice {self.slice_id}: pulse sequence is too long for integration time given"
            )
        return self

    @model_validator(mode="after")
    def check_cfs_duration(self):
        if self.cfs_flag:
            if self.cfs_duration < 10:
                raise ValueError(
                    f"Clear frequency search duration of {self.cfs_duration} ms is too short. "
                    f"Must be at least 10 ms long."
                )

        return self

    @model_validator(mode="after")
    def check_cfs_scheme(self):
        if self.cfs_flag:
            if len(self.cfs_scheme.stages) > options.max_filtering_stages:
                errmsg = (
                    f"Number of cfs decimation stages ({len(self.cfs_scheme.stages)}) is greater than max"
                    f" available {options.max_filtering_stages}"
                )
                raise ValueError(errmsg)

            # Check that the rx_bandwidth matches input rate of the DecimationScheme
            input_rate = self.cfs_scheme.input_rates[0]
            if input_rate != self.rx_bandwidth:
                raise ValueError(
                    f"decimation_scheme input data rate {input_rate} does not match rx_bandwidth "
                    f"{self.rx_bandwidth}"
                )

            # Make sure default cfs scheme is only used with expected 300kHz range
            if self.cfs_range is not None:
                cfs_width = int(self.cfs_range[1] - self.cfs_range[0])
                if cfs_width > 300:
                    test_scheme = create_default_cfs_scheme()
                    if self.cfs_scheme == test_scheme:
                        raise ValueError(
                            f"CFS slice {self.slice_id} range is greater than the default 300kHz width. "
                            f"You must define a custom decimation scheme to match the {cfs_width}kHz width or "
                            f"adjust the cfs_range values of the experiment."
                        )

        return self

    @model_validator(mode="after")
    def check_cfs_freq_res(self):
        # TODO: Implement a check to default to cfs_fft_n if the parameter was set by the user
        if self.cfs_freq_res is not None:
            dm = 1
            for stage in self.cfs_scheme.stages:
                dm = dm * stage.dm_rate
            new_n = int((self.rx_bandwidth / dm) / self.cfs_freq_res)
            log.info(
                f"CFS frequency resolution of {self.cfs_freq_res} Hz was requested",
                resolution_set=(self.rx_bandwidth / dm) / new_n,
            )
            self.cfs_fft_n = new_n
        return self

    @field_validator("tx_antennas", mode="after")
    @classmethod
    def check_tx_antennas(cls, tx_antennas):
        if tx_antennas is None:
            tx_antennas = [i for i in options.tx_main_antennas]
        for ant in tx_antennas:
            if ant not in options.tx_main_antennas:
                raise ValueError(f"TX antenna {ant} not specified in config file")
        tx_antennas.sort()
        return tx_antennas

    @field_validator("rx_main_antennas", mode="after")
    @classmethod
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

    @field_validator("rx_intf_antennas", mode="after")
    @classmethod
    def check_rx_intf_antennas(cls, rx_intf_antennas):
        if rx_intf_antennas is None:
            return [i for i in options.rx_intf_antennas]
        for ant in rx_intf_antennas:
            if ant not in options.rx_intf_antennas:
                raise ValueError(f"RX intf antenna {ant} not specified in config file")
        rx_intf_antennas.sort()
        return rx_intf_antennas

    @field_validator("tx_antenna_pattern", mode="after")
    @classmethod
    def check_tx_antenna_pattern(cls, pattern, info):
        if pattern is None:  # No value given
            return pattern

        antenna_pattern = pattern(
            info.data["freq"], info.data["tx_antennas"], options.main_antenna_locations
        )
        if not isinstance(antenna_pattern, np.ndarray):
            raise ValueError(
                f"Slice {info.data['slice_id']} tx antenna pattern return is not a numpy array"
            )
        else:
            if len(antenna_pattern.shape) != 2:
                raise ValueError(
                    f"Slice {info.data['slice_id']} tx antenna pattern return shape "
                    f"{antenna_pattern.shape} must be 2-dimensional"
                )
            elif antenna_pattern.shape[1] != options.main_antenna_count:
                raise ValueError(
                    f"Slice {info.data['slice_id']} tx antenna pattern return 2nd dimension "
                    f"({antenna_pattern.shape[1]}) must be equal to number of main antennas "
                    f"({options.main_antenna_count})"
                )
            antenna_pattern_mag = np.abs(antenna_pattern)
            if np.argwhere(antenna_pattern_mag > 1.0).size > 0:
                raise ValueError(
                    f"Slice {info.data['slice_id']} tx antenna pattern return must not have any "
                    f"values with a magnitude greater than 1"
                )
        return pattern

    @field_validator("rx_antenna_pattern", mode="after")
    @classmethod
    def check_rx_antenna_pattern(cls, pattern, info):
        if pattern is None:  # No value given
            return pattern

        # Main and interferometer patterns
        antenna_pattern = [
            pattern(
                info.data["beam_angle"],
                info.data["freq"],
                options.main_antenna_locations,
            ),
            pattern(
                info.data["beam_angle"],
                info.data["freq"],
                options.intf_antenna_locations,
            ),
        ]
        for index in range(0, len(antenna_pattern)):
            if index == 0:
                array = "main"
                antenna_num = len(options.rx_main_antennas)
            else:
                array = "interferometer"
                antenna_num = len(options.rx_intf_antennas)
            if not isinstance(antenna_pattern[index], np.ndarray):
                raise ValueError(
                    f"Slice {info.data['slice_id']} {array} array rx antenna pattern return is "
                    f"not a numpy array"
                )
            else:
                if antenna_pattern[index].shape != (
                    len(info.data["beam_angle"]),
                    antenna_num,
                ):
                    raise ValueError(
                        f"Slice {info.data['slice_id']} {array} array must be the same shape as"
                        f" ([beam angle], [antenna_count])"
                    )
            antenna_pattern_mag = np.abs(antenna_pattern[index])
            if np.argwhere(antenna_pattern_mag > 1.0).size > 0:
                raise ValueError(
                    f"Slice {info.data['slice_id']} {array} array rx antenna pattern return must not have "
                    f"any values with a magnitude greater than 1"
                )
        return pattern

    @field_validator("rx_beam_order", mode="after")
    @classmethod
    def check_rx_beam_order(cls, rx_beam_order, info):
        if "beam_angle" not in info.data:
            return rx_beam_order

        for rx_beam in rx_beam_order:
            if isinstance(rx_beam, list):
                for beamnum in rx_beam:
                    if beamnum >= len(info.data["beam_angle"]):
                        raise ValueError(
                            f"Beam number {beamnum} could not index in beam_angle list of length "
                            f"{len(info.data['beam_angle'])}. Slice: {info.data['slice_id']}"
                        )
            else:
                if rx_beam >= len(info.data["beam_angle"]):
                    raise ValueError(
                        f"Beam number {rx_beam} could not index in beam_angle list of length "
                        f"{len(info.data['beam_angle'])}. Slice: {info.data['slice_id']}"
                    )
        return rx_beam_order

    @field_validator("tx_beam_order", mode="after")
    @classmethod
    def check_tx_beam_order(cls, tx_beam_order, info):
        if tx_beam_order is None:  # Empty list, was not specified
            return tx_beam_order

        if info.data.get("rx_beam_order", None) is not None and len(
            tx_beam_order
        ) != len(info.data["rx_beam_order"]):
            raise ValueError(
                f"tx_beam_order does not have same length as rx_beam_order. Slice: {info.data['slice_id']}"
            )

        num_beams = None
        if info.data.get("tx_antenna_pattern", None) is not None:
            antenna_pattern = info.data["tx_antenna_pattern"](
                info.data["freq"],
                info.data["tx_antennas"],
                options.main_antenna_locations,
            )
            if isinstance(antenna_pattern, np.ndarray):
                num_beams = antenna_pattern.shape[0]
        elif info.data.get("beam_angle", None) is not None:
            num_beams = len(info.data["beam_angle"])
        else:
            return tx_beam_order

        for bmnum in tx_beam_order:
            if bmnum >= num_beams:
                raise ValueError(
                    f"Slice {info.data['slice_id']} scan tx beam number {bmnum} DNE"
                )
        if (
            info.data.get("tx_antennas", None) is None
            or len(info.data["tx_antennas"]) == 0
        ):
            raise ValueError(
                "Must have TX antennas specified if tx_beam_order specified"
            )

        return tx_beam_order

    @field_validator("scanbound", mode="after")
    @classmethod
    def check_scanbound(cls, scanbound, info):
        if scanbound is None:  # No scanbound defined
            return scanbound

        if info.data.get("intt", None) is None:
            raise ValueError(
                f"Slice {info.data['slice_id']} must have intt enabled to use scanbound"
            )
        intt = info.data["intt"]
        slice_id = info.data["slice_id"]
        # Check if any scanbound times are shorter than the intt.
        tolerance = 1e-9
        if len(scanbound) == 1:
            if intt > (scanbound[0] * 1000 + tolerance):
                raise ValueError(
                    f"Slice {slice_id} intt {intt}ms longer than "
                    f"scanbound time {scanbound[0]}s"
                )
        else:
            for i in range(len(scanbound) - 1):
                beam_time = (scanbound[i + 1] - scanbound[i]) * 1000
                if intt > beam_time + tolerance:
                    raise ValueError(
                        f"Slice {slice_id} intt {intt}ms longer than "
                        f"one of the scanbound times"
                    )
        return scanbound

    @field_validator("pulse_phase_offset", mode="after")
    @classmethod
    def check_pulse_phase_offset(cls, ppo, info):
        if ppo is None:  # No value given
            return ppo

        # Test the encoding fn with beam iterator of 0 and sequence num of 0. test the user's
        # phase encoding function on first beam (beam_iterator = 0) and first sequence
        # (sequence_number = 0)
        phase_encoding = ppo(0, 0, len(info.data["pulse_sequence"]))
        slice_id = info.data["slice_id"]
        if not isinstance(phase_encoding, np.ndarray):
            raise ValueError(
                f"Slice {slice_id} Phase encoding return is not numpy array"
            )
        else:
            if len(phase_encoding.shape) > 1:
                raise ValueError(
                    f"Slice {slice_id} Phase encoding return must be 1 dimensional"
                )
            else:
                if phase_encoding.shape[0] != len(info.data["pulse_sequence"]):
                    raise ValueError(
                        f"Slice {slice_id} Phase encoding return dimension must be equal to "
                        f"number of pulses"
                    )
        return ppo

    @field_validator("txctrfreq", "rxctrfreq", mode="before")
    @classmethod
    def check_ctrfreq(cls, ctrfreq):
        if isinstance(ctrfreq, (float, int)):
            # convert from kHz to Hz to get correct clock divider. Return the result back in kHz.
            clock_multiples = options.usrp_master_clock_rate / 2**32
            clock_divider = math.ceil(ctrfreq * 1e3 / clock_multiples)
            ctrfreq = (clock_divider * clock_multiples) / 1e3

        return ctrfreq

    @field_validator("tx_freq_bounds", mode="after")
    @classmethod
    def check_tx_freq_bounds(cls, _, info):
        # max frequency is defined as [center freq] + [bandwidth / 2] - [bandwidth * 0.15]
        # min frequency is defined as [center freq] - [bandwidth / 2] + [bandwidth * 0.15]
        # [bandwidth * 0.15] is the transition bandwidth. This was set a 750 kHz originally
        # but for smaller bandwidth this value is too large. For the typical operating
        # bandwidth of 5 MHz, the calculated transition bandwidth here will be 750 kHz
        if info.data.get("txctrfreq", None) is not None:
            tx_center = info.data["txctrfreq"]

            tx_maxfreq = (
                tx_center * 1000
                + (info.data["tx_bandwidth"] / 2.0)
                - (info.data["tx_bandwidth"] * 0.15)
            )
            tx_minfreq = (
                tx_center * 1000
                - (info.data["tx_bandwidth"] / 2.0)
                + (info.data["tx_bandwidth"] * 0.15)
            )

            tx_freq_bounds = (tx_minfreq / 1000, tx_maxfreq / 1000)
        else:
            tx_freq_bounds = (options.min_freq / 1000, options.max_freq / 1000)

        return tx_freq_bounds

    @field_validator("rx_freq_bounds", mode="after")
    @classmethod
    def check_rx_freq_bounds(cls, _, info):
        # max frequency is defined as [center freq] + [bandwidth / 2] - [bandwidth * 0.15]
        # min frequency is defined as [center freq] - [bandwidth / 2] + [bandwidth * 0.15]
        # [bandwidth * 0.15] is the transition bandwidth. This was set a 750 kHz originally
        # but for smaller bandwidth this value is too large. For the typical operating
        # bandwidth of 5 MHz, the calculated transition bandwidth here will be 750 kHz
        if info.data.get("rxctrfreq", None) is not None:
            rx_center = info.data["rxctrfreq"]
            rx_maxfreq = (
                rx_center * 1000
                + (info.data["rx_bandwidth"] / 2.0)
                - (info.data["rx_bandwidth"] * 0.15)
            )
            rx_minfreq = (
                rx_center * 1000
                - (info.data["rx_bandwidth"] / 2.0)
                + (info.data["rx_bandwidth"] * 0.15)
            )

            rx_freq_bounds = (rx_minfreq / 1000, rx_maxfreq / 1000)
        else:
            rx_freq_bounds = (options.min_freq / 1000, options.max_freq / 1000)

        return rx_freq_bounds

    @field_validator("freq", mode="after")
    @classmethod
    def check_freq(cls, freq, info):
        if freq is None:
            return freq

        for freq_range in options.restricted_ranges:
            if freq_range[0] <= freq <= freq_range[1]:
                raise ValueError(
                    f"freq is within a restricted frequency range {freq_range}"
                )

        # Frequency must be within bandwidth of rx and tx center frequency
        if (freq > info.data["rx_freq_bounds"][1]) or (
            freq < info.data["rx_freq_bounds"][0]
        ):
            raise ValueError(
                f"Slice frequency is outside rx frequency bounds {info.data['rx_freq_bounds']}"
            )
        if info.data.get("rxctrfreq", None) is not None:
            rx_center = info.data["rxctrfreq"]
            # Frequency cannot be set to the rx or tx center frequency (100kHz bandwidth around center freqs)
            if abs(freq - rx_center) < 50:
                raise ValueError(
                    f"Slice frequency cannot be within 50kHz of rx center frequency {rx_center:.3f}"
                )

        # TODO review issue #195 - Characterize transmit waveforms near edge of tx bandwidth
        if info.data["rxonly"] is False:
            tx_center = info.data["txctrfreq"]
            if (freq > info.data["tx_freq_bounds"][1]) or (
                freq < info.data["tx_freq_bounds"][0]
            ):
                raise ValueError(
                    f"Slice frequency is outside tx frequency bounds {info.data['tx_freq_bounds']}"
                )
            if tx_center is not None and abs(freq - tx_center) < 50:
                raise ValueError(
                    f"Slice frequency cannot be within 50kHz of tx center frequency {tx_center:.3f}"
                )

        return freq

    @field_validator("cfs_range", mode="after")
    @classmethod
    def check_cfs_range(cls, cfs_range, info):
        if cfs_range is None:
            return cfs_range

        # Need to prevent the cfs_range from being outside the tx and rx operating ranges.
        if (
            cfs_range[0] < info.data["tx_freq_bounds"][0]
            or cfs_range[0] < info.data["rx_freq_bounds"][0]
        ):
            raise ValueError(
                f"Slice {info.data['slice_id']} cfs_range minimum value needs to be equal to "
                f"or greater than the tx and rx minimum operating frequencies: "
                f"{info.data['tx_freq_bounds'][0]} and {info.data['rx_freq_bounds'][0]}"
            )

        if (
            cfs_range[1] > info.data["tx_freq_bounds"][1]
            or cfs_range[1] > info.data["rx_freq_bounds"][1]
        ):
            raise ValueError(
                f"Slice {info.data['slice_id']} cfs_range maximum value needs to be equal to "
                f"or less than the tx and rx maximum operating frequencies: "
                f"{info.data['tx_freq_bounds'][1]} and {info.data['rx_freq_bounds'][1]}"
            )

        for freq_range in options.restricted_ranges:
            if freq_range[0] <= cfs_range[0] <= freq_range[1]:
                if freq_range[0] <= cfs_range[1] <= freq_range[1]:
                    # the range is entirely within the restricted range.
                    raise ValueError(
                        f"cfs_range is entirely within restricted range {freq_range}. Slice: "
                        f"{info.data['slice_id']}"
                    )

        if info.data.get("txctrfreq", None) is not None:
            tx_band = (info.data["txctrfreq"] - 50, info.data["txctrfreq"] + 50)
            if cfs_range[0] <= tx_band[1] and cfs_range[1] >= tx_band[0]:
                log.warning(
                    f"Slice {info.data['slice_id']} cfs range {cfs_range} is close to the "
                    f"tx center frequency {info.data['txctrfreq']}. The cfs frequency "
                    f"selection cannot chose a frequency within 50kHz of the center freq. "
                    f"Frequencies within {tx_band} will not be used for transmission"
                )

        if info.data.get("rxctrfreq", None) is not None:
            rx_band = (info.data["rxctrfreq"] - 50, info.data["rxctrfreq"] + 50)
            if cfs_range[0] <= rx_band[1] and cfs_range[1] >= rx_band[0]:
                log.warning(
                    f"Slice {info.data['slice_id']} cfs range {cfs_range} is close to the "
                    f"rx center frequency {info.data['rxctrfreq']}. The cfs frequency "
                    f"selection cannot chose a frequency within 50kHz of the center freq. "
                    f"Frequencies within {rx_band} will not be used for transmission"
                )

        return cfs_range

    @field_validator("decimation_scheme", mode="after")
    @classmethod
    def check_decimation_rates(cls, decimation_scheme, info):
        # check that number of stages is not too large
        if len(decimation_scheme.stages) > options.max_filtering_stages:
            errmsg = (
                f"Number of decimation stages ({len(decimation_scheme.stages)}) is greater than max"
                f" available {options.max_filtering_stages}"
            )
            raise ValueError(errmsg)

        # Check that the rx_bandwidth matches input rate of the DecimationScheme
        input_rate = decimation_scheme.input_rates[0]
        if input_rate != info.data["rx_bandwidth"]:
            raise ValueError(
                f"decimation_scheme input data rate {input_rate} does not match rx_bandwidth "
                f"{info.data['rx_bandwidth']}"
            )

        dm_rate = 1
        for stage in decimation_scheme.stages:
            dm_rate *= stage.dm_rate
        output_rx_rate = info.data["rx_bandwidth"] / dm_rate
        if info.data.get("tau_spacing", None) is not None:
            if not math.isclose(
                (info.data["tau_spacing"] * output_rx_rate % 1.0), 0.0, abs_tol=0.0001
            ):
                raise ValueError(
                    f"Slice {info.data['slice_id']} correlation lags will be off because tau_spacing "
                    f"{info.data['tau_spacing']} us is not a multiple of the output rx sampling period "
                    f"(1/output_rx_rate {output_rx_rate:.3f} Hz)."
                )

        return decimation_scheme

    # Validators that set dynamic default values (depends on user-specified fields)

    @field_validator("xcf", mode="after")
    @classmethod
    def check_xcf(cls, xcf, info):
        if not info.data["acf"]:
            xcf = False
            log.verbose(
                f"XCF defaulted to False as ACF not set. Slice: {info.data['slice_id']}"
            )
        if xcf and len(info.data.get("rx_intf_antennas", [])) == 0:
            raise ValueError("XCF set to True but no interferometer antennas present")
        return xcf

    @field_validator("acfint", mode="after")
    @classmethod
    def check_acfint(cls, acfint, info):
        if not info.data["acf"]:
            acfint = False
            log.verbose(
                f"ACFINT defaulted to False as ACF not set. Slice: {info.data['slice_id']}"
            )
        if acfint and len(info.data.get("rx_intf_antennas", [])) == 0:
            raise ValueError(
                "ACFINT set to True but no interferometer antennas present"
            )
        return acfint

    @field_validator("range_sep", mode="after")
    @classmethod
    def check_range_sep(cls, range_sep, info):
        # This is the distance travelled by the wave in the length of the pulse, divided by
        # two because it's an echo (travels there and back). In km.
        if info.data.get("pulse_len", None) is None:
            return range_sep

        correct_range_sep = info.data["pulse_len"] * 1.0e-9 * speed_of_light / 2.0  # km
        if info.data.get("acf", False) and range_sep is not None:
            if not math.isclose(range_sep, correct_range_sep, abs_tol=0.01):
                errmsg = (
                    f"range_sep = {range_sep} was set incorrectly. range_sep will be overwritten "
                    f"based on pulse_len, which must be equal to 1/rx_rate. The new range_sep = "
                    f"{correct_range_sep}"
                )
                log.warning(errmsg)
        range_sep = correct_range_sep
        return range_sep

    @field_validator("averaging_method", mode="after")
    @classmethod
    def check_averaging_method(cls, averaging_method, info):
        if info.data["acf"]:
            return averaging_method or "mean"
        else:
            log.verbose(
                f"Averaging method unset as ACF not set. Slice: {info.data['slice_id']}"
            )
            return averaging_method

    @field_validator("lag_table", mode="after")
    @classmethod
    def check_lag_table(cls, lag_table, info):
        if info.data["acf"]:
            if lag_table is not None and len(lag_table) > 0:
                # Check that lags are valid
                for lag in lag_table:
                    if not set(np.array(lag).flatten()).issubset(
                        set(info.data["pulse_sequence"])
                    ):
                        raise ValueError(
                            f"Lag {lag} not valid; One of the pulses does not exist in the sequence. "
                            f"Slice: {info.data['slice_id']}"
                        )
            else:
                # build lag table from pulse_sequence
                lag_table = list(itertools.combinations(info.data["pulse_sequence"], 2))
                lag_table.append(
                    [info.data["pulse_sequence"][0], info.data["pulse_sequence"][0]]
                )  # lag 0
                # sort by lag number
                lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
                lag_table.append(
                    [info.data["pulse_sequence"][-1], info.data["pulse_sequence"][-1]]
                )  # alternate lag 0
                lag_table = lag_table
        else:
            log.verbose(
                f"Lag table unused as ACF not set. Slice: {info.data['slice_id']}"
            )
            lag_table = []
        return lag_table

    # Validators for when a check requires that an early-validated field and later-validated field have both been
    # validated. E.g. could not validate pulse_len fully off the bat because it depends on acf, which gets validated
    # later.

    @model_validator(mode="after")
    def check_pulse_len(self):
        if self.pulse_len > self.tau_spacing:
            raise ValueError(
                f"Slice {self.slice_id} pulse length greater than tau_spacing"
            )
        if self.pulse_len <= 2 * options.pulse_ramp_time * 1.0e6:
            raise ValueError(f"Slice {self.slice_id} pulse length too small")

        if self.acf:
            dm_rate = 1
            for stage in self.decimation_scheme.stages:
                dm_rate *= stage.dm_rate
            output_rx_rate = self.rx_bandwidth / dm_rate
            # The below check is an assumption that is made during acf calculation
            # (1 output received sample = 1 range separation)
            if not math.isclose(
                self.pulse_len,
                (1 / output_rx_rate * 1e6),
                abs_tol=0.0000001,
            ):
                raise ValueError(
                    f"For an experiment slice with real-time acfs, pulse length must be equal (within 1 "
                    f"us) to 1/output_rx_rate to make acfs valid. Current pulse length is "
                    f"{self.pulse_len} us, output rate is {output_rx_rate:.3f} Hz. "
                    f"Slice: {self.slice_id}"
                )
        return self
