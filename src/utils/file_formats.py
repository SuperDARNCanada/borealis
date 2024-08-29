"""
file_formats
~~~~~~~~~~~~

Contains the dataclass `SliceData` defining the data and metadata that is stored in files produced by borealis.

`SliceData` fields contain associated metadata that determines which file types (`rawrf`, `antennas_iq`, `bfiq`,
`rawacf`, and `txdata`) should contain the aforementioned field, and at which level (`file` or `record`) the field
should be written. Fields at the `file` level are written only once, with the associated data immutable throughout
the experiment for the given slice. Fields at the `record` level are written to each averaging period, within the record
for that averaging period.
"""

from dataclasses import dataclass, field, fields
import numpy as np


@dataclass(init=False)
class SliceData:
    """
    This class defines all fields that need to be written by any type of data file. The 'groups' metadata lists
    the applicable file types for each field.

    Each field contains metadata that determines how the field is written to file:
    * `groups`: the types of data file that need this field to be written.
    * `level`: the level within the file that the data will be stored at. Either `file` or `record`, indicating
       that the field is either written once per file, or once per record.
    * `description`: A short description of the field.
    """

    agc_status_word: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "32 bits, a 1 in bit position corresponds to an AGC fault on that transmitter",
            "dim_labels": ["sequence"],
        }
    )
    antenna_arrays_order: list[str] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq"],
            "level": "file",
            "description": "Descriptors for the data layout",
            "dim_labels": ["antenna"],
        }
    )
    averaging_method: str = field(
        metadata={
            "groups": ["rawacf"],
            "level": "file",
            "description": "Averaging method, e.g. mean, median",
        }
    )
    beam_azms: list[float] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Beams azimuths for each beam in degrees CW of boresight",
            "dim_labels": ["beam"],
        }
    )
    beam_nums: list[int] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Beam numbers used in this slice",
            "dim_labels": ["beam"],
        }
    )
    blanked_samples: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Samples blanked during transmission",
            "dim_labels": ["time"],
        }
    )
    borealis_git_hash: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Version and commit hash of Borealis at runtime",
        }
    )
    cfs_freqs: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Frequencies measured during clear frequency search",
            "dim_labels": ["freq"],
        }
    )
    cfs_masks: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Mask for cfs_freqs restricting freqs available for setting cfs slice freq",
            "dim_labels": ["freq"],
        }
    )
    cfs_noise: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Power measured during clear frequency search",
            "dim_labels": ["freq"],
        }
    )
    cfs_range: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Range of frequencies examined by clear frequency search",
            "dim_labels": ["freq"],
        }
    )
    data: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawrf"],
            "level": "record",
            "description": "Contiguous set of samples at the given sample rate",
            "dim_labels": [],
        }
    )
    data_descriptors: list[str] = field(  # todo: moot, with dim_labels?
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Denotes what each data dimension represents",
        }
    )
    data_normalization_factor: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Cumulative scale of all of the filters for a total scaling factor to normalize by",
        }
    )
    decimated_tx_samples: list = field(
        metadata={
            "groups": ["txdata"],
            "level": "record",
            "description": "Samples decimated by dm_rate",
        }
    )  # todo: Is this after each stage, or just the final samples?
    dm_rate: list[int] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Total decimation rate of the filtering scheme",
        }
    )  # todo: Is this supposed to be a list of ALL dm_rates, or just the total?
    experiment_comment: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Comment about the whole experiment",
        }
    )
    experiment_id: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Number used to identify experiment",
        }
    )
    experiment_name: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Name of the experiment class",
        }
    )
    first_range: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Distance to first range in km",
        }
    )
    first_range_rtt: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Round trip time of flight to first range in microseconds",
        }
    )
    freq: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Frequency used for this experiment slice in kHz",
        }
    )
    gps_locked: bool = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "True if the GPS was locked during the entire averaging period",
            "dim_labels": ["sequence"],
        }
    )
    gps_to_system_time_diff: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Max time diff in seconds between GPS and system/NTP time during the averaging "
            "period",
            "dim_labels": ["sequence"],
        }
    )
    int_time: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Integration time in seconds",
        }
    )
    intf_acfs: np.ndarray = field(
        metadata={
            "groups": ["rawacf"],
            "level": "record",
            "description": "Interferometer array autocorrelations",
            "dim_labels": ["beam", "range", "lag"],
        }
    )
    intf_antenna_count: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Number of interferometer array antennas",
        }
    )
    lags: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Unique pairs of pulses in pulse array, in units of tau_spacing",
            "dim_labels": ["lag"],
        }
    )
    lp_status_word: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "32 bits, a 1 in bit position corresponds to a low power condition on that "
            "transmitter",
            "dim_labels": ["sequence", "antenna"],
        }
    )
    main_acfs: np.ndarray = field(
        metadata={
            "groups": ["rawacf"],
            "level": "record",
            "description": "Main array autocorrelations",
            "dim_labels": ["beam", "range", "lag"],
        }
    )
    main_antenna_count: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Number of main array antennas",
        }
    )
    noise_at_freq: np.ndarray = field(  # TODO: Implement and give units
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Noise at the receive frequency",
            "dim_labels": ["sequence"],
        }
    )
    num_ranges: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq"],
            "level": "file",
            "description": "Number of ranges to calculate correlations for",
        }
    )
    num_samps: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawrf"],
            "level": "file",
            "description": "Number of samples in the sampling period",
        }
    )
    num_sequences: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Number of sampling periods in the averaging period",
        }
    )
    num_slices: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Number of slices in the experiment at this averaging period",
        }
    )
    pulse_phase_offset: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq"],
            "level": "record",
            "description": "Phase offset in degrees for each pulse in pulses",
            "dim_labels": ["sequence", "pulse"],
        }
    )
    pulse_sample_start: list[float] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Beginning of pulses in sequence measured in samples",
            "dim_labels": ["pulse"],
        }
    )
    pulse_timing_us: list[float] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Beginning of pulses in sequence measured in microseconds",
            "dim_labels": ["pulse"],
        }
    )
    pulses: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Pulse sequence in units of tau_spacing",
            "dim_labels": ["pulse"],
        }
    )
    range_sep: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Range gate separation (equivalent distance between samples) in km",
        }
    )
    rx_center_freq: float = field(
        metadata={
            "groups": ["rawrf"],
            "level": "file",
            "description": "Center frequency of the data in kHz",
        }
    )
    rx_sample_rate: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Sampling rate of the samples being written to file in Hz",
        }
    )
    rx_main_phases: list[complex] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "record",
            "description": "Phases of main array receive antennas for each antenna. Magnitude between 0 (off) "
            "and 1 (full power)",
            "dim_labels": ["beam", "antenna"],
        }
    )
    rx_intf_phases: list[complex] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "record",
            "description": "Phases of intf array receive antennas for each antenna. Magnitude between 0 (off) "
            "and 1 (full power)",
            "dim_labels": ["beam", "antenna"],
        }
    )
    samples_data_type: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "C data type of the samples",
        }
    )
    scan_start_marker: bool = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Designates if the record is the first in a scan",
        }
    )
    scheduling_mode: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Type of scheduling time at the time of this dataset",
        }
    )
    slice_comment: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Comment that describes the slice",
        }
    )
    slice_id: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Slice ID of the file and dataset",
        }
    )
    slice_interfacing: dict = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Interfacing of this slice to other slices",
        }
    )
    sqn_timestamps: list[float] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "GPS timestamps of start of first pulse for each sampling period in the averaging "
            "period in seconds since epoch",
            "dim_labels": ["sequence"],
        }
    )
    station: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Three letter radar identifier",
        }
    )
    tau_spacing: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Unit of spacing between pulses in microseconds",
        }
    )
    tx_antenna_phases: list[complex] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "record",
            "description": "Phases of transmit signal for each antenna. Magnitude between 0 (off) and 1 "
            "(full power)",
            "dim_labels": ["antenna"],
        }
    )
    tx_center_freq: list[float] = field(  # todo: Why is this a list?
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Center frequency of the transmitted data in kHz",
        }
    )
    tx_pulse_len: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Length of the pulse in microseconds",
        }
    )
    tx_rate: list[float] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Sampling rate of the samples being sent to the USRPs",
        }
    )  # todo: Why is this a list? Shouldn't it be a single number?
    tx_samples: list = field(
        metadata={
            "groups": ["txdata"],
            "level": "record",
            "description": "Samples sent to USRPs for transmission",
            "dim_labels": ["antenna", "time"],  # todo: verify
        }
    )
    xcfs: np.ndarray = field(
        metadata={
            "groups": ["rawacf"],
            "level": "record",
            "description": "Cross-correlations between main and interferometer arrays",
            "dim_labels": ["beam", "range", "lag"],
        }
    )

    @classmethod
    def type_fields(cls, file_type: str):
        """
        Returns a list of names for all fields which belong in 'file_type' files.
        """
        return [f.name for f in fields(cls) if file_type in f.metadata.get("groups")]

    @classmethod
    def file_level_fields(cls):
        """
        Returns a list of the names for all fields which are at the `file` level, and thus are constant
        for a given file.
        """
        return [f.name for f in fields(cls) if f.metadata.get("level") == "file"]

    @classmethod
    def record_level_fields(cls):
        """
        Returns a list of the names for all fields which are at the `record` level, and thus vary
        from averaging period to averaging period.
        """
        return [f.name for f in fields(cls) if f.metadata.get("level") == "record"]

    @classmethod
    def generate_data_dim_labels(cls, file_type: str):
        """
        Returns the dimension labels for the ``data`` field for ``file_type``.
        """
        dim_labels = {
            "rawrf": ["sequence", "antenna", "time"],  # todo: verify these dimensions
            "antennas_iq": ["antenna", "sequence", "time"],
            "bfiq": ["array", "sequence", "beam", "time"],
            "txdata": [""],  # todo: Get the dimensionality for this type
        }
        if file_type in dim_labels.keys():
            return dim_labels[file_type]

    def __repr__(self):
        """Print all available fields"""
        return f"{vars(self)}"
