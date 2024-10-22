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

import h5py
import numpy as np
import pydarnio


@dataclass(init=False)
class SliceData:
    """
    This class defines all fields that need to be written by any type of data file. The 'groups' metadata lists
    the applicable file types for each field.

    Each field contains metadata that determines how the field is written to file.

    ``description``
       A description of the field.

    ``dim_labels``
       If applicable, a brief descriptor for each dimension of the dataset. This could
       be different for different ``group`` values. If so, this metadata will be a dict, with the keys
       being the ``group`` name and the values the associated list of dimension labels.

    ``dim_scales``
       If applicable, dimension scales will be associated to the field. These are datasets that match
       one of the dimensions of the data, such as timestamps to go along with an array of collected data. Note that
       some dimensions may be associated with multiple fields. If a dimension has no associated dataset, the list
       will have a ``None`` entry.

    ``groups``
       The types of data file that need this field to be written.

    ``level``
       The level within the file that the data will be stored at. Either ``file`` or ``record``, indicating
       that the field is either written once per file, or once per record.

    ``nickname``
       A nickname for the field, used for making Dimension Scale names.

    ``units``
       Units for the data.
    """

    agc_status_word: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "32 bits, a 1 in bit position corresponds to an AGC fault on that transmitter",
            "required": True,
        }
    )
    antenna_arrays: list[str] = field(
        metadata={
            "groups": ["bfiq"],
            "level": "file",
            "nickname": "array",
            "description": "Descriptor of each antenna array contained in the data",
            "required": True,
        }
    )
    antenna_locations: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "file",
            "units": "m",
            "description": "Relative antenna locations",
            "dim_labels": ["antenna", "coordinate"],
            "required": True,
        }
    )
    antennas: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "file",
            "description": "Labels for each antenna of the radar",
            "dim_labels": ["antenna"],
            "required": True,
        }
    )
    antennas_iq_data: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq"],
            "level": "record",
            "units": "a.u. ~ V",
            "description": "Filtered and downsampled I&Q complex voltage samples for each antenna",
            "dim_labels": ["antenna", "sequence", "time"],
            "dim_scales": ["rx_antennas", "sqn_timestamps", "sample_time"],
            "required": True,
        }
    )
    averaging_method: str = field(
        metadata={
            "groups": ["rawacf"],
            "level": "file",
            "description": "Averaging method, e.g. mean, median",
            "required": True,
        }
    )
    beam_azms: list[float] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "nickname": "beam direction",
            "description": "Beams azimuths for each beam in degrees CW of boresight",
            "units": "degrees",
            "dim_labels": ["beam"],
            "required": True,
        }
    )
    beam_nums: list[int] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "nickname": "beam number",
            "description": "Beam numbers used in this slice",
            "dim_labels": ["beam"],
            "required": True,
        }
    )
    bfiq_data: np.ndarray = field(
        metadata={
            "groups": ["bfiq"],
            "level": "record",
            "units": "a.u. ~ V",
            "description": "Beamformed I&Q complex voltage samples for each antenna array",
            "dim_labels": ["array", "sequence", "beam", "time"],
            "dim_scales": [
                "antenna_arrays",
                "sqn_timestamps",
                ["beam_nums", "beam_azms"],
                "sample_time",
            ],
            "required": True,
        }
    )
    blanked_samples: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Samples blanked during transmission of a pulse",
            "dim_labels": ["time"],
            "required": True,
        }
    )
    borealis_git_hash: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Version and commit hash of Borealis at runtime",
            "required": True,
        }
    )
    cfs_freqs: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "nickname": "freq",
            "description": "Frequencies measured during clear frequency search",
            "units": "Hz",
            "dim_labels": ["freq"],
            "required": False,
        }
    )
    cfs_masks: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Mask for cfs_freqs restricting freqs available for setting cfs slice freq",
            "dim_labels": ["freq"],
            "dim_scales": ["cfs_freqs"],
            "required": False,
        }
    )
    cfs_noise: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "record",
            "description": "Power measured during clear frequency search",
            "units": "a.u. ~ dBW",
            "dim_labels": ["freq"],
            "dim_scales": ["cfs_freqs"],
            "required": False,
        }
    )
    cfs_range: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Lower and upper bound of frequencies examined by clear frequency search",
            "units": "Hz",
            "required": False,
        }
    )
    coordinates: list[str] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "file",
            "description": "Descriptors for location coordinate dimensions",
            "nickname": "coordinate",
            "required": True,
        }
    )
    data_normalization_factor: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Cumulative scale of all of the filters for a total scaling factor to normalize by",
            "required": True,
        }
    )
    decimated_tx_samples: list = field(
        metadata={
            "groups": ["txdata"],
            "level": "record",
            "units": "a.u. ~ V",
            "description": "Samples decimated by dm_rate",
            "dim_labels": ["antenna", "sequence", "time"],
            "dim_scales": ["tx_antennas", "sqn_timestamps", "sample_time"],
            "required": True,
        }
    )
    dm_rate: list[int] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Total decimation rate of the filtering scheme",
            "required": True,
        }
    )
    experiment_comment: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Comment about the whole experiment",
            "required": True,
        }
    )
    experiment_id: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Number used to identify experiment",
            "required": True,
        }
    )
    experiment_name: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Name of the experiment class",
            "required": True,
        }
    )
    first_range: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "units": "km",
            "description": "Distance to first range in km",
            "required": True,
        }
    )
    first_range_rtt: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "units": "μs",
            "description": "Round trip time of flight to first range in microseconds",
            "required": True,
        }
    )
    freq: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "units": "kHz",
            "description": "Frequency used for this experiment slice, in kHz",
            "required": True,
        }
    )
    gps_locked: bool = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "True if the GPS was locked during the entire averaging period",
            "required": True,
        }
    )
    gps_to_system_time_diff: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "units": "s",
            "description": "Max time diff in seconds between GPS and system/NTP time during the averaging "
            "period",
            "required": True,
        }
    )
    int_time: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "units": "s",
            "description": "Integration time in seconds",
            "required": True,
        }
    )
    intf_acfs: np.ndarray = field(
        metadata={
            "groups": ["rawacf"],
            "level": "record",
            "description": "Interferometer array autocorrelations",
            "units": "a.u. ~ W",
            "dim_labels": ["beam", "range", "lag"],
            "dim_scales": [["beam_azms", "beam_nums"], "range_gates", "lag_numbers"],
        }
    )
    lags: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "nickname": "lag",
            "description": "Lag indices",
            "required": True,
        }
    )
    lag_numbers: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "nickname": "lag",
            "units": "tau_spacing",
            "description": "Difference in units of tau_spacing of unique pairs of pulse in the pulse array",
            "required": True,
        }
    )
    lag_pulses: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "units": "tau_spacing",
            "description": "Unique pairs of pulses in pulse array, in units of tau_spacing",
            "dim_labels": ["lag", "pulse"],
            "dim_scales": ["lags", "lag_pulse_descriptors"],
            "required": True,
        }
    )
    lag_pulse_descriptors: list[str] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Descriptor of the pulse pairs used in a lag",
            "required": True,
        }
    )
    lp_status_word: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "32 bits, a 1 in bit position corresponds to a low power condition on that transmitter",
            "required": True,
        }
    )
    main_acfs: np.ndarray = field(
        metadata={
            "groups": ["rawacf"],
            "level": "record",
            "units": "a.u. ~ W",
            "description": "Main array autocorrelations",
            "dim_labels": ["beam", "range", "lag"],
            "dim_scales": [["beam_azms", "beam_nums"], "range_gates", "lag_numbers"],
            "required": True,
        }
    )
    num_sequences: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Number of sampling periods in the averaging period",
            "required": True,
        }
    )
    num_slices: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Number of slices in the experiment for this averaging period",
            "required": True,
        }
    )
    pulse_phase_offset: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq"],
            "level": "record",
            "units": "degrees",
            "description": "Phase offset in degrees for each pulse in pulses",
            "dim_labels": ["sequence", "pulse"],
            "dim_scales": ["sqn_timestamps", "pulses"],
            "required": False,
        }
    )
    pulse_sample_start: list[np.ndarray] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "description": "Beginning of pulses in sequence measured in samples",
            "dim_labels": ["sequence", "pulse"],
            "required": True,
        }
    )
    pulse_timing: list[np.ndarray] = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "units": "μs",
            "description": "Relative timing of pulse start for all pulses in the sequence",
            "dim_labels": ["sequence", "pulse"],
            "required": True,
        }
    )
    pulses: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "nickname": "pulse",
            "description": "Pulse sequence in units of tau_spacing",
            "dim_labels": ["pulse"],
            "required": True,
        }
    )
    range_gates: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Range gates of interest for the experiment, beginning at ``first_range`` and spaced by "
            "``range_sep``",
            "nickname": "range gate",
            "required": True,
        }
    )
    range_sep: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "units": "km",
            "description": "Range gate separation (equivalent distance between samples) in km",
            "required": True,
        }
    )
    rawrf_data: np.ndarray = field(
        metadata={
            "groups": ["rawrf"],
            "level": "record",
            "description": "I&Q complex voltage samples for each antenna",
            "units": "a.u. ~ V",
            "dim_labels": [
                "antenna",
                "sequence",
                "time",
            ],  # todo: verify these dimensions
            "dim_scales": ["rx_antennas", "sqn_timestamps", "sample_time"],
            "required": True,
        }
    )
    rx_antennas: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "rawrf"],
            "level": "file",
            "nickname": "rx antenna",
            "description": "Indices into ``antenna_locations`` of the antennas with recorded data",
            "required": True,
        }
    )
    rx_intf_antennas: list[int] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Indices into ``antenna_locations`` of the interferometer array antennas used in this experiment",
            "required": False,
        }
    )
    rx_main_antennas: list[int] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Indices into ``antenna_locations`` of the main array antennas used in this experiment",
            "required": True,
        }
    )
    rx_center_freq: float = field(
        metadata={
            "groups": ["rawrf"],
            "level": "file",
            "units": "kHz",
            "description": "Center frequency of the data in kHz",
            "required": True,
        }
    )
    rx_sample_rate: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "units": "Hz",
            "description": "Sampling rate of the samples being written to file in Hz",
            "required": True,
        }
    )
    rx_main_phases: list[complex] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Phases of main array receive antennas for each antenna. Magnitude between 0 (off) and 1 (full power)",
            "dim_labels": ["beam", "antenna"],
            "dim_scales": [["beam_azms", "beam_nums"], "rx_main_antennas"],
            "required": True,
        }
    )
    rx_intf_phases: list[complex] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Phases of interferometer array receive antennas for each antenna. Magnitude between 0 (off) and 1 (full power)",
            "dim_labels": ["beam", "antenna"],
            "dim_scales": [["beam_azms", "beam_nums"], "rx_intf_antennas"],
            "required": False,
        }
    )
    samples_data_type: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "C data type of the samples",
            "required": True,
        }
    )
    sample_time: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawrf"],
            "level": "file",
            "description": "Time of measurement relative to the first pulse in the sequence",
            "dim_labels": ["time"],
            "required": True,
            "units": "μs",
        }
    )
    scan_start_marker: bool = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "description": "Designates if the record is the first in a scan",
            "required": True,
        }
    )
    scheduling_mode: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Type of scheduling time at the time of this dataset",
            "required": True,
        }
    )
    slice_comment: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Comment that describes the slice",
            "required": True,
        }
    )
    slice_id: int = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Slice ID of the file and dataset",
            "required": True,
        }
    )
    slice_interfacing: dict = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "description": "Interfacing of this slice to other slices",
            "required": True,
        }
    )
    sqn_timestamps: list[float] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "record",
            "nickname": "timestamp",
            "units": "seconds since 1970-01-01 00:00:00 UTC",
            "description": "GPS timestamps of start of first pulse for each sampling period in the averaging period",
            "dim_labels": ["sequence"],
            "required": True,
        }
    )
    station: str = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Three letter radar identifier",
            "required": True,
        }
    )
    station_location: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf"],
            "level": "file",
            "description": "Location of the radar",
            "dim_labels": ["coordinate"],
            "required": True,
        }
    )
    tau_spacing: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "units": "μs",
            "description": "Unit of spacing between pulses in microseconds",
            "required": True,
        }
    )
    tx_antennas: list[int] = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "file",
            "description": "Indices into ``antenna_locations`` of the antennas used for transmitting in this experiment",
            "nickname": "tx antenna",
            "required": True,
        }
    )
    tx_antenna_phases: np.ndarray = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf", "rawrf", "txdata"],
            "level": "record",
            "units": "a.u.",
            "description": "Phases of transmit signal for each antenna. Magnitude between 0 (off) and 1 (full power)",
            "dim_labels": ["antenna"],
            "dim_scales": ["tx_antennas"],
            "required": False,
        }
    )
    tx_center_freq: float = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "units": "kHz",
            "description": "Center frequency of the transmitted data in kHz",
            "required": True,
        }
    )
    tx_pulse_len: float = field(
        metadata={
            "groups": ["antennas_iq", "bfiq", "rawacf"],
            "level": "file",
            "units": "μs",
            "description": "Length of the pulse in microseconds",
            "required": True,
        }
    )
    tx_rate: float = field(
        metadata={
            "groups": ["txdata"],
            "level": "file",
            "units": "Hz",
            "description": "Sampling rate of the samples being sent to the USRPs",
            "required": True,
        }
    )
    tx_samples: list[np.ndarray] = field(
        metadata={
            "groups": ["txdata"],
            "level": "record",
            "units": "a.u. ~ V",
            "description": "Samples sent to USRPs for transmission",
            "dim_labels": ["sequence", "antenna", "time"],
            "dim_scales": ["sqn_timestamps", "tx_antennas", "sample_time"],
            "required": True,
        }
    )
    xcfs: np.ndarray = field(
        metadata={
            "groups": ["rawacf"],
            "level": "record",
            "units": "a.u. ~ W",
            "description": "Cross-correlations between main and interferometer arrays",
            "dim_labels": ["beam", "range", "lag"],
            "dim_scales": [["beam_azms", "beam_nums"], "range_gates", "lag_numbers"],
            "required": False,
        }
    )

    @classmethod
    def all_fields(cls, file_type: str):
        """
        Returns a list of names for all fields which belong in 'file_type' files.
        """
        return [f.name for f in fields(cls) if file_type in f.metadata.get("groups")]

    @classmethod
    def required_fields(cls, file_type: str):
        """
        Returns a list of names for all fields which are required for ``file_type`` files.
        """
        return [
            f.name
            for f in fields(cls)
            if (
                file_type in f.metadata.get("groups")
                and f.metadata.get("required") is True
            )
        ]

    @classmethod
    def optional_fields(cls, file_type: str):
        """
        Returns a list of names for all fields which are optional for ``file_type`` files.
        """
        return [
            f.name
            for f in fields(cls)
            if (
                file_type in f.metadata.get("groups")
                and f.metadata.get("required") is False
            )
        ]

    @classmethod
    def _file_level_fields(cls):
        """
        Returns a list of the names for all fields which are at the ``file`` level, and thus are constant
        for a given file.
        """
        return [f.name for f in fields(cls) if f.metadata.get("level") == "file"]

    @classmethod
    def _record_level_fields(cls):
        """
        Returns a list of the names for all fields which are at the ``record`` level, and thus vary
        from averaging period to averaging period.
        """
        return [f.name for f in fields(cls) if f.metadata.get("level") == "record"]

    def _dim_scale_fields(self, file_type: str):
        """
        Returns a list of all fields that are [Dimension Scales](https://docs.h5py.org/en/stable/high/dims.html)
        for another field.

        The ``dim_scales`` metadata contains a list of field names which should be associated as dimension scales
        for a field. If the list depends on the file type, then the ``dim_scales`` metadata will be a dictionary
        keyed by file type. Some dimensions may have multiple associated scales, represented by a list of names
        for that dimension. Some dimensions may not have associated scales, indicated by a ``None`` value for that
        dimension.

        E.g. ``main_acfs`` field:
        ``dim_labels = ["beam", "range", "lag"]``
        ``dim_scales = [["beam_azms", "beam_nums"], None, "lag_numbers"]``
        The first dimension is associated with two other fields: ``beam_azms`` and ``beam_nums``.
        The second dimensions is not associated with any field.
        The third dimension is associated with the field ``lag_numbers``.
        """
        dim_scale_fields = set()
        for f in fields(self):
            if file_type not in f.metadata.get("groups"):
                continue
            dim_scales = f.metadata.get("dim_scales", None)
            if dim_scales is None:
                continue
            for dim in dim_scales:
                if dim is None:  # no scale for a particular dimension
                    continue
                elif isinstance(dim, list):  # multiple scales
                    dim_scale_fields = dim_scale_fields.union(set(dim))
                else:
                    dim_scale_fields.add(dim)
        return list(dim_scale_fields)

    @classmethod
    def _associate_dim_scales(cls, name: str, group: h5py.Group, dim_scales: list):
        """
        Associates fields as a [Dimension Scale](https://docs.h5py.org/en/stable/high/dims.html)
        of another field's dimension.
        """
        if len(group[name].shape) != len(dim_scales):
            raise ValueError(
                f"{name} has incompatible dimensionality {group[name].shape} with scales {dim_scales}"
            )
        for i, dim in enumerate(dim_scales):
            if dim is None:
                continue
            elif isinstance(dim, list):
                for d in dim:
                    group[name].dims[i].attach_scale(group[d])
            else:
                group[name].dims[i].attach_scale(group[dim])

    @staticmethod
    def _format_for_hdf5(field_data):
        """
        Converts ``field_data`` to supported types for a Borealis HDF5 file.
        """
        if isinstance(field_data, dict):
            return np.bytes_(str(field_data))
        elif isinstance(field_data, str):
            return np.bytes_(field_data)
        elif isinstance(field_data, bool):
            return np.bool_(field_data)
        elif isinstance(field_data, list):
            if len(field_data) > 0 and isinstance(field_data[0], str):
                return np.bytes_(field_data)
            else:
                return np.array(field_data)
        else:
            return field_data

    def _dispatch_to_write_method(
        self,
        name: str,
        fields_map: dict,
        group: h5py.Group,
        metadata_group: h5py.Group,
        data_type: str,
    ):
        """
        Determines whether a field is file-level or record-level, and writes to file accordingly.
        File-level fields are written once, in the top-level ``metadata`` group, and linked in each
        record.
        """
        try:
            data = getattr(self, name)
        except AttributeError as e:
            if name in self.required_fields(data_type):
                raise e
            else:
                return False
        formatted_data = self._format_for_hdf5(data)
        metadata = fields_map[name].metadata

        if name in self._file_level_fields():
            # Field is file-level metadata, so should be written to the metadata group
            self._write_metadata_field(
                name,
                formatted_data,
                metadata,
                metadata_group,
                data_type,
            )
            group[name] = metadata_group[name]  # create a hard link
        else:
            # Field is record-level, so write it to the group
            self._write_hdf5_field(name, formatted_data, metadata, group, data_type)

        return True

    @staticmethod
    def _write_hdf5_field(
        name: str, data, metadata: dict, group: h5py.Group, ftype: str
    ):
        """
        Write ``data`` to ``group`` along with the associated ``metadata``
        """
        kw = dict()
        if not np.isscalar(data):
            kw = {"compression": "gzip", "compression_opts": 9}
        group.create_dataset(name, data=data, **kw)
        group[name].attrs["description"] = metadata.get("description")

        units = metadata.get("units", None)
        if units is not None:
            group[name].attrs["units"] = units

        dim_labels = metadata.get("dim_labels", None)
        if dim_labels is not None:
            if isinstance(dim_labels, dict):
                dim_labels = dim_labels[ftype]
            if len(dim_labels) != len(data.shape):
                raise ValueError(
                    f"{name} shape {data.shape} does not match dimension labels {dim_labels}"
                )
            for i, dim in enumerate(dim_labels):
                group[name].dims[i].label = dim

        dim_scales = metadata.get("dim_scales", None)
        if dim_scales is not None:
            if isinstance(dim_scales, dict):
                dim_scales = dim_scales[ftype]
            SliceData._associate_dim_scales(name, group, dim_scales)

    @staticmethod
    def _write_metadata_field(
        name: str, data, metadata: dict, group: h5py.Group, ftype: str
    ):
        """
        Checks if a field is already in the ``group``. If so, verifies the data is identical.
        If not, writes the field to ``group``.
        """
        if name in group.keys():
            file_data = group[name][()]
            # verify it hasn't changed
            if np.issubdtype(file_data.dtype, bytes):
                equal = np.all(file_data == data)
            else:
                equal = np.allclose(file_data, data)
            if not equal:
                raise ValueError(
                    f"{name} already exists in file with different value.\n"
                    f"\tExisting: {file_data}\n"
                    f"\tNew: {data}"
                )
        else:
            # First time, write the metadata
            SliceData._write_hdf5_field(name, data, metadata, group, ftype)

    def to_hdf5(self, group: h5py.Group, metadata_group: h5py.Group, data_type: str):
        """
        Converts data from ``self`` of relevance for the data type into a ``group`` and ``metadata_group``

        :param group:          HDF5 group that the data will be placed into
        :type  group:          h5py.Group
        :param metadata_group: HDF5 group which file-level metadata should exist in
        :type  metadata_group: h5py.Group
        :param data_type:      Type of data that is being written.
        :type  data_type:      str
        """
        dataclass_fields = {f.name: f for f in fields(self)}
        dim_scale_fields = [
            f for f in self._dim_scale_fields(data_type) if f in dataclass_fields.keys()
        ]

        # First, writes all fields that are Dimension Scales for other fields
        for f in dim_scale_fields:
            written = self._dispatch_to_write_method(
                f, dataclass_fields, group, metadata_group, data_type
            )
            if written:
                group[f].make_scale(dataclass_fields[f].metadata.get("nickname", f))

        # Then, write the remaining fields, and associate the Dimension Scale fields with them
        remaining_fields = list(set(self.all_fields(data_type)) - set(dim_scale_fields))
        for f in remaining_fields:
            self._dispatch_to_write_method(
                f, dataclass_fields, group, metadata_group, data_type
            )

    def to_dmap(self):
        """
        Converts data from ``self`` into a valid DMAP record.
        """
        group = dict()
        metadata = dict()
        for relevant_field in SliceData.all_fields("rawacf"):
            try:
                data = getattr(self, relevant_field)
            except AttributeError as err:
                if relevant_field in self.required_fields("rawacf"):
                    raise err
                else:
                    continue

            # Massage the data into the correct types
            if isinstance(data, dict):
                data = str(data)
            elif isinstance(data, str):
                data = data.encode("utf-8")
            elif isinstance(data, list):
                if isinstance(data[0], str):
                    data = np.bytes_(data)
                else:
                    data = np.array(data)

            if relevant_field in self._file_level_fields():
                metadata[relevant_field] = data
            else:
                group[relevant_field] = data

        FILENAME = ""  # todo: Use the name of the rawacf file? Or just the timestamp of the start of the file?
        dmap_record = pydarnio.BorealisV1Convert.convert_rawacf_record(
            group, metadata, FILENAME
        )

        return dmap_record

    def __repr__(self):
        """Print all available fields"""
        return f"{vars(self)}"
