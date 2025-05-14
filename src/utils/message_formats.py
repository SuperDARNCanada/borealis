#!/usr/bin/python3
#
# Copyright 2022 SuperDARN Canada
# Author: Remington Rohel
from dataclasses import dataclass, field, fields
import numpy as np


@dataclass
class DebugDataStage:
    """
    Defines a stage of debug data (filtered data or antennas_iq data plus associated metadata).
    """

    stage_name: str = None
    main_shm: str = None
    intf_shm: str = None
    num_samps: int = None


@dataclass
class OutputDataset:
    """
    Defines an output dataset message.
    """

    slice_id: int = None
    num_beams: int = None
    num_ranges: int = None
    num_lags: int = None
    main_acf_shm: str = None
    intf_acf_shm: str = None
    xcf_shm: str = None
    cfs_data: list = field(default_factory=list)


@dataclass
class ProcessedSequenceMessage:
    """
    Defines a message containing metadata about a processed sequence of data.
    This message format is for communication from rx_signal_processing to data_write.
    """

    sequence_num: int = None
    rx_sample_rate: float = None
    output_sample_rate: float = None
    initialization_time: float = None
    sequence_start_time: float = None
    gps_to_system_time_diff: float = None
    agc_status_bank_h: int = None
    lp_status_bank_h: int = None
    agc_status_bank_l: int = None
    lp_status_bank_l: int = None
    gps_locked: bool = None
    bfiq_main_shm: str = None
    bfiq_intf_shm: str = None
    max_num_beams: int = None
    num_samps: int = None
    main_corrs_shm: str = None
    intf_corrs_shm: str = None
    cross_corrs_shm: str = None
    rawrf_shm: str = None
    rawrf_num_samps: int = None
    debug_data: list[DebugDataStage] = field(default_factory=list)
    output_datasets: list[OutputDataset] = field(default_factory=list)
    cfs_freq: list = field(default_factory=list)

    def add_debug_data(self, stage: DebugDataStage):
        """Add a stage of debug data to the message"""
        self.debug_data.append(stage)

    def add_output_dataset(self, data_set: OutputDataset):
        """Add an output dataset to the message"""
        self.output_datasets.append(data_set)


@dataclass
class DecimationStageMessage:
    """Defines a decimation_stage structure within a SequenceMetadataMessage"""

    stage_num: int = None
    input_rate: float = None
    dm_rate: int = None
    filter_taps: list[float] = field(default_factory=list)


@dataclass
class Lag:
    """Defines a lag structure within an RxChannel dataclass"""

    pulse_1: int = None
    pulse_2: int = None
    lag_num: int = None
    phase_offset_real: float = None
    phase_offset_imag: float = None


@dataclass
class RxChannel:
    """Defines an rx_channel structure within a SequenceMetadataMessage"""

    slice_id: int = None
    tau_spacing: int = None
    rx_freq: float = None
    cfs_flag: bool = None
    num_ranges: int = None
    first_range: int = None
    range_sep: float = None
    rx_intf_antennas: list[int] = field(default_factory=list)
    beam_phases: np.ndarray = None
    lags: list[Lag] = field(default_factory=list)
    pulses: list = field(default_factory=list)
    acf: bool = False
    xcf: bool = False
    acfint: bool = False

    def add_lag(self, lag: Lag):
        """Add a Lag dataclass to the message."""
        self.lags.append(lag)


@dataclass
class SequenceMetadataMessage:
    """
    Defines a message containing metadata about a sequence of data.
    This message format is for communication from radar_control to
    rx_signal_processing.
    """

    sequence_num: int = None
    sequence_time: float = None
    offset_to_first_rx_sample: int = None
    rx_rate: float = None
    output_sample_rate: float = None
    rx_ctr_freq: float = None
    decimation_stages: list[DecimationStageMessage] = field(default_factory=list)
    rx_channels: list[RxChannel] = field(default_factory=list)
    acf: bool = False
    xcf: bool = False
    acfint: bool = False
    cfs_scan_flag: bool = False
    cfs_fft_n: int = None

    def add_decimation_stage(self, stage: DecimationStageMessage):
        """Add a decimation stage to the message."""
        self.decimation_stages.append(stage)

    def add_rx_channel(self, channel: RxChannel):
        """Add an rx_channel dataclass to the message."""
        self.rx_channels.append(channel)


@dataclass
class Beam:
    """Defines a beam structure for inclusion in an RxChannelMetadata dataclass"""

    beam_azimuth: float = None
    beam_num: int = None


@dataclass
class LagTable:
    """Defines a ltab structure for inclusion in an RxChannelMetadata dataclass"""

    pulse_position: list[int] = field(default_factory=list)
    lag_num: int = None


@dataclass
class RxChannelMetadata:
    """Defines an RxChannelMetadata structure for inclusion in an AveperiodMetadataMessage"""

    slice_id: int = None
    slice_comment: str = None
    interfacing: str = None
    rx_only: bool = None
    pulse_len: int = None
    tau_spacing: int = None
    rx_freq: float = None
    ptab: list[int] = field(default_factory=list)
    sequence_encodings: list = field(default_factory=list)
    rx_main_antennas: list[int] = field(default_factory=list)
    rx_intf_antennas: list[int] = field(default_factory=list)
    rx_main_excitations: list[complex] = field(default_factory=list)
    rx_intf_excitations: list[complex] = field(default_factory=list)
    tx_antennas: list[int] = field(default_factory=list)
    tx_excitations: list[complex] = field(default_factory=list)
    beams: list[Beam] = field(default_factory=list)
    first_range: float = None
    num_ranges: int = None
    range_sep: int = None
    acf: bool = None
    xcf: bool = None
    acfint: bool = None
    ltabs: list[LagTable] = field(default_factory=list)
    averaging_method: str = None

    def add_sqn_encodings(self, encodings: list):
        """Add a sequence_encodings list to the message."""
        self.sequence_encodings.append(encodings)

    def add_beam(self, beam: Beam):
        """Add a Beam dataclass to the message."""
        self.beams.append(beam)

    def add_ltab(self, ltab: LagTable):
        """Add a LagTable dataclass to the message."""
        self.ltabs.append(ltab)


@dataclass
class Sequence:
    """Defines a sequence structure for inclusion in an AveperiodMetadataMessage"""

    blanks: list[int] = field(default_factory=list)
    output_sample_rate: float = None
    rx_channels: list[RxChannelMetadata] = field(default_factory=list)

    def add_rx_channel(self, channel: RxChannelMetadata):
        """Add an rx channel metadata dataclass to the message."""
        self.rx_channels.append(channel)


@dataclass
class AveperiodMetadataMessage:
    """
    Defines a message containing metadata about an averaging period of data.
    This message format is for communication from radar_control to
    data_write.
    """

    experiment_id: int = None
    experiment_name: str = None
    experiment_comment: str = None
    rx_ctr_freq: float = None
    num_sequences: int = None
    last_sqn_num: int = None
    scan_flag: bool = None
    aveperiod_time: float = None
    input_sample_rate: float = None
    data_normalization_factor: float = None
    scheduling_mode: str = None
    sequences: list = field(default_factory=list)
    cfs_freqs: list = field(default_factory=list)
    cfs_noise: dict = field(default_factory=dict)
    cfs_range: dict = field(default_factory=dict)
    cfs_masks: dict = field(default_factory=dict)
    cfs_slice_ids: list = field(default_factory=list)

    def add_sequence(self, sequence: dict):
        """Add a sequence dict to the message."""
        self.sequences.append(sequence)


@dataclass
class RxSamplesMetadata:
    """
    Message from usrp_driver to rx_signal_processing.
    """

    sequence_num: int = 0
    num_rx_samps: int = 0
    rx_rate: float = 0.0
    sequence_time: float = 0.0
    initialization_time: float = 0.0
    sequence_start_time: float = 0.0
    ringbuffer_size: int = 0
    agc_status_bank_h: int = 0
    lp_status_bank_h: int = 0
    agc_status_bank_l: int = 0
    lp_status_bank_l: int = 0
    gps_locked: bool = False
    gps_to_system_time_diff: float = 0.0

    @staticmethod
    def parse(message: str):
        """Parses a string of `k1=v1 k2=v2` into RxSamplesMetadata"""
        rx_metadata = RxSamplesMetadata()
        split_reply = message.split(" ")  # expect "k1=v1 k2=v2 k3=v3"
        for token in split_reply:
            split_token = token.split("=")
            k = split_token[0]
            v = split_token[1]
            var_type = getattr(rx_metadata, k)
            if isinstance(var_type, bool):
                v = bool(int(v))  # bool("0") -> True, bool(int("0")) -> False
            else:
                v = type(var_type)(v)
            setattr(rx_metadata, k, v)
        return rx_metadata

    def format_for_ipc(self):
        str_list = list()
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                v = int(v)
            str_list.append(f"{f.name}={v}")
        msg_str = " ".join(str_list)
        return msg_str.encode("utf-8")


@dataclass
class DriverPacket:
    sequence_num: int = 0
    rxrate: float = 0.0
    txrate: float = 0.0
    txcenterfreq: float = 0.0
    rxcenterfreq: float = 0.0
    num_rx_samps: int = 0
    num_tx_samps: int = 0
    seqtime: float = 0.0
    sample_timing: float = 0.0
    burst_start: bool = False
    burst_end: bool = False
    align_sequences: bool = False
    buffer_offset: int = 0

    @staticmethod
    def parse(message: str):
        """Parses a string of `k1=v1 k2=v2` into DriverPacket"""
        packet = DriverPacket()
        split_reply = message.split(" ")  # expect "k1=v1 k2=v2 k3=v3"
        for token in split_reply:
            split_token = token.split("=")
            k = split_token[0]
            v = split_token[1]
            var_type = type(getattr(packet, k))
            if var_type is bool:
                v = bool(int(v))  # bool("0") -> True, bool(int("0")) -> False
            else:
                v = var_type(v)
            setattr(packet, k, v)
        return packet

    def format_for_ipc(self):
        str_list = list()
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                v = int(v)
            str_list.append(f"{f.name}={v}")
        msg_str = " ".join(str_list)
        return msg_str.encode("utf-8")
