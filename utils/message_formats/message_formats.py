#!/usr/bin/python3
#
# Copyright 2022 SuperDARN Canada
# Author: Remington Rohel
from dataclasses import dataclass, field
from typing import List, Any

import numpy as np


def check_dict(data: dict, types: dict, identifier: str):
    """Checks that the entries of a dictionary are of the correct types."""
    for key, val in data.items():
        if key not in types.keys():
            raise KeyError("Invalid key {} when adding {}".format(key, identifier))
        if not isinstance(val, types[key]):
            raise ValueError("Value of '{}' is not type {}".format(key, types[key]))


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

    def remove_all_debug_data(self):
        """Remove all debug_data entries so the list can be refilled for the next sequence"""
        self.debug_data = []

    def add_debug_data(self, stage: DebugDataStage):
        """Add a stage of debug data to the message"""
        self.debug_data.append(stage)

    def remove_all_output_datasets(self):
        """Remove all output_dataset entries so the list can be refilled for the next sequence"""
        self.output_datasets = []

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
class Phase:
    """Defines a phase structure within a BeamDirections dataclass"""
    real_phase: float = None
    imag_phase: float = None


@dataclass
class BeamDirection:
    """Defines a beam_direction structure within an RxChannel dataclass"""
    phase: list[Phase] = field(default_factory=list)

    def remove_all_phases(self):
        """Remove all phase entries so the list can be refilled for the next sequence"""
        self.phase = []

    def add_phase(self, channel: Phase):
        """Add a Phase structure to the dataclass."""
        self.phase.append(channel)


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
    clrfrqflag: bool = None
    num_ranges: int = None
    first_range: int = None
    range_sep: float = None
    beam_directions: list[BeamDirection] = field(default_factory=list)
    lags: list[Lag] = field(default_factory=list)

    def remove_all_beam_directions(self):
        """Remove all beam_direction entries so the list can be refilled for the next sequence"""
        self.beam_directions = []

    def add_beam_direction(self, beam: BeamDirection):
        """Add a BeamDirection dataclass to the message."""
        self.beam_directions.append(beam)

    def remove_all_lags(self):
        """Remove all lag entries so the list can be refilled for the next sequence"""
        self.lags = []

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

    def add_decimation_stage(self, stage: DecimationStageMessage):
        """Add a decimation stage to the message."""
        self.decimation_stages.append(stage)

    def remove_all_decimation_stages(self):
        """Remove all decimation_stage entries so the list can be refilled for the next sequence"""
        self.decimation_stages = []

    def remove_all_rx_channels(self):
        """Remove all rx_channel entries so the list can be refilled for the next sequence"""
        self.rx_channels = []

    def add_rx_channel(self, channel: RxChannel):
        """Add an rx_channel dict to the message."""
        self.rx_channels.append(channel)


class AveperiodMetadataMessage(object):
    """
    Defines a message containing metadata about an averaging period of data.
    This message format is for communication from radar_control to
    data_write.
    """
    def __init__(self):
        super().__init__()

        self._experiment_id = 0
        self._experiment_name = ""
        self._experiment_comment = ""
        self._rx_ctr_freq = 0.0
        self._num_sequences = 0
        self._last_sqn_num = 0
        self._scan_flag = False
        self._aveperiod_time = 0.0
        self._output_sample_rate = 0.0
        self._data_normalization_factor = 0.0
        self._scheduling_mode = ""
        self._sequences = []

        self.__sequence_allowed_types = {'blanks': list, 'tx_data': dict, 'rx_channels': list}
        self.__tx_data_allowed_types = {'tx_rate': float, 'tx_ctr_freq': float, 'pulse_timing_us': float,
                                        'pulse_sample_start': int, 'tx_samples': np.ndarray, 'dm_rate': int,
                                        'decimated_tx_samples': np.ndarray}
        self.__rx_channel_allowed_types = {'slice_id': int, 'slice_comment': str, 'interfacing': str, 'rx_only': bool,
                                           'pulse_len': int, 'tau_spacing': int, 'rx_freq': float, 'ptab': list,
                                           'sequence_encodings': list, 'rx_main_antennas': list,
                                           'rx_intf_antennas': list, 'beams': list, 'first_range': int,
                                           'num_ranges': int, 'range_sep': int, 'acf': bool, 'xcf': bool,
                                           'acfint': bool, 'ltab': dict, 'averaging_method': str}
        self.__beam_allowed_types = {'beam_azimuth': float, 'beam_num': int}
        self.__ltab_allowed_types = {'pulse_position': list, 'lag_num': int}

    @property
    def experiment_id(self):
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, exp_id: int):
        self._experiment_id = exp_id

    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, name: str):
        self._experiment_name = name

    @property
    def experiment_comment(self):
        return self._experiment_comment

    @experiment_comment.setter
    def experiment_comment(self, comment: str):
        self._experiment_comment = comment

    @property
    def rx_ctr_freq(self):
        return self._rx_ctr_freq

    @rx_ctr_freq.setter
    def rx_ctr_freq(self, freq: float):
        self._rx_ctr_freq = freq

    @property
    def num_sequences(self):
        return self._num_sequences

    @num_sequences.setter
    def num_sequences(self, num: int):
        self._num_sequences = num

    @property
    def last_sqn_num(self):
        return self._last_sqn_num

    @last_sqn_num.setter
    def last_sqn_num(self, num: int):
        self._last_sqn_num = num

    @property
    def scan_flag(self):
        return self._scan_flag

    @scan_flag.setter
    def scan_flag(self, flag: bool):
        self._scan_flag = flag

    @property
    def aveperiod_time(self):
        return self._aveperiod_time

    @aveperiod_time.setter
    def aveperiod_time(self, time: float):
        self._aveperiod_time = time

    @property
    def output_sample_rate(self):
        return self._output_sample_rate

    @output_sample_rate.setter
    def output_sample_rate(self, rate: float):
        self._output_sample_rate = rate

    @property
    def data_normalization_factor(self):
        return self._data_normalization_factor

    @data_normalization_factor.setter
    def data_normalization_factor(self, factor: float):
        self._data_normalization_factor = factor

    @property
    def scheduling_mode(self):
        return self._scheduling_mode

    @scheduling_mode.setter
    def scheduling_mode(self, mode: str):
        self._scheduling_mode = mode

    @property
    def sequences(self):
        return self._sequences

    def remove_all_sequences(self):
        """Remove all sequence entries so the list can be refilled for the next averaging period."""
        self._sequences = []

    def add_sequence(self, sequence: dict):
        """Add a sequence dict to the message."""
        check_dict(sequence, self.__sequence_allowed_types, 'sequence')
        if 'tx_data' in sequence.keys():
            check_dict(sequence['tx_data'], self.__tx_data_allowed_types, 'tx_data')
        if 'rx_channels' in sequence.keys():
            for channel in sequence['rx_channels']:
                check_dict(channel, self.__rx_channel_allowed_types, 'rx_channels')
                if 'beams' in channel.keys():
                    for beam in channel['beams']:
                        check_dict(beam, self.__beam_allowed_types, 'beam')
                if 'ltab' in channel.keys():
                    for ltab in channel['ltab']:
                        check_dict(ltab, self.__ltab_allowed_types, 'lag_table')

        self._sequences.append(sequence)

