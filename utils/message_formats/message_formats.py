#!/usr/bin/python3
#
# Copyright 2022 SuperDARN Canada
import numpy as np


def check_dict(data: dict, types: dict, identifier: str):
    """Checks that the entries of a dictionary are of the correct types."""
    for key, val in data.items():
        if key not in types.keys():
            raise KeyError("Invalid key {} when adding {}".format(key, identifier))
        if not isinstance(val, types[key]):
            raise ValueError("Value of '{}' is not type {}".format(key, types[key]))


class ProcessedSequenceMessage(object):
    """
    Defines a message containing metadata about a processed sequence of data.
    This message format is for communication from rx_signal_processing to
    data_write.
    """
    def __init__(self):
        super().__init__()

        self._sequence_num = 0
        self._rx_sample_rate = 0.0
        self._output_sample_rate = 0.0
        self._initialization_time = 0.0
        self._sequence_start_time = 0.0
        self._gps_to_system_time_diff = 0.0
        self._agc_status_bank_h = 0b0
        self._lp_status_bank_h = 0b0
        self._agc_status_bank_l = 0b0
        self._lp_status_bank_l = 0b0
        self._gps_locked = False
        self._bfiq_main_shm = ""
        self._bfiq_intf_shm = ""
        self._max_num_beams = 0
        self._num_samps = 0
        self._main_corrs_shm = ""
        self._intf_corrs_shm = ""
        self._cross_corrs_shm = ""
        self._rawrf_shm = ""
        self._rawrf_num_samps = 0
        self._debug_data = []

        self.__stage_allowed_types = {'stage_name': str, 'main_shm': str, 'intf_shm': str, 'num_samps': int}

    @property
    def sequence_num(self):
        return self._sequence_num

    @sequence_num.setter
    def sequence_num(self, num: int):
        self._sequence_num = num

    @property
    def rx_sample_rate(self):
        return self._rx_sample_rate

    @rx_sample_rate.setter
    def rx_sample_rate(self, rate: float):
        self._rx_sample_rate = rate

    @property
    def output_sample_rate(self):
        return self._output_sample_rate

    @output_sample_rate.setter
    def output_sample_rate(self, rate: float):
        self._output_sample_rate = rate

    @property
    def initialization_time(self):
        return self._initialization_time

    @initialization_time.setter
    def initialization_time(self, time: float):
        self._initialization_time = time

    @property
    def sequence_start_time(self):
        return self._sequence_start_time

    @sequence_start_time.setter
    def sequence_start_time(self, time: float):
        self._sequence_start_time = time

    @property
    def gps_to_system_time_diff(self):
        return self._gps_to_system_time_diff

    @gps_to_system_time_diff.setter
    def gps_to_system_time_diff(self, time: float):
        self._gps_to_system_time_diff = time

    @property
    def agc_status_bank_h(self):
        return self._agc_status_bank_h

    @agc_status_bank_h.setter
    def agc_status_bank_h(self, status: int):
        self._agc_status_bank_h = status

    @property
    def agc_status_bank_l(self):
        return self._agc_status_bank_l

    @agc_status_bank_l.setter
    def agc_status_bank_l(self, status: int):
        self._agc_status_bank_l = status

    @property
    def lp_status_bank_h(self):
        return self._lp_status_bank_h

    @lp_status_bank_h.setter
    def lp_status_bank_h(self, status: int):
        self._lp_status_bank_h = status

    @property
    def lp_status_bank_l(self):
        return self._lp_status_bank_l

    @lp_status_bank_l.setter
    def lp_status_bank_l(self, status: int):
        self._lp_status_bank_l = status

    @property
    def gps_locked(self):
        return self._gps_locked

    @gps_locked.setter
    def gps_locked(self, lock: bool):
        self._gps_locked = lock

    @property
    def bfiq_main_shm(self):
        return self._bfiq_main_shm

    @bfiq_main_shm.setter
    def bfiq_main_shm(self, shm_name: str):
        self._bfiq_main_shm = shm_name

    @property
    def bfiq_intf_shm(self):
        return self._bfiq_intf_shm

    @bfiq_intf_shm.setter
    def bfiq_intf_shm(self, shm_name: str):
        self._bfiq_intf_shm = shm_name

    @property
    def max_num_beams(self):
        return self._max_num_beams

    @max_num_beams.setter
    def max_num_beams(self, num: int):
        self._max_num_beams = num

    @property
    def num_samps(self):
        return self._num_samps

    @num_samps.setter
    def num_samps(self, num: int):
        self._num_samps = num

    @property
    def main_corrs_shm(self):
        return self._main_corrs_shm

    @main_corrs_shm.setter
    def main_corrs_shm(self, shm_name: str):
        self._main_corrs_shm = shm_name

    @property
    def intf_corrs_shm(self):
        return self._intf_corrs_shm

    @intf_corrs_shm.setter
    def intf_corrs_shm(self, shm_name: str):
        self._intf_corrs_shm = shm_name

    @property
    def cross_corrs_shm(self):
        return self._cross_corrs_shm

    @cross_corrs_shm.setter
    def cross_corrs_shm(self, shm_name: str):
        self._cross_corrs_shm = shm_name

    @property
    def rawrf_shm(self):
        return self._rawrf_shm

    @rawrf_shm.setter
    def rawrf_shm(self, shm_name: str):
        self._rawrf_shm = shm_name

    @property
    def rawrf_num_samps(self):
        return self._rawrf_num_samps

    @rawrf_num_samps.setter
    def rawrf_num_samps(self, num: int):
        self._rawrf_num_samps = num

    @property
    def debug_data(self):
        return self._debug_data

    def remove_all_debug_data(self):
        """Remove all debug_data entries so the list can be refilled for the next sequence"""
        self._debug_data = []

    def add_stage(self, stage: dict):
        """Add a stage of debug data to the message"""
        check_dict(stage, self.__stage_allowed_types, 'stage')
        self._debug_data.append(stage)


class SequenceMetadataMessage(object):
    """
    Defines a message containing metadata about a sequence of data.
    This message format is for communication from radar_control to
    rx_signal_processing.
    """
    def __init__(self):
        super().__init__()

        self._sequence_num = 0
        self._sequence_time = 0.0
        self._offset_to_first_rx_sample = 0
        self._rx_rate = 0.0
        self._output_sample_rate = 0.0
        self._rx_ctr_freq = 0.0
        self._decimation_stages = []
        self._rx_channels = []
        self.__decimation_stage_allowed_types = {'stage_num': str, 'input_rate': float, 'dm_rate': float,
                                                 'filter_taps': np.ndarray}

        self.__rx_channel_allowed_types = {'slice_id': str, 'tau_spacing': int, 'rx_freq': float, 'clrfrqflag': bool,
                                           'num_ranges': int, 'first_range': int, 'range_sep': int,
                                           'beam_directions': list, 'lags': list}
        self.__beam_directions_allowed_types = {'phase': list}
        self.__phase_allowed_types = {'real_phase': float, 'imag_phase': float}
        self.__lag_allowed_types = {'pulse_1': int, 'pulse_2': int, 'lag_num': int, 'phase_offset_real': float,
                                    'phase_offset_imag': float}

    @property
    def sequence_num(self):
        return self._sequence_num

    @sequence_num.setter
    def sequence_num(self, num: int):
        self._sequence_num = num

    @property
    def sequence_time(self):
        return self._sequence_time

    @sequence_time.setter
    def sequence_time(self, time: float):
        self._sequence_time = time

    @property
    def offset_to_first_rx_sample(self):
        return self._offset_to_first_rx_sample

    @offset_to_first_rx_sample.setter
    def offset_to_first_rx_sample(self, offset: int):
        self._offset_to_first_rx_sample = offset

    @property
    def rx_rate(self):
        return self._rx_rate

    @rx_rate.setter
    def rx_rate(self, rate: float):
        self._rx_rate = rate

    @property
    def output_sample_rate(self):
        return self._output_sample_rate

    @output_sample_rate.setter
    def output_sample_rate(self, rate: float):
        self._output_sample_rate = rate

    @property
    def rx_ctr_freq(self):
        return self._rx_ctr_freq

    @rx_ctr_freq.setter
    def rx_ctr_freq(self, freq: float):
        self._rx_ctr_freq = freq

    @property
    def decimation_stages(self):
        return self._decimation_stages

    def add_decimation_stage(self, stage: dict):
        """Add a decimation stage to the message."""
        check_dict(stage, self.__decimation_stage_allowed_types, 'decimation_stage')
        self._decimation_stages.append(stage)

    def remove_all_decimation_stages(self):
        """Remove all decimation_stage entries so the list can be refilled for the next sequence"""
        self._decimation_stages = []

    @property
    def rx_channels(self):
        return self._rx_channels

    def remove_all_rx_channels(self):
        """Remove all rx_channel entries so the list can be refilled for the next sequence"""
        self._rx_channels = []

    def add_rx_channel(self, channel: dict):
        """Add an rx_channel dict to the message."""
        check_dict(channel, self.__rx_channel_allowed_types, 'rx_channel')
        if 'beam_directions' in channel.keys():
            for bd in channel['beam_directions']:
                check_dict(bd, self.__beam_directions_allowed_types, 'beam_direction')
                if 'phase' in bd.keys():
                    for phase in bd['phase']:
                        check_dict(phase, self.__phase_allowed_types, 'phase')
        self._rx_channels.append(channel)

    def _check_lags(self, lag: dict):
        """Check that all items in a lag dict are valid"""
        check_dict(lag, self.__lag_allowed_types, 'lag')


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
        if 'rx_channel' in sequence.keys():
            for channel in sequence['rx_channel']:
                check_dict(channel, self.__rx_channel_allowed_types, 'rx_channel')
                if 'beams' in channel.keys():
                    for beam in channel['beams']:
                        check_dict(beam, self.__beam_allowed_types, 'beam')
                if 'ltab' in channel.keys():
                    for ltab in channel['ltab']:
                        check_dict(ltab, self.__ltab_allowed_types, 'lag_table')

        self._sequences.append(sequence)

