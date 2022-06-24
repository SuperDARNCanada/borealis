#!/usr/bin/python3
#
# Copyright 2022 SuperDARN Canada


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

        self.__stage_allowed_items = {'stage_name': str, 'main_shm': str, 'intf_shm': str, 'num_samps': int}

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

    def add_stage(self, stage: dict):
        """Add a stage of debug data to the message"""
        for key, val in stage.items():
            if key not in self.__stage_allowed_items.keys():
                raise KeyError("Invalid key {} when adding stage to ProcessedSequenceMessage".format(key))
            if not isinstance(val, self.__stage_allowed_items[key]):
                raise ValueError("Value of '{}' is not a {}".format(key, self.__stage_allowed_items[key]))

        self._debug_data.append(stage)



