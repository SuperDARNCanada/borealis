"""
Aggregator class for collecting data from each sequence within an averaging period.
Once the averaging period is complete, the data is merged into numpy arrays.
"""

import collections
import copy
from dataclasses import dataclass, fields, field
from multiprocessing import shared_memory

import numpy as np

from options import Options
from message_formats import ProcessedSequenceMessage


@dataclass
class Aggregator:
    """
    This class is for aggregating data during an averaging period.
    """

    agc_status_word: int = 0b0
    antenna_iq_accumulator: dict = field(default_factory=dict)
    antenna_iq_available: bool = False
    bfiq_accumulator: dict = field(default_factory=dict)
    bfiq_available: bool = False
    intfacfs_available: bool = False
    gps_locked: bool = (
        True  # init True so that logical AND works properly in update() method
    )
    gps_to_system_time_diff: float = 0.0
    intfacfs_accumulator: dict = field(default_factory=dict)
    lp_status_word: int = 0b0
    mainacfs_accumulator: dict = field(default_factory=dict)
    mainacfs_available: bool = False
    options: Options = None
    processed_data: ProcessedSequenceMessage = field(init=False)
    rawrf_available: bool = False
    rawrf_locations: list[str] = field(default_factory=list)
    rawrf_num_samps: int = 0
    sequence_num: int = field(init=False)
    slice_ids: set = field(default_factory=set)
    timestamps: list[float] = field(default_factory=list)
    xcfs_accumulator: dict = field(default_factory=dict)
    xcfs_available: bool = False

    def _get_accumulators(self):
        """Returns a list of all accumulator dictionaries in this object."""
        accumulators = []
        for f in fields(self):
            name = f.name
            if "accumulator" in name:
                accumulators.append(getattr(self, name))
        return accumulators

    def parse_correlations(self):
        """
        Parses out the possible correlation (acf/xcf) data from the message. Runs on every new
        ProcessedSequenceMessage (contains all sampling period data). The expectation value is
        calculated at the end of an averaging period by a different function.
        """

        for data_set in self.processed_data.output_datasets:
            slice_id = data_set.slice_id
            data_shape = (data_set.num_beams, data_set.num_ranges, data_set.num_lags)

            def accumulate_data(accumulator: dict, message_data: str):
                """
                Opens a numpy array from shared memory into the accumulator.

                :param  accumulator:    accumulator to hold data
                :type   accumulator:    dict
                :param  message_data:   unique message field for parsing
                :type   message_data:   str
                """

                # Open the shared memory
                shm = shared_memory.SharedMemory(name=message_data)
                acf_data = np.ndarray(data_shape, dtype=np.complex64, buffer=shm.buf)

                # Put the data in the accumulator
                if "data" not in accumulator[slice_id]:
                    accumulator[slice_id]["data"] = []
                accumulator[slice_id]["data"].append(acf_data.copy())
                shm.close()
                shm.unlink()

            if data_set.main_acf_shm:
                self.mainacfs_available = True
                accumulate_data(self.mainacfs_accumulator, data_set.main_acf_shm)

            if data_set.xcf_shm:
                self.xcfs_available = True
                accumulate_data(self.xcfs_accumulator, data_set.xcf_shm)

            if data_set.intf_acf_shm:
                self.intfacfs_available = True
                accumulate_data(self.intfacfs_accumulator, data_set.intf_acf_shm)

    def parse_bfiq(self):
        """
        Parses out any possible beamformed IQ data from the message. Runs on every
        ProcessedSequenceMessage (contains all sampling period data).
        """
        num_slices = len(self.processed_data.output_datasets)
        max_num_beams = self.processed_data.max_num_beams
        num_samps = self.processed_data.num_samps

        def extract_for_array(name: str):
            """
            Extracts data from shared memory and adds it to the appropriate accumulator
            """
            shm = shared_memory.SharedMemory(
                name=getattr(self.processed_data, f"bfiq_{name}_shm")
            )
            temp_data = np.ndarray(
                (num_slices, max_num_beams, num_samps),
                dtype=np.complex64,
                buffer=shm.buf,
            )
            data = temp_data.copy()
            shm.close()
            shm.unlink()

            for i, data_set in enumerate(self.processed_data.output_datasets):
                slice_id = data_set.slice_id
                num_beams = data_set.num_beams

                if f"{name}_data" not in self.bfiq_accumulator[slice_id]:
                    self.bfiq_accumulator[slice_id][f"{name}_data"] = []
                self.bfiq_accumulator[slice_id][f"{name}_data"].append(
                    data[i, :num_beams, :]
                )

        extract_for_array("main")
        if self.processed_data.bfiq_intf_shm:
            extract_for_array("intf")

        self.bfiq_available = True

    def parse_antenna_iq(self):
        """
        Parses out any pre-beamformed IQ if available. Runs on every ProcessedSequenceMessage
        (contains all sampling period data).
        """
        # Get data dimensions for reading in the shared memory
        num_slices = len(self.processed_data.output_datasets)
        num_main_antennas = len(self.options.rx_main_antennas)
        num_intf_antennas = len(self.options.rx_intf_antennas)

        stages = []
        # Loop through all the filter stage data
        for debug_stage in self.processed_data.debug_data:
            stage_samps = debug_stage.num_samps
            stage_main_shm = shared_memory.SharedMemory(name=debug_stage.main_shm)
            stage_main_data = np.ndarray(
                (num_slices, num_main_antennas, stage_samps),
                dtype=np.complex64,
                buffer=stage_main_shm.buf,
            )
            stage_data = (
                stage_main_data.copy()
            )  # Move data out of shared memory so we can close it
            stage_main_shm.close()
            stage_main_shm.unlink()

            if debug_stage.intf_shm:
                stage_intf_shm = shared_memory.SharedMemory(name=debug_stage.intf_shm)
                stage_intf_data = np.ndarray(
                    (num_slices, num_intf_antennas, stage_samps),
                    dtype=np.complex64,
                    buffer=stage_intf_shm.buf,
                )
                stage_data = np.hstack((stage_data, stage_intf_data.copy()))
                stage_intf_shm.close()
                stage_intf_shm.unlink()

            stage_dict = {
                "stage_name": debug_stage.stage_name,
                "stage_samps": debug_stage.num_samps,
                "main_shm": debug_stage.main_shm,
                "intf_shm": debug_stage.intf_shm,
                "data": stage_data,
            }
            stages.append(stage_dict)

        self.antenna_iq_available = True

        # Iterate over every data set, one data set per slice
        for i, data_set in enumerate(self.processed_data.output_datasets):
            slice_id = data_set.slice_id

            # non beamformed IQ samples are available
            for debug_stage in stages:
                stage_name = debug_stage["stage_name"]

                if stage_name not in self.antenna_iq_accumulator[slice_id]:
                    self.antenna_iq_accumulator[slice_id][stage_name] = (
                        collections.OrderedDict()
                    )

                antenna_iq_stage = self.antenna_iq_accumulator[slice_id][stage_name]

                antennas_data = debug_stage["data"][i]
                antenna_iq_stage["num_samps"] = antennas_data.shape[-1]

                # All possible antenna numbers, given the config file
                antenna_indices = copy.deepcopy(self.options.rx_main_antennas)
                # The interferometer antenna numbers start after the last main antenna number
                antenna_indices.extend(
                    [
                        ant + self.options.main_antenna_count
                        for ant in self.options.rx_intf_antennas
                    ]
                )

                # Loops over antenna data within stage
                for ant_num in range(antennas_data.shape[0]):
                    # Convert index in the data array to antenna number from the config file
                    ant_name = antenna_indices[ant_num]

                    if ant_name not in antenna_iq_stage:
                        antenna_iq_stage[ant_name] = {}

                    if "data" not in antenna_iq_stage[ant_name]:
                        antenna_iq_stage[ant_name]["data"] = []
                    antenna_iq_stage[ant_name]["data"].append(antennas_data[ant_num, :])

    def finalize(self):
        """
        Consolidates data for each data type to one array.

        In parse_[type](), new data arrays are appended to a list for speed considerations.
        This function converts these lists into numpy arrays.
        """
        for slice_id, slice_data in self.antenna_iq_accumulator.items():
            for param_data in slice_data.values():
                for array_name, array_data in param_data.items():
                    if array_name != "num_samps":
                        array_data["data"] = np.array(
                            array_data["data"], dtype=np.complex64
                        )

        for slice_id, slice_data in self.bfiq_accumulator.items():
            for param_name, param_data in slice_data.items():
                if param_name == "num_samps":
                    slice_data[param_name] = param_data
                else:
                    slice_data[param_name] = np.array(param_data, dtype=np.complex64)

        for slice_data in self.mainacfs_accumulator.values():
            slice_data["data"] = np.array(slice_data.get("data", []), np.complex64)

        for slice_data in self.intfacfs_accumulator.values():
            slice_data["data"] = np.array(slice_data.get("data", []), np.complex64)

        for slice_data in self.xcfs_accumulator.values():
            slice_data["data"] = np.array(slice_data.get("data", []), np.complex64)

    def update(self, data):
        """
        Parses the message and updates the accumulator fields with the new data.

        :param  data: Processed sequence from rx_signal_processing module.
        :type   data: ProcessedSequenceMessage
        """
        self.processed_data = data
        self.sequence_num = data.sequence_num
        self.timestamps.append(data.sequence_start_time)

        for data_set in data.output_datasets:
            self.slice_ids.add(data_set.slice_id)

            for accumulator in self._get_accumulators():
                if data_set.slice_id not in accumulator.keys():
                    accumulator[data_set.slice_id] = {}

        if data.rawrf_shm != "":
            self.rawrf_available = True
            self.rawrf_num_samps = data.rawrf_num_samps
            self.rawrf_locations.append(data.rawrf_shm)

        # Logical AND to catch any time the GPS may have been unlocked during the integration period
        self.gps_locked = self.gps_locked and data.gps_locked

        # Find the max time diff between GPS and system time to report for this integration period
        if abs(self.gps_to_system_time_diff) < abs(data.gps_to_system_time_diff):
            self.gps_to_system_time_diff = data.gps_to_system_time_diff

        # Bitwise OR to catch any AGC faults during the integration period
        self.agc_status_word = self.agc_status_word | data.agc_status_bank_h

        # Bitwise OR to catch any low power conditions during the integration period
        self.lp_status_word = self.lp_status_word | data.lp_status_bank_h

        self.parse_correlations()
        self.parse_bfiq()
        self.parse_antenna_iq()
