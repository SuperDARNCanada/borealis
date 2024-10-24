#!/usr/bin/python

"""
averaging_periods
~~~~~~~~~~~~~~~~~
This is the module containing the AveragingPeriod class. The AveragingPeriod class contains the
InterfaceClassBase members, as well as cfs_flag, intn (number of integrations to run),
or intt(max time for integrations), and it contains sequences of class Sequence.

:copyright: 2018 SuperDARN Canada
:author: Marci Detwiller
"""

# built-in
import inspect
from pathlib import Path
import copy

# third party
import numpy as np
import structlog

# local
import borealis_experiments.superdarn_common_fields as scf
from utils.options import Options
from experiment_prototype.interface_classes.sequences import Sequence
from experiment_prototype.interface_classes.interface_class_base import (
    InterfaceClassBase,
)
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.experiment_slice import ExperimentSlice

# Obtain the module name that imported this log_config
caller = Path(inspect.stack()[-1].filename)
module_name = caller.name.split(".")[0]
log = structlog.getLogger(module_name)

"""
Scans are made up of AveragingPeriods, these are typically a 3sec time of the same pulse sequence
pointing in one direction.  AveragingPeriods are made up of Sequences, typically the same sequence
run ave. 20-30 times after a clear frequency search.  Sequences are made up of pulse_time lists,
which give timing, slice, and pulsenumber. CPObject provides channels, pulseshift (if any), freq,
pulse length, beamdir, and wavetype.
"""

options = Options()


class AveragingPeriod(InterfaceClassBase):
    """
    Set up the AveragingPeriods.

    An averagingperiod contains sequences and integrates one or multiple pulse sequences together in
    a given time frame or in a given number of averages, if that is the preferred limiter.

    **The unique members of the averagingperiod are (not a member of the interfaceclassbase):**

    slice_to_beamorder
        passed in by the scan that this AveragingPeriod instance is contained in. A dictionary of
        slice: beam_order for all slices contained in this aveperiod.
    slice_to_beamdir
        passed in by the scan that this AveragingPeriod instance is contained in. A dictionary of
        slice: beamdir(s) for all slices contained in this aveperiod.
    cfs_flag
        Boolean, True if clrfrqsearch should be performed.
    cfs_range
        The range of frequency to search if cfs_flag is True.  Otherwise empty.
    intt
        The priority limitation. The time limit (ms) at which time the aveperiod will end. If None,
        we will use intn to end the aveperiod (a number of sequences).
    intn
        Number of averages (# of times the sequence transmits) to end after for the averagingperiod.
    sequences
        The list of sequences included in this aveperiod. This does not indicate how many averages
        will be transmitted in the aveperiod. If there are multiple sequences in the list, they will
        be alternated between until the end of the aveperiod.
    one_pulse_only
        boolean, True if this averaging period only has one unique set of pulse samples in it. This
        is true if there is only one sequence in the averaging period, and all pulses after the
        first pulse in the sequence have the isarepeat key = True. This boolean can be used to speed
        up the process of sending data to the driver which means we can get more averages in less
        time.
    """

    def __init__(
        self,
        ave_keys,
        ave_slice_dict,
        ave_interface,
        transmit_metadata,
        slice_to_beamorder_dict,
        slice_to_beamdir_dict,
    ):
        InterfaceClassBase.__init__(
            self, ave_keys, ave_slice_dict, ave_interface, transmit_metadata
        )

        self.slice_to_beamorder = slice_to_beamorder_dict
        self.slice_to_beamdir = slice_to_beamdir_dict

        # Metadata for an AveragingPeriod: clear frequency search, integration time, number of averages goal
        self.cfs_flag = False
        self.cfs_always_run = False
        self.cfs_sequence = None
        self.cfs_slice_ids = []
        self.cfs_scan_order = []
        self.cfs_stable_time = 0
        self.cfs_pwr_threshold = 0
        self.cfs_fft_n = 0
        # there may be multiple slices in this averaging period at different frequencies so
        # we may have to search multiple ranges.
        self.cfs_range = []
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id].cfs_flag:
                self.cfs_stable_time = self.slice_dict[slice_id].cfs_stable_time
                self.cfs_pwr_threshold = self.slice_dict[slice_id].cfs_pwr_threshold
                self.cfs_fft_n = self.slice_dict[slice_id].cfs_fft_n
                self.cfs_flag = True
                self.cfs_slice_ids.append(slice_id)
                self.cfs_range.append(self.slice_dict[slice_id].cfs_range)
                if self.slice_dict[slice_id].cfs_always_run:
                    self.cfs_always_run = True

        self.intt = self.slice_dict[self.slice_ids[0]].intt
        self.intn = self.slice_dict[self.slice_ids[0]].intn
        self.txctrfreq = self.slice_dict[self.slice_ids[0]].txctrfreq
        self.rxctrfreq = self.slice_dict[self.slice_ids[0]].rxctrfreq
        if self.intt is not None:  # intt has priority over intn
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id].intt != self.intt:
                    errmsg = (
                        f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                        " interfaced and do not have the same Averaging Period duration intt"
                    )
                    raise ExperimentException(errmsg)
        elif self.intn is not None:
            for slice_id in self.slice_ids:
                if self.slice_dict[slice_id].intn != self.intn:
                    errmsg = (
                        f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                        " interfaced and do not have the same NAVE goal intn"
                    )
                    raise ExperimentException(errmsg)

        for slice_id in self.slice_ids:
            if len(self.slice_dict[slice_id].rx_beam_order) != len(
                self.slice_dict[self.slice_ids[0]].rx_beam_order
            ):
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                    " interfaced but do not have the same number of averaging periods in"
                    " their beam order"
                )
                raise ExperimentException(errmsg)

        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id].txctrfreq != self.txctrfreq:
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                    " interfaced and do not have the same txctrfreq"
                )
                raise ExperimentException(errmsg)
            if self.slice_dict[slice_id].rxctrfreq != self.rxctrfreq:
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                    " interfaced and do not have the same rxctrfreq"
                )
                raise ExperimentException(errmsg)
        for slice_id in self.cfs_slice_ids:
            if self.slice_dict[slice_id].cfs_pwr_threshold != self.cfs_pwr_threshold:
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                    " interfaced and do not have the same cfs_power_threshold"
                )
                raise ExperimentException(errmsg)
            if self.slice_dict[slice_id].cfs_fft_n != self.cfs_fft_n:
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                    " interfaced and do not have the same cfs_fft_n"
                )
                raise ExperimentException(errmsg)
            if self.slice_dict[slice_id].cfs_stable_time != self.cfs_stable_time:
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are SEQUENCE or CONCURRENT"
                    " interfaced and do not have the same cfs_stable_time"
                )
                raise ExperimentException(errmsg)

        self.num_beams_in_scan = len(self.slice_dict[self.slice_ids[0]].rx_beam_order)

        # NOTE: Do not need beam information inside the AveragingPeriod, this is in Scan.

        if self.cfs_flag:
            self.build_cfs_sequence()

        # Determine how this averaging period is made by separating out the SEQUENCE interfaced.
        self.nested_slice_list = self.get_nested_slice_ids()
        self.sequences = []

        self.cfs_sequences = []
        for params in self.prep_for_nested_interface_class():
            new_sequence = Sequence(*params)
            if new_sequence.cfs_flag:
                self.cfs_sequences.append(new_sequence)
            self.sequences.append(new_sequence)

        self.one_pulse_only = False

        self.beam_iter = 0  # used to keep track of place in beam order.

    def set_beamdirdict(self, beamiter):
        """
        Get a dictionary of 'slice_id' : 'beamdir(s)' for this averaging period.

        At a given beam iteration, this averagingperiod instance will select the beam directions
        that it will shift to.

        :param      beamiter:   the index into the beam_order list, or the index of an averaging
                                period into the scan
        :type       beamiter:   int

        :returns:   dictionary of slice to beamdir where beamdir is always a list (may be of length
                    one though). Beamdir is azimuth angle.
        :rtype:     dict
        """

        slice_to_beamdir_dict = {}
        try:
            for slice_id in self.slice_ids:
                beam_number = self.slice_to_beamorder[slice_id][beamiter]
                if isinstance(beam_number, int):
                    beamdir = []
                    beamdir.append(self.slice_to_beamdir[slice_id][beam_number])
                else:  # is a list
                    beamdir = [
                        self.slice_to_beamdir[slice_id][bmnum] for bmnum in beam_number
                    ]
                slice_to_beamdir_dict[slice_id] = beamdir
        except IndexError:
            errmsg = f"Looking for BeamNumber or Beamdir that does not Exist at BeamIter {beamiter}"
            raise ExperimentException(errmsg)

        return slice_to_beamdir_dict

    def select_cfs_freqs(self, cfs_spectrum):
        """
        Accepts the analysis results of the clear frequency search and uses the passed frequencies and powers
        to determine what frequencies to set each clear frequency search slice to.

        * Loops through all sequences that have CFS slices:
           * Records all non-CFS slice frequencies
        * Loops through each CFS slice in the averaging period:
           * Calculates the tx bandwidth for the slice pulses;
           * Creates a mask of the CFS frequency band
           * Masks restricted frequencies
           * Masks already-used frequencies by other slices
           * Masks around the N200 tuned center frequencies (rx and tx)
           * Masks frequencies outside requested CFS range
           * Sorts measured spectrum by power
           * Sets slice frequency to the least powerful frequency
           * Updates already-used frequency list
           * Stores the frequency mask
        * Builds each CFS sequence
        * Return the frequency masks

        :param      cfs_spectrum: Analyzed CFS sequence data
        :type       cfs_spectrum: ProcessedSequenceMessage
        """
        cfs_freq_hz = np.array(cfs_spectrum.cfs_freq)  # at baseband
        cfs_data = [dset.cfs_data for dset in cfs_spectrum.output_datasets]
        # Sort measured frequencies based on measured power at each freq
        slice_masks = dict()
        slice_used_freqs = dict()
        # {cfs_slice_id: list of used frequencies that the slice must avoid}

        cfs_slice_set = set(self.cfs_slice_ids)
        cfs_set_freq = dict()

        for sequence in self.cfs_sequences:
            used_range = []
            sqn_slice_ids = set(sequence.slice_ids)
            sqn_cfs_ids = sqn_slice_ids.intersection(cfs_slice_set)
            for slice_id in sqn_slice_ids.difference(cfs_slice_set):
                slice_obj = sequence.slice_dict[slice_id]
                df = int(round(1e3 / (2 * slice_obj.pulse_len)))
                # bandwidth of tx pulse / 2 in kHz
                used_range.append([slice_obj.freq - df, slice_obj.freq + df])
                # record pulse widths of all non-cfs slices in use
            slice_used_freqs.update({k: used_range for k in sqn_cfs_ids})

        for i, slice_id in enumerate(self.cfs_slice_ids):
            slice_obj = self.slice_dict[slice_id]

            df = int(round(1e3 / (2 * slice_obj.pulse_len)))
            slice_range = slice_obj.cfs_range
            center_freq_khz = int((slice_range[0] + slice_range[1]) / 2)
            shifted_cfs_khz = cfs_freq_hz / 1000 + center_freq_khz

            mask = np.full(len(shifted_cfs_khz), True)
            for f_rng in options.restricted_ranges:
                mask[
                    np.argwhere(
                        np.logical_and(
                            shifted_cfs_khz >= np.floor(f_rng[0]),
                            shifted_cfs_khz <= np.ceil(f_rng[1]),
                        )
                    )
                ] = False
                # Rounding when setting the freq below could cause the freq to set to a restricted value
                # so ceil and floor are used to ensure restricted ranges are avoided

            for tx_range in slice_used_freqs[slice_id]:
                mask[
                    np.argwhere(
                        np.logical_and(
                            shifted_cfs_khz >= tx_range[0] - df,
                            shifted_cfs_khz <= tx_range[1] + df,
                        )
                    )
                ] = False
                # Mask pulse width around all used frequencies

            for ctr_freq in [self.txctrfreq, self.rxctrfreq]:
                mask[
                    np.argwhere(
                        np.logical_and(
                            shifted_cfs_khz >= np.floor(ctr_freq - 50),
                            shifted_cfs_khz <= np.ceil(ctr_freq + 50),
                        )
                    )
                ] = False
                # Avoid frequencies within 50 kHz for the center freqs

            f_rng = slice_obj.cfs_range
            mask[
                np.argwhere(
                    np.logical_or(
                        shifted_cfs_khz < f_rng[0], shifted_cfs_khz > f_rng[1]
                    )
                )
            ] = False
            shifted_cfs_khz = shifted_cfs_khz[mask]
            # Mask all restricted frequencies, all frequencies within the tx pulse
            # of used frequencies, frequencies outside cfs_range, and frequencies
            # too close to the center frequencies in the cfs spectrum frequencies

            if len(shifted_cfs_khz) < 1:
                log.critical(
                    "All searched frequencies were too close to used frequencies or the tx or rx"
                    "center frequencies or were restricted. The radar will crash!!!",
                    current_slice_id=slice_obj.slice_id,
                    used_freq_tx_widths=slice_used_freqs[slice_id],
                    cfs_sorted_freqs=shifted_cfs_khz,
                )

            slice_masks[slice_id] = mask
            ind = np.argmin(cfs_data[i][mask])
            cfs_set_freq[slice_id] = int(np.round(shifted_cfs_khz[ind]))

            for sqn in self.cfs_sequences:
                if slice_id in sqn.slice_ids:
                    other_ids = set(sqn.slice_ids).intersection(self.cfs_slice_ids)
                    other_ids.remove(slice_id)
                    for other_id in other_ids:
                        slice_used_freqs[other_id].append(
                            [
                                cfs_set_freq[slice_id] - df,
                                cfs_set_freq[slice_id] + df,
                            ]
                        )
            # Set cfs slice frequency and add frequency to used_freqs for all other concurrent slices

        self.update_cfs_freqs(cfs_set_freq)

        return slice_masks, cfs_set_freq

    def update_cfs_freqs(self, cfs_set_freq):
        for i, slice_id in enumerate(self.cfs_slice_ids):
            slice_obj = self.slice_dict[slice_id]
            slice_obj.freq = cfs_set_freq[slice_id]
            log.verbose(
                "selecting cfs slice freq",
                slice_id=slice_obj.slice_id,
                set_freq=slice_obj.freq,
            )
        for sequence in self.cfs_sequences:
            sequence.build_sequence_pulses()

    def build_cfs_sequence(self):
        """
        Builds an empty rx only pulse sequence to collect clear frequency search data
        """
        pulse_length = 100  # us
        slice_id = self.cfs_slice_ids[0]  # get first cfs slice id
        num_ranges = int(
            round((self.slice_dict[slice_id].cfs_duration * 1000) / pulse_length)
        )
        # calculate number of ranges (cfs_duration is in ms)

        # Create a CFS slice for the pulse
        default_slice = {
            "cpid": self.slice_dict[slice_id].cpid,
            "slice_id": 0,
            "transition_bandwidth": self.slice_dict[slice_id].transition_bandwidth,
            "rx_bandwidth": self.slice_dict[slice_id].rx_bandwidth,
            "tx_bandwidth": self.slice_dict[slice_id].tx_bandwidth,
            "txctrfreq": self.txctrfreq,
            "rxctrfreq": self.rxctrfreq,
            "rxonly": True,
            "pulse_sequence": [0],
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": pulse_length,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intn": 1,  # number of integration times
            "beam_angle": [0.0],
            "rx_beam_order": [0],
            "freq": None,  # kHz
            "decimation_scheme": self.slice_dict[slice_id].cfs_scheme,
        }

        cfs_slices = {}
        seq_keys = []
        slice_counter = 0
        self.cfs_scan_order = []
        for cfs_id in self.cfs_slice_ids:
            listening_slice = copy.deepcopy(default_slice)
            slice_range = self.slice_dict[cfs_id].cfs_range
            listening_slice["freq"] = int((slice_range[0] + slice_range[1]) / 2)
            listening_slice["slice_id"] = slice_counter
            self.cfs_scan_order.append(cfs_id)

            cfs_slices[slice_counter] = ExperimentSlice(**listening_slice)
            seq_keys.append(slice_counter)
            slice_counter += 1

        # Build interface dictionary
        interface_dict = {}
        for i in range(slice_counter):
            for j in range(i + 1, slice_counter):
                interface_dict[(i, j)] = "CONCURRENT"

        self.cfs_sequence = Sequence(
            seq_keys, cfs_slices, interface_dict, self.transmit_metadata
        )
