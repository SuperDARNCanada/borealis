#!/usr/bin/python

"""
    averaging_periods
    ~~~~~~~~~~~~~~~~~
    This is the module containing the AveragingPeriod class. The AveragingPeriod class contains the
    InterfaceClassBase members, as well as cfs_flag (to be implemented), intn (number of integrations
    to run), or intt(max time for integrations), and it contains sequences of class Sequence.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
# built-in
import inspect
from pathlib import Path

# third party
import numpy as np
import structlog

# local
from utils.options import Options
from experiment_prototype.interface_classes.sequences import Sequence
from experiment_prototype.interface_classes.interface_class_base import (
    InterfaceClassBase,
)
from experiment_prototype.experiment_exception import ExperimentException

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
        self.cfs_sequence = None
        self.cfs_slice_ids = []
        # there may be multiple slices in this averaging period at different frequencies so
        # we may have to search multiple ranges.
        self.cfs_range = []
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id].cfs_flag:
                self.cfs_flag = True
                self.cfs_slice_ids.append(slice_id)
                self.cfs_range.append(self.slice_dict[slice_id].cfs_range)

        # TODO: SET UP CLEAR FREQUENCY SEARCH CAPABILITY
        # also note for when setting this up cfs_ranges may overlap

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

    def update_cfs_freqs(self, cfs_spectrum):
        """
        Accepts the analysis results of the clear frequency search and uses the passed frequencies and powers
        to determine what frequencies to set each clear frequency search slice to.

        :param      cfs_spectrum: Analyzed CFS sequence data
        :type       cfs_spectrum: dictionary
        """
        freq = cfs_spectrum.cfs_freq
        mags = [mag.cfs_data for mag in cfs_spectrum.output_datasets]
        sorted_freqs = [
            [x for _, x in sorted(zip(mags[slc], freq))] for slc in range(len(mags))
        ]
        # extract and sort frequencies based on magnitudes. Lowest noise to highest noise.

        for sqn_num, sequence in enumerate(self.cfs_sequences):
            freq_index = 0  # starting (most ideal) cfs spectrum frequency index
            used_freqs = []
            for slice_obj in sequence.slice_dict.values():
                df = int(
                    round(1e3 / (2 * slice_obj.pulse_len))
                )  # bandwidth of tx pulse / 2 in kHz
                if not slice_obj.cfs_flag:
                    used_freqs.append(slice_obj.freq)
                    # record all non-cfs frequencies already in use
            for slice_obj in sequence.slice_dict.values():
                if slice_obj.cfs_flag:
                    slice_range = slice_obj.cfs_range
                    center_freq = int((slice_range[0] + slice_range[1]) / 2)
                    shifted_freq = np.array(sorted_freqs[sqn_num]) / 1000 + center_freq

                    mask = np.full(len(shifted_freq), True)
                    for f_rng in options.restricted_ranges:
                        mask[
                            np.argwhere(
                                np.logical_and(
                                    shifted_freq >= f_rng[0] - 0.51,
                                    shifted_freq <= f_rng[1] + 0.5,
                                )
                            )
                        ] = False
                        # Rounding when setting the freq below could cause the freq to set to a restricted value
                        # increase the restricted ranges by 0.5 to prevent rounding error
                    f_rng = slice_obj.cfs_range
                    mask[
                        np.argwhere(
                            np.logical_or(
                                shifted_freq < f_rng[0], shifted_freq > f_rng[1]
                            )
                        )
                    ] = False
                    shifted_freq = shifted_freq[mask]
                    # Mask all restricted frequencies and frequencies outside cfs_range in the
                    # cfs spectrum frequencies

                    repeat = True
                    while repeat:
                        # Search for lowest noise usable frequency
                        if freq_index >= len(shifted_freq):
                            log.critical(
                                f"All given frequencies were to close to used frequencies or the tx or rx"
                                f"center frequencies. The radar will crash!!!",
                                used_freqs=used_freqs,
                                cfs_sorted_freqs=shifted_freq,
                            )

                        if any(
                            abs(shifted_freq[freq_index] - x) <= df for x in used_freqs
                        ):
                            freq_index += 1
                            # skip current shifted freq index if it is too close to a used frequency
                        elif abs(shifted_freq[freq_index] - self.txctrfreq) <= 50:
                            freq_index += 1
                            # Skip frequency if too close to tx center frequency
                        elif abs(shifted_freq[freq_index] - self.rxctrfreq) <= 50:
                            freq_index += 1
                            # Skip frequency if too close to rx center frequency
                        else:
                            repeat = False
                            slice_obj.freq = int(np.round(shifted_freq[freq_index]))
                            used_freqs.append(int(np.round(shifted_freq[freq_index])))
                            # Set cfs slice frequency and add frequency to used_freqs for this sequence
                            freq_index += 1
                            log.verbose(
                                "setting cfs slice freq",
                                slice_id=slice_obj.slice_id,
                                set_freq=slice_obj.freq,
                            )

            sequence.build_sequence_pulses()
            # Build sequence pulses once all cfs slices have been assigned a frequency

    def build_cfs_sequence(self):
        """
        Builds an empty rx only pulse sequence to collect clear frequency search data
        """
        import copy
        import borealis_experiments.superdarn_common_fields as scf
        from experiment_prototype.experiment_slice import ExperimentSlice

        pulse_length = 100  # us
        num_ranges = int(round((self.slice_dict[0].cfs_duration * 1000) / pulse_length))
        # calculate number of ranges (cfs_duration is in ms)

        # Create a CFS slice for the pulse
        default_slice = {
            "cpid": self.slice_dict[0].cpid,
            "slice_id": 0,
            "transition_bandwidth": self.slice_dict[0].transition_bandwidth,
            "output_rx_rate": self.slice_dict[0].output_rx_rate,
            "rx_bandwidth": self.slice_dict[0].rx_bandwidth,
            "tx_bandwidth": self.slice_dict[0].tx_bandwidth,
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
            "decimation_scheme": self.slice_dict[0].cfs_scheme,
        }

        CFS_slices = {}
        seq_keys = []
        slice_counter = 0
        for cfs_id in self.cfs_slice_ids:
            listening_slice = copy.deepcopy(default_slice)
            slice_range = self.slice_dict[cfs_id].cfs_range
            listening_slice["freq"] = int((slice_range[0] + slice_range[1]) / 2)
            listening_slice["slice_id"] = slice_counter

            CFS_slices[slice_counter] = ExperimentSlice(**listening_slice)
            seq_keys.append(slice_counter)
            slice_counter += 1

        # Build interface dictionary
        interface_dict = {}
        for ref_ind in range(slice_counter):
            for int_ind in range(ref_ind + 1, slice_counter):
                interface_dict[(ref_ind, int_ind)] = "CONCURRENT"

        self.cfs_sequence = Sequence(
            seq_keys, CFS_slices, interface_dict, self.transmit_metadata
        )
