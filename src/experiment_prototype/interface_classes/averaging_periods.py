#!/usr/bin/python

"""
    averaging_periods
    ~~~~~~~~~~~~~~~~~
    This is the module containing the AveragingPeriod class. The AveragingPeriod class contains the
    InterfaceClassBase members, as well as clrfrqflag (to be implemented), intn (number of integrations
    to run), or intt(max time for integrations), and it contains sequences of class Sequence.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
# built-in
import inspect
from pathlib import Path

# third party
import structlog

# local
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
    clrfrqflag
        Boolean, True if clrfrqsearch should be performed.
    clrfrqrange
        The range of frequency to search if clrfrqflag is True.  Otherwise empty.
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
        self.clrfrqflag = False
        # there may be multiple slices in this averaging period at different frequencies so
        # we may have to search multiple ranges.
        self.clrfrqrange = []
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id].clrfrqflag:
                self.clrfrqflag = True
                self.clrfrqrange.append(self.slice_dict[slice_id].clrfrqrange)

        # TODO: SET UP CLEAR FREQUENCY SEARCH CAPABILITY
        # also note for when setting this up clrfrqranges may overlap

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

        # Determine how this averaging period is made by separating out the SEQUENCE interfaced.
        self.nested_slice_list = self.get_nested_slice_ids()
        self.sequences = []

        for params in self.prep_for_nested_interface_class():
            self.sequences.append(Sequence(*params))

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
