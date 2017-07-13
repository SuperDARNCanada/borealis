#!/usr/bin/env python

# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# REVIEW #37 Best way to call interpreter? there's also "/usr/bin/env python" that allows you to
# have python installed anywhere REPLY sure
# REVIEW #7 We need some kind of license at top of all files - or a referral to the
# license/copyright/etc REPLY agreed we should discuss and standardize this for all our files

from __future__ import print_function

"""
The template for an experiment. 
"""

import sys
import copy
import traceback  # probably useful for logging when I get there

from experiment_exception import ExperimentException

import list_tests

# TODO: Set up python path in scons PYTHONPATH = ....../placeholderOS
from utils.experiment_options.experimentoptions import ExperimentOptions
from radar_control.scan_classes import scans

interface_types = frozenset(['SCAN', 'INTTIME', 'INTEGRATION', 'PULSE'])

"""
INTERFACING TYPES:

NONE : Only the default, must be changed.
SCAN : Scan by scan interfacing. exp_slice #1 will scan first
    followed by exp_slice #2 and subsequent exp_slice's.
INTTIME : nave by nave interfacing (full integration time of
     one pulse_sequence, then the next). Time/number of pulse_sequences
    dependent on intt and intn in exp_slice. Effectively
    simultaneous scan interfacing, interleaving each
    integration time in the scans. exp_slice #1 first inttime or
    beam direction will run followed by exp_slice #2's first inttime,
    etc. if exp_slice #1's len(scan) is greater than exp_slice #2's, exp_slice
    #2's last integration will run and then all the rest of exp_slice
    #1's will continue until the full scan is over. exp_slice 1
    and 2 must have the same scan boundary, if any boundary.
    All other may differ.
INTEGRATION : integration by integration interfacing (one
    #pulse_sequence of one exp_slice, then the next). exp_slice #1 and
    exp_slice #2 must have same intt and intn. Integrations will
    switch between one and the other until time is up/nave is
    reached.
PULSE : Simultaneous pulse_sequence interfacing, pulse by pulse
    creates a single pulse_sequence. exp_slice A and B might have different
    frequencies (stereo) and/or may have different pulse
    length, mpinc, pulse_sequence, but must have the same integration
    time. They must also have same len(scan), although they may
    use different directions in scan. They must have the same
    scan boundary if any. A time offset between the pulses
    starting may be set (seq_timer in exp_slice). exp_slice A
    and B will have integrations that run at the same time.
"""


debug_flag = True # TODO move this somewhere central. config?

class ExperimentPrototype(object):
    """A prototype experiment class composed of metadata, including experiment slices (exp_slice) 
    which are dictionaries of radar parameters. Basic, traditional experiments will be composed 
    of a single slice. More complicated experiments will be composed of multiple slices that 
    interface in one of four pre-determined ways, as described in more detail below. 
    
    This class is used via inheritance to create experiments.
    
    Some variables shouldn't be changed after init, and their properties do not have setters. 
    Some variables in this class are given property setters and can be modified with a class 
    method entitled 'update' in your experiment class. 

    :param cpid: unique id necessary for each control program (experiment)
    :
    """

    __slice_keys = ["slice_id", "cpid", "txantennas", "rx_main_antennas",
                    "rx_int_antennas", "pulse_sequence", "pulse_shift", "mpinc",
                    "pulse_len", "nrang", "frang", "intt", "intn", "beam_angle",
                    "beam_order", "scanboundflag", "scanbound", "txfreq", "rxfreq",
                    "clrfrqrange", "acf", "xcf", "acfint", "wavetype", "seqtimer"]

    __hidden_slice_keys = ['rxonly', 'clrfrqflag']

    """ The slice keys are described as follows: 
    
    DESCRIPTION OF SLICE KEYS

slice_id : The ID of this object. An experiment can have multiple objects.
cpid: The ID of the experiment, consistent with existing radar control programs.
txantennas: The antennas to transmit on, default is all antennas given max number from config.
rx_main_antennas: The antennas to receive on in main array, default = all antennas given max number from config.
rx_int_antennas : The antennas to receive on in interferometer array, default is all antennas given max number from config.
pulse_sequence: The pulse sequence timing, given in quantities of mpinc, for example normalscan = [0, 14, 22, 24, 27, 31, 42, 43]
mpinc: multi-pulse increment in us, Defines minimum space between pulses.
pulse_shift: Allows phase shifting between pulses. Built in for a capability Ashton would like to use within a pulse sequence. Default all zeros for all pulses in pulse_sequence.
pulse_len: length of pulse in us. Range gate size is also determined by this.
nrang: Number of range gates.
frang: first range gate, in km
intt: duration of an integration, in ms. (maximum)
intn: number of averages to make a single integration, if intt = None.
beam_angle: list of beam directions, in degrees off azimuth. Positive is E of N. Array length = number of beams.
beam_order: beamnumbers written in order of preference, one element in this list corresponds to one integration period. Can have list within lists. a beamnubmer of 0 in this list gives us beam_angle[0] as a direction.
scanboundflag: flag for whether there is a scan boundary to wait for in order to start a new scan.
scanbound: time that is alloted for a scan before a new scan boundary occurs (ms).
clrfrqrange: range for clear frequency search, should be a list of length = 2, [min_freq, max_freq] in kHz.
txfreq: transmit frequency, in kHz. Note if you specify clrfrqrange it won't be used.
rxfreq: receive frequency, in kHz. Note if you specify clrfrqrange or txfreq it won't be used. Only necessary to specify if you want a receive-only slice.
acf: flag for rawacf and generation. Default True.
xcf: flag for cross-correlation data. Default True
acfint: flag for interferometer autocorrelation data. Default True.
wavetype: default SINE. Any others not currently supported but possible to add in at later date.
seqtimer: timing in us that this object's sequence will begin at, after the start of the sequence.

Should add:

scanboundt : time past the hour to start a scan at ?
    
        Traditionally beams have been 3.75 degrees separated but we don't refer to them as beam 
        -22.75 degrees, we refer as beam 1, beam 2. This is like a mapping of beam number
        to beam direction off azimuth . Then you can use the beam numbers in the beam_order list so you 
        can reuse beams within one scan, or use multiple beamnumbers in a single integration time, which 
        would trigger an imaging integration. When we do imaging we will still have to quantize the directions
        we are looking in to certain beam directions, and if you wanted to add directions you could 
        do so in experiment modification functions.

        exp_slice['beam_angle'] is going to be a list of possible beam directions for this
        experiment slice in degrees off azimuth. It doesn't mean that these are the beam 
        directions that will be used in the scan, those beams are in the beam_order list given by beamnumber. 

    """  # TODO describe these

    def __init__(self, cpid):

        try:
            assert isinstance(cpid, int)
        except AssertionError:
            errmsg = 'CPID must be a unique int'
            raise ExperimentException(errmsg)

        self.__cpid = cpid

        self._num_slices = 0
        # slice_list = []

        self.__slice_list = []
        # self._slice_list = slice_list

        self.__new_slice_id = 0
        self.__slice_ids = []

        # Load the config data
        self.__options = ExperimentOptions()

        # Next some metadata that you can change, with defaults.

        self.__txctrfreq = 12000  # in kHz.
        self._txrate = 12000000  # sampling rate, samples per sec.
        # Based on above settings, you can transmit from 6 - 18 MHz.

        self.__rxctrfreq = 12000  # in kHz.
        # NOTE: rx sampling rate is set in config.

        self._xcf = True
        # Get cross-correlation data in processing block.

        self._acfint = True
        # Determine lag-zero interferometer power in fitacf.

        self._interface = {}  # setup_interfacing(self.num_slices)
        # Dictionary of how each exp_slice interacts with the other slices. Default is "NONE" for
        #  all, but must be modified in experiment. NOTE keys are as such: (0,1), (0,2), (1,2),
        # NEVER includes (2,0) etc. The only interface options are: interface_types = frozenset([
        # 'SCAN', 'INTTIME', 'INTEGRATION', 'PULSE'])

        # The following are for internal use only, and should not be modified in the experimental
        #  class, but will be modified by the class methods to build_scans. For this reason they
        # are private with setters.

        self.__slice_id_scan_lists = None
        self.__scan_objects = []
        # These are used internally by the radar_control block to build iterable objects out of
        # the slice using the interfacing specified.

    @property
    def cpid(self):
        """
        The CPID is read-only once established in instantiation.
        :return: This experiment's CPID (control program ID, a term that comes from ROS).
        """
        return self.__cpid

    @property
    def num_slices(self):
        """
        The number of slices in the experiment. May change with updates.
        :return: 
        """
        return self._num_slices

    @num_slices.setter
    def num_slices(self, value):
        """
        The number of slices in the experiment. May change with updates.
        :param: value: to set the number of slices to. 
        :return:
        """
        if isinstance(value, int):
            self._num_slices = value
        else:
            pass  # TODO error

    @property
    def slice_keys(self):
        """
        Get the list of slice keys available. This cannot be updated.
        :return: the keys in the current ExperimentPrototype slice_keys dictionary (parameters 
         available for slices)
        """
        return self.__slice_keys

    @property
    def slice_list(self):
        """
        Get the list of slices. The slice list can be updated in add_slice, edit_slice, and 
        del_slice.
        :return: the list of slice dictionaries in this experiment.
        """
        return self.__slice_list

    @property
    def new_slice_id(self):
        """
        Get the next unique slice id available to this instance of the experiment. Increment 
        because this is accessed to ensure it is unique each time.
        :return: the next unique slice id.
        """
        self.__new_slice_id += 1
        return self.__new_slice_id - 1

    @property
    def slice_ids(self):
        """
        Get the slice ids that are currently available in this experiment. This is updated
        regularly through add_slice, edit_slice, and del_slice.
        :return: the list of slice ids in the experiment.
        """
        return self.__slice_ids

    @property
    def options(self):
        """
        Get the config options for running this experiment. These cannot be set or removed.
        :return: the config options.
        """
        return self.__options

    @property
    def xcf(self):
        """
        Get the cross-correlation flag status.
        :return: cross-correlation flag boolean.
        """
        return self._xcf

    @xcf.setter
    def xcf(self, value):
        """
        To set the cross-correlation flag.
        :param value: boolean for cross-correlation processing flag.
        :return: 
        """
        if isinstance(value, bool):
            self._xcf = value
        else:
            pass  # TODO errors

    @property
    def acfint(self):
        """
        To get the interferometer autocorrelation flag
        :return: interferometer autocorrelation flag boolean.
        """
        return self._acfint

    @acfint.setter
    def acfint(self, value):
        """
        To set the interferometer autocorrelation flag
        :param value: boolean for interferometer autocorrelation processing flag.
        :return: 
        """
        if isinstance(value, bool):
            self._acfint = value
        else:
            pass  # TODO errors

    @property
    def txrate(self):
        """
        To get the transmission sample rate to the DAC.
        :return: the transmission sample rate to the DAC (Hz).
        """
        return self._txrate

    @txrate.setter
    def txrate(self, value):
        """
        To set the transmission sample rate to the DAC (Hz)
        :param value: int for transmission sample rate to the DAC (Hz)
        :return: 
        """
        # TODO review if this should be modifiable in-experiment. Probably takes resetting of USRPs.
        if isinstance(value, int):
            self._txrate = value
        else:
            pass  # TODO errors

    @property
    def txctrfreq(self):
        """
        To get the transmission centre frequency that USRP is tuned to.
        :return: the transmission centre frequency that USRP is tuned to (Hz).
        """
        return self.__txctrfreq

    @txctrfreq.setter
    def txctrfreq(self, value):
        """
        To set the transmission centre frequency that USRP is tuned to.
        :param value: int for transmission centre frequency to tune USRP to (Hz).
        :return: 
        """
        # TODO review if this should be modifiable, definitely takes tuning time.
        if isinstance(value, int):
            self.__txctrfreq = value
        else:
            pass  # TODO errors

    @property
    def rxctrfreq(self):
        """
        To get the receive centre frequency that USRP is tuned to.
        :return: the receive centre frequency that USRP is tuned to (Hz).
        """
        return self.__rxctrfreq

    @rxctrfreq.setter
    def rxctrfreq(self, value):
        """
        To set the receive centre frequency that USRP is tuned to.
        :param value: int for receive centre frequency to tune USRP to (Hz).
        :return: 
        """
        # TODO review if this should be modifiable, definitely takes tuning time.
        if isinstance(value, int):
            self.__rxctrfreq = value
        else:
            pass  # TODO errors

    @property
    def interface(self):
        """
        To get the list of interfacing for the experiment slices.
        :return:the list of interfacing defined as [(slice_id1, slice_id2) : INTERFACING_TYPE] for
                all current slice_ids. 
        """
        return self._interface

    def add_slice(self, exp_slice, interfacing_dict=None):
        """
        Add slices to the experiment.
        :param exp_slice: 
        :param interfacing_dict: dictionary of type {slice_id : INTERFACING , ... } that defines how
         this slice interacts with all the other slices.
        :return: the slice_id of the new slice that was just added.
        """
        if not isinstance(exp_slice, dict):
            # TODO error log
            return
        exp_slice['slice_id'] = self.new_slice_id
        # each added slice has a unique id, even if previous slices have been deleted.
        exp_slice['cpid'] = self.cpid
        new_exp_slice = self.setup_slice(exp_slice)
        # check for any errors after defaults have been filled.
        print('Requested Add {}'.format(exp_slice))
        print('Adding (with Defaults) {}'.format(new_exp_slice))
        self.__slice_list.append(new_exp_slice)
        self.__slice_ids.append(exp_slice['slice_id'])
        self.check_slice_ids()  # This is critical
        self.num_slices = len(self.__slice_list)
        for ind in self.slice_ids:
            if ind == new_exp_slice['slice_id']:
                continue
            try:
                self._interface[(ind, new_exp_slice['slice_id'])] = interfacing_dict[ind]
                # update to add interfacing. new slice_id will be greater than all others so
                # we can add with ind first and maintain interfacing list rule of key1 < key2.
            except TypeError or IndexError:
                self._interface[(ind, new_exp_slice['slice_id'])] = None
                print('Interfacing not Fully Updated - Will Cause Errors so Please Update.')
                # TODO return a warning if interfacing dictionary not updated at this time.

        return new_exp_slice['slice_id']

    def del_slice(self, remove_slice_id):
        self.check_slice_ids()
        if isinstance(remove_slice_id, int) and remove_slice_id in self.slice_ids:
            for slc in self.slice_list:
                if slc['slice_id'] == remove_slice_id:
                    remove_me = self.__slice_list.pop(slc)
            self.num_slices = len(self.slice_list)
            self.__slice_ids.remove(remove_slice_id)
            for key1, key2 in self._interface.keys():
                if key1 == remove_slice_id or key2 == remove_slice_id:
                    del self._interface[(key1, key2)]
            self.check_slice_ids()  # Do this again for testing
        else:
            print('Slice ID does not exist in Slice_IDs list.')
            # TODO log error

    def edit_slice(self, edit_slice_id, param, value):
        if isinstance(edit_slice_id, int) and edit_slice_id in self.slice_ids:
            if isinstance(param, str) and param in self.slice_keys:
                edited_slice = self.slice_list[edit_slice_id].copy()
                edited_slice[param] = value
                new_interface_values = {}
                for ifkey, ifvalue in self._interface:
                    if edit_slice_id == ifkey[0]:
                        new_interface_values[ifkey[1]] = ifvalue
                    elif edit_slice_id == ifkey[1]:
                        new_interface_values[ifkey[0]] = ifvalue
                new_slice_id = self.add_slice(edited_slice, new_interface_values)
                # checks done when slice is added.
                self.del_slice(edit_slice_id)
                # slice ids are checked after slice is removed.
                return new_slice_id
            else:
                print('Parameter for Edit is Not a Valid Parameter')
                # TODO log error
        else:
            print('Slice ID does not exist in Slice_IDs list.')
            # TODO log error

    def __repr__(self):
        represent = 'self.cpid = {}\nself.num_slices = {}\nself.slice_ids = {}\nself.slice_keys = {}\nself.options = \
                    {}\nself.txctrfreq = {}\nself.txrate = {}\nself.rxctrfreq = {}\nself. xcf = {}\n' \
                    'self.acfint = {}\nself.slice_list = {}\nself.interface = {}\n'.format(
            self.cpid, self.num_slices, self.slice_ids, self.slice_keys, self.options.__repr__(),
            self.txctrfreq, self.txrate, self.rxctrfreq, self.xcf, self.acfint,
            self.slice_list, self.interface)
        return represent

    def build_scans(self):
        """ 
        Will be run by experiment handler, to build iterable objects for radar_control to use. 
        Creates scan_objects in the instance and the slice_id_scan_lists for identifying which 
        slices are in the scans.
        :return:
        """

        # Check interfacing
        self.self_check()
        self.check_interfacing()
        self.__slice_id_scan_lists = self.get_scans()
        # Returns list of scan lists. Each scan list is a list of the slice_ids for the slices
        # included in that scan.
        if self.__slice_id_scan_lists:  # if list is not empty, can make scans
            self.__scan_objects = [scans.Scan(self, scan_list) for scan_list in
                                    self.__slice_id_scan_lists]
        # Append a scan instance, passing in the list of slice ids to include in scan.

        if debug_flag:
            print("Number of Scan types: {}".format(len(self.__scan_objects)))
            print("Number of AveragingPeriods in Scan #1: {}".format(len(self.__scan_objects[
                                                                             0].aveperiods)))
            print("Number of Sequences in Scan #1, Averaging Period #1: {}".format(
                len(self.__scan_objects[0].aveperiods[0].integrations)))
            print("Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1:"
                  " {}".format(len(self.__scan_objects[0].aveperiods[0].integrations[0].cpos)))

        else:
            pass  # TODO error

    def get_scans(self):
        """
        Take my own interfacing and get info on how many scans and which slices make which scans.
        :rtype list
        :return list of lists. The list has one element per scan. Each element is a list of 
        slice_id's signifying which slices are combined inside that scan. The element could be a 
        list of length 1, meaning only one slice_id is included in that scan. The list returned 
        could be of length 1, meaning only one scan is present in the experiment.
        """
        scan_combos = []

        for k in self.interface.keys():
            if self.interface[k] != "SCAN":
                scan_combos.append([k])

        scan_combos = sorted(scan_combos)
        # if [2,4] and [1,4], then also must be [1,2] in the scan_combos
        # Now we are going to modify the list of lists of length = 2 to be a list of length x so that if [1,2] and [2,4]
        # and [1,4] are in scan_combos, we want only one list element for this scan : [1,2,4] .
        scan_i = 0  # REVIEW #3 This needs a detailed explaination with examples. REPLY OK TODO
        while scan_i < len(scan_combos):  # i: element in scan_combos (representing one scan)
            slice_id_k = 0
            while slice_id_k < len(
                    scan_combos[scan_i]):  # k: element in scan (representing a slice)
                scan_j = scan_i + 1  # j: iterates through the other elements of scan_combos, to combine them into
                # the first, i, if they are in fact part of the same scan.
                while scan_j < len(scan_combos):
                    if scan_combos[scan_i][slice_id_k] == scan_combos[scan_j][
                        0]:  # if an element (slice_id) inside
                        # the i scan is the same as a slice_id in the j scan (somewhere further in the scan_combos),
                        # then we need to combine that j scan into the i scan. We only need to check the first element
                        # of the j scan because scan_combos has been sorted and we know the first slice_id in the scan
                        # is less than the second slice id.
                        add_n_slice_id = scan_combos[scan_j][
                            1]  # the slice_id to add to the i scan from the j scan.
                        scan_combos[scan_i].append(add_n_slice_id)
                        # Combine the indices if there are 3+ slices combining in same scan
                        for m in range(0, len(scan_combos[
                                                  scan_i]) - 1):  # if we have added z to scan_i, such that
                            # scan_i is now [x,y,z], we now have to remove from the scan_combos list [x,z], and [y,z].
                            # If x,z existed as SCAN but y,z did not, we have an error.
                            # Try all values in scan_combos[i] except the last value, which is = to add_n.
                            """
                            if scan_combos[i][m] > add_n:
                                .......
                            """
                            try:
                                scan_combos.remove([scan_combos[scan_i][m], add_n_slice_id])
                                # scan_combos[j][1] is the known last value in scan_combos[i]
                            except ValueError:
                                # This error would occur if you had set [x,y] and [x,z] to PULSE but [y,z] to
                                # SCAN. This means that we couldn't remove the scan_combo y,z from the list because it
                                # was not added to scan_combos because it wasn't a scan type, so the interfacing would
                                # not make sense (conflict).
                                errmsg = 'Interfacing not Valid: exp_slice {} and exp_slice {} are combined in-scan and do not \
                                    interface the same with exp_slice {}'.format(
                                    scan_combos[scan_i][m],
                                    scan_combos[scan_i][slice_id_k],
                                    add_n_slice_id)
                                sys.exit(errmsg)
                        scan_j = scan_j - 1
                        # This means that the former scan_combos[j] has been deleted and there are new values at
                        #   index j, so decrement before incrementing in while.
                        # The above for loop will delete more than one element of scan_combos (min 2) but the
                        # while scan_j < len(scan_combos) will reevaluate the length of scan_combos.
                    scan_j = scan_j + 1
                slice_id_k = slice_id_k + 1  # if interfacing has been properly set up, the loop will only ever find
                # elements to add to scan_i when slice_id_k = 0. If there were errors though (ex. x,y and y,z = PULSE
                # but x,z did not) then iterating through the slice_id elements will allow us to find the
                # error.
            scan_i = scan_i + 1  # At this point, all elements in the just-finished scan_i will not be found anywhere
            #  else in scan_combos.

        # now scan_combos is a list of lists, where a slice_id occurs only once, within the nested list.

        for slice_id in range(self.num_slices):
            for sc in scan_combos:
                if slice_id in sc:
                    break
            else:  # no break
                scan_combos.append([slice_id])
                # Append the slice_id on its own, is not combined within scan.

        scan_combos = sorted(scan_combos)
        return scan_combos

    def check_slice_minimum_requirements(self, exp_slice):
        """
        Check for the minimum requirements of the slice. The following keys are always required:
        "pulse_sequence", "mpinc", "pulse_len", "nrang", "frang", "intt", "intn", "beam_angle",
        and "beam_order". This function may modify these keys. Ensure the values make sense.
        :param exp_slice: 
        :return: 
        """

        # TODO: add checks for values that make sense, not just check for types

        try:
            assert 'pulse_sequence' in exp_slice.keys()
            assert isinstance(exp_slice['pulse_sequence'], list)
            for element in exp_slice['pulse_sequence']:
                assert isinstance(element, int)
        except AssertionError:
            errmsg = "Slice must specify pulse_sequence that must be a list of integers"
            raise ExperimentException(errmsg, exp_slice)

        try:
            assert 'mpinc' in exp_slice.keys()
            assert isinstance(exp_slice['mpinc'], int)
        except AssertionError:
            errmsg = "Slice must specify mpinc that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        try:  # TODO may want to add a field for range_gate which could set this param.
            assert 'pulse_len' in exp_slice.keys()
            assert isinstance(exp_slice['pulse_len'], int)
        except AssertionError:
            errmsg = "Slice must specify pulse_len that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        try:
            assert 'nrang' in exp_slice.keys()
            assert isinstance(exp_slice['nrang'], int)
        except AssertionError:
            errmsg = "Slice must specify nrang that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        try:
            assert 'frang' in exp_slice.keys()
            assert isinstance(exp_slice['frang'], int)
        except AssertionError:
            errmsg = "Slice must specify frang that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        try:
            assert 'intt' in exp_slice.keys()
        except AssertionError:
            try:
                assert 'intn' in exp_slice.keys()
            except AssertionError:
                errmsg = "Slice must specify either an intn or intt"
                raise ExperimentException(errmsg, exp_slice)
            else:
                try:
                    assert isinstance(exp_slice['intn'], int)
                except AssertionError:
                    errmsg = "intn must be an integer"
                    raise ExperimentException(errmsg, exp_slice)
        else:
            try:
                assert isinstance(exp_slice['intt'], int)
            except AssertionError:
                errmsg = "intt must be an integer"
                raise ExperimentException(errmsg, exp_slice)
            else:
                try:
                    assert 'intn' in exp_slice.keys()
                except AssertionError:
                    pass
                else:
                    # TODO Log warning intn will not be used
                    exp_slice.pop('intn')

        try:
            assert 'beam_angle' in exp_slice.keys()
            assert isinstance(exp_slice['beam_angle'], list)
            for element in exp_slice['beam_angle']:
                assert isinstance(element, float) or isinstance(element, int) or isinstance(element,
                                                                                            list)
                if isinstance(element, list):
                    for angle in element:
                        assert isinstance(angle, float) or isinstance(angle, int)
        except AssertionError:
            errmsg = "Slice must specify beam_angle that must be a list of floats or lists (of floats)"
            raise ExperimentException(errmsg, exp_slice)

        try:
            assert 'beam_order' in exp_slice.keys()
            assert isinstance(exp_slice['beam_order'], list)
            for element in exp_slice['beam_order']:
                assert isinstance(element, int)
                assert element < len(exp_slice['beam_angle'])
        except AssertionError:
            errmsg = "Slice must specify beam_order that must be a list of ints corresponding to " \
                     "the order of the angles in the beam_angle list."
            raise ExperimentException(errmsg, exp_slice)

    @staticmethod
    def set_slice_identifiers(exp_slice):
        """
        This function sets up internal identifier flags 'clrfrqflag' and 'rxonly' in the slice so 
        that we know how to properly set up the slice and know which keys in the slice must be 
        specified and which are unnecessary. If these keys are ever written by the user, they will 
        be rewritten here.
        :param exp_slice: 
        :return: 
        """

        if 'clrfrqrange' in exp_slice.keys():

            exp_slice['clrfrqflag'] = True
            exp_slice['rxonly'] = False

            txfreq = exp_slice.pop('txfreq', None)
            if txfreq is not None and txfreq not in \
                    range(exp_slice['clrfrqrange'][0],
                          exp_slice['clrfrqrange'][1]):
                pass  # TODO log a warning. Txfreq is removed but we may not be doing as you intended.

            rxfreq = exp_slice.pop('rxfreq', None)
            if rxfreq is not None and rxfreq not in \
                    range(exp_slice['clrfrqrange'][0],
                          exp_slice['clrfrqrange'][1]):
                pass  # TODO log a warning. Rxfreq is removed but we may not be doing as you intended.

        elif 'txfreq' in exp_slice.keys():
            exp_slice['clrfrqflag'] = False
            exp_slice['rxonly'] = False
            rxfreq = exp_slice.pop('rxfreq', None)
            if rxfreq is not None and rxfreq != exp_slice['txfreq']:
                pass  # TODO log a warning. Rxfreq is removed but we may not be doing as you intended.
        else:
            exp_slice['rxonly'] = True
            exp_slice['clrfrqflag'] = False

    def check_slice_specific_requirements(self, exp_slice):
        """
        Check the requirements for the specific slice type as identified by the identifiers
        rxonly and clrfrqflag. The keys that need to be checked depending on these identifiers 
        are "txfreq", "rxfreq", and "clrfrqrange". This function may modify these keys.
        :param exp_slice: 
        :return: 
        """
        if exp_slice['clrfrqflag']:  # TX and RX mode with clear frequency search.
            # In this mode, clrfrqrange is required along with the other requirements.
            try:
                assert isinstance(exp_slice['clrfrqrange'], list)
                assert len(exp_slice['clrfrqrange']) == 2
                assert isinstance(exp_slice['clrfrqrange'][0], int)
                assert isinstance(exp_slice['clrfrqrange'][1], int)
            except AssertionError:
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)
            try:
                assert exp_slice['clrfrqrange'][0] < exp_slice['clrfrqrange'][1]
                assert (exp_slice['clrfrqrange'][1] * 1000) < self.options.max_freq
                assert (exp_slice['clrfrqrange'][0] * 1000) > self.options.min_freq
            except AssertionError:
                errmsg = """clrfrqrange must be between min and max frequencies for the radar
                            and must have lower frequency first.
                            """
                raise ExperimentException(errmsg)

            still_checking = True
            while still_checking:
                for freq_range in self.options.restricted_ranges:
                    try:
                        assert exp_slice['clrfrqrange'][0] not in range(freq_range[0],
                                                                        freq_range[1])
                    except AssertionError:
                        try:
                            assert exp_slice['clrfrqrange'][1] not in range(freq_range[0],
                                                                            freq_range[1])
                        except AssertionError:
                            # the range is entirely within the restricted range.
                            raise ExperimentException('clrfrqrange is entirely within restricted '
                                                      'range {}'.format(freq_range))
                        else:
                            # TODO Log warning, changing clrfrqrange because lower portion is in a
                            # restricted frequency range.
                            exp_slice['clrfrqrange'][0] = freq_range[
                                                              1] + 1  # outside of restricted range.
                            break  # we have changed the 'clrfrqrange' - must restart the
                            # check in case it's in another range.
                    else:
                        # lower end is not in restricted frequency range.
                        try:
                            assert exp_slice['clrfrqrange'][1] not in range(freq_range[0],
                                                                            freq_range[1])
                        except AssertionError:
                            # TODO Log warning, changing clrfrqrange because upper portion is in a
                            # restricted frequency range.
                            exp_slice['clrfrqrange'][1] = freq_range[
                                                              0] - 1  # outside of restricted range.
                            break  # we have changed the 'clrfrqrange' - must restart the for loop
                            # checking in case it's in another range.
                        else:  # neither end of clrfrqrange is inside the restricted range but
                            # we should check if the range is inside the clrfrqrange.
                            try:
                                assert freq_range[0] not in range(exp_slice['clrfrqrange'][0],
                                                                  exp_slice['clrfrqrange'][1])
                            except AssertionError:
                                # TODO Log a warning that there is a restricted range in the middle
                                # of the
                                # clrfrqrange that will be avoided OR could make this an Error.
                                # Still need to implement clear frequency searching.
                                pass
                else:  # no break, so no changes to the clrfrqrange
                    still_checking = False

        elif exp_slice['rxonly']:  # RX only mode.
            # In this mode, rxfreq is required.
            try:
                assert isinstance(exp_slice['rxfreq'], int) or isinstance(exp_slice['rxfreq'],
                                                                          float)
                assert (exp_slice['rxfreq'] * 1000) < self.options.max_freq
                assert (exp_slice['rxfreq'] * 1000) > self.options.min_freq
            except AssertionError:
                errmsg = """rxfreq must be a number (kHz) between min and max frequencies for the 
                            radar"""
                raise ExperimentException(errmsg)

        else:  # TX-specific mode , without a clear frequency search.
            # In this mode, txfreq is required along with the other requirements.
            try:
                assert isinstance(exp_slice['txfreq'], int) or isinstance(exp_slice['txfreq'],
                                                                          float)
                assert (exp_slice['txfreq'] * 1000) < self.options.max_freq
                assert (exp_slice['txfreq'] * 1000) > self.options.min_freq
            except AssertionError:
                errmsg = """txfreq must be a number (kHz) between min and max frequencies for the 
                                            radar"""
                raise ExperimentException(errmsg)
            for freq_range in self.options.restricted_ranges:
                try:
                    assert exp_slice['txfreq'] not in range(freq_range[0], freq_range[1])
                except AssertionError:
                    errmsg = """txfreq is within a restricted frequency range"""
                    raise ExperimentException(errmsg)
                    # TODO or could just log this and set to default frequency, or the closest
                    # frequency that is available. Do we want to remove default frequency?

    def set_slice_defaults(self, exp_slice):
        """
        Set up defaults in case of some parameters being left blank.
        :param exp_slice: 
        :return: slice_with_defaults: updated slice
        """

        slice_with_defaults = copy.deepcopy(exp_slice)

        if 'txantennas' not in exp_slice:
            slice_with_defaults['txantennas'] = [i for i in range(0,
                                                                  self.options.main_antenna_count)]
            # all possible antennas.
        if 'rx_main_antennas' not in exp_slice:
            slice_with_defaults['rx_main_antennas'] = [i for i in
                                                       range(0, self.options.main_antenna_count)]
        if 'rx_int_antennas' not in exp_slice:
            slice_with_defaults['rx_int_antennas'] = \
                [i for i in range(0, self.options.interferometer_antenna_count)]
        if 'pulse_shift' not in exp_slice:
            slice_with_defaults['pulse_shift'] = [0 for i in range(0, len(
                slice_with_defaults['pulse_sequence']))]
        if 'scanboundflag' and 'scanbound' not in exp_slice:
            slice_with_defaults['scanboundflag'] = False
            slice_with_defaults['scanbound'] = None
        elif 'scanboundflag' not in exp_slice:  # but scanbound is
            slice_with_defaults['scanboundflag'] = True

        if 'scanboundflag' in exp_slice:
            try:
                assert 'scanbound' in exp_slice
            except AssertionError:
                errmsg = 'ScanboundFlag is set without a Scanbound specified.'
                raise ExperimentException(errmsg)

        # we only have one because of slice checks already completed.
        if 'intt' in exp_slice:
            slice_with_defaults['intn'] = None
        elif 'intn' in exp_slice:
            slice_with_defaults['intt'] = None

        if 'acf' not in exp_slice:
            slice_with_defaults['acf'] = True
        if 'xcf' not in exp_slice:
            slice_with_defaults['xcf'] = True
        if 'acfint' not in exp_slice:
            slice_with_defaults['acfint'] = True
        if 'wavetype' not in exp_slice:
            slice_with_defaults['wavetype'] = 'SINE'
        if 'seqtimer' not in exp_slice:
            slice_with_defaults['seqtimer'] = 0

        return slice_with_defaults

    def setup_slice(self, exp_slice):
        """
        Before adding the slice, ensure that the internal parameters are set, remove unnecessary
        keys and check values of keys that are needed, and set defaults of keys that are optional.
        
        The following are always able to be defaulted, so are optional:
        "txantennas", "rx_main_antennas", "rx_int_antennas", "pulse_shift", "scanboundflag", 
        "scanbound", "acf", "xcf", "acfint", "wavetype", "seqtimer"


        The following are always required for processing acf, xcf, and acfint which we will assume
        we are always doing:
        "pulse_sequence", "mpinc", "pulse_len", "nrang", "frang", "intt", "intn", "beam_angle",
        "beam_order"

        The following are required depending on slice type:
        "txfreq", "rxfreq", "clrfrqrange" 
        
        :param: exp_slice: a slice to setup
        :return: complete_slice : a checked slice with all defaults
        """

        complete_slice = copy.deepcopy(exp_slice)

        # None values are useless to us - if they do not exist we know they are None.
        for key, value in complete_slice.items():
            if value is None:
                complete_slice.pop(key)

        try:
            assert 'rxfreq' or 'txfreq' or 'clrfrqrange' in complete_slice.keys()
        except AssertionError:
            errmsg = 'An rxfreq, txfreq, or clrfrqrange must be Specified in a Slice'
            # TODO log
            raise ExperimentException(errmsg, exp_slice)

        self.set_slice_identifiers(complete_slice)
        self.check_slice_specific_requirements(complete_slice)
        self.check_slice_minimum_requirements(complete_slice)
        complete_slice = self.set_slice_defaults(complete_slice)
        # set_slice_defaults will check for any missing values that should be given a default and
        # fill them.

        errors = self.check_slice(complete_slice)

        if errors:
            raise ExperimentException(errors)

        return complete_slice

    def self_check(self):
        """
        Check that the values in this experiment are valid
        """

        if self.num_slices < 1:
            errmsg = "Error: Invalid num_slices less than 1"
            raise ExperimentException(errmsg)

        if self._num_slices != len(self.slice_list):
            errmsg = """Error: slice_list length {} doesn't equal num_slices {}. You may have not updated your slice 
                        list or have added too many slices.""".format(len(self.slice_list),
                                                                      self._num_slices)
            raise ExperimentException(errmsg)

        # TODO: somehow check if self.cpid is not unique - incorporate known cpids from git repo?

        for a_slice in range(self.num_slices):
            selferrs = self.check_slice(self.slice_list[a_slice])
            if not selferrs:
                # If returned error dictionary is empty
                continue
            errmsg = "Self Check Errors Occurred with slice Number : {} \nSelf \
                Check Errors are : {}".format(a_slice, selferrs)
            raise ExperimentException(errmsg)

        print("No Self Check Errors. Continuing...")

        return None

    def check_slice(self, exp_slice):
        """
        This is the first test of the dictionary in the experiment done to ensure values in this 
        slice make sense. This is a self-check to ensure the parameters (for example, txfreq, 
        antennas) are appropriate. All fields should be full at this time (whether filled by the 
        user or given default values in set_slice_defaults).
        :param: exp_slice: a slice to check
        """
        error_count = 0
        error_dict = {}

        options = self.options

        for param in self.slice_keys:
            try:
                 assert param in exp_slice.keys()
            except AssertionError:
                if param == 'txfreq' and exp_slice['clrfrqflag']:
                    pass
                elif param == 'rxfreq' and not exp_slice['rxonly']:
                    pass
                elif param == 'clrfrqrange' and not exp_slice['clrfrqrange']:
                    pass
                else:
                    errmsg = "Slice {} is missing Necessary Parameter {}".format(
                        exp_slice['slice_id'], param)
                    raise ExperimentException(errmsg)
                    # set defaults if possible
            try:
                assert param is not None
            except AssertionError:
                pass  # TODO may want to check certain params are not None

        if error_dict:  # if not empty
            return error_dict  # cannot check all params because they don't exist

        for param in exp_slice.keys():
            if param not in self.slice_keys and param not in self.__hidden_slice_keys:
                error_dict[error_count] = "Slice {} has A Parameter that is not Used: {} = {}". \
                    format(exp_slice['slice_id'], param, exp_slice[param])
                error_count += 1

        if len(exp_slice['txantennas']) > options.main_antenna_count:
            error_dict[
                error_count] = "Slice {} Has Too Many Main TX Antenna Channels {} Greater than Config {}" \
                .format(exp_slice['slice_id'], len(exp_slice['txantennas']),
                        options.main_antenna_count)
        if len(exp_slice['rx_main_antennas']) > options.main_antenna_count:
            error_dict[
                error_count] = "Slice {} Has Too Many Main RX Antenna Channels {} Greater than Config {}" \
                .format(exp_slice['slice_id'], len(exp_slice['rx_main_antennas']),
                        options.main_antenna_count)
            error_count = error_count + 1
        if len(exp_slice['rx_int_antennas']) > options.interferometer_antenna_count:
            error_dict[
                error_count] = "Slice {} Has Too Many RX Interferometer Antenna Channels {} " \
                               "Greater than Config {}".format(
                                    exp_slice['slice_id'],
                                    len(exp_slice['rx_int_antennas']),
                                    options.interferometer_antenna_count)
            error_count = error_count + 1

        # Check if the antenna identifier number is greater than the config file's
        # maximum antennas for all three of tx antennas, rx antennas and rx int antennas
        # Also check for duplicates
        for i in range(len(exp_slice['txantennas'])):
            if exp_slice['txantennas'][i] >= options.main_antenna_count:
                error_dict[
                    error_count] = "Slice {} Specifies Main Array Antenna Numbers Over Config " \
                                   "Max {}" .format(exp_slice['slice_id'],
                                                    options.main_antenna_count)
                error_count = error_count + 1

        if list_tests.isduplicates(exp_slice['txantennas']):
            error_dict[error_count] = "Slice {} TX Main Antennas Has Duplicate Antennas".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        for i in range(len(exp_slice['rx_main_antennas'])):
            if exp_slice['rx_main_antennas'][i] >= options.main_antenna_count:
                error_dict[
                    error_count] = "Slice {} Specifies Main Array Antenna Numbers Over Config " \
                                   "Max {}" .format(exp_slice['slice_id'],
                                                    options.main_antenna_count)
                error_count = error_count + 1

        if list_tests.isduplicates(exp_slice['rx_main_antennas']):
            error_dict[error_count] = "Slice {} RX Main Antennas Has Duplicate Antennas".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        for i in range(len(exp_slice['rx_int_antennas'])):
            if exp_slice['rx_int_antennas'][i] >= options.interferometer_antenna_count:
                error_dict[
                    error_count] = "Slice {} Specifies Interferometer Array Antenna Numbers Over " \
                                   "Config Max {}".format(exp_slice['slice_id'],
                                                          options.interferometer_antenna_count)
                error_count = error_count + 1

        if list_tests.isduplicates(exp_slice['rx_int_antennas']):
            error_dict[
                error_count] = "Slice {} RX Interferometer Antennas Has Duplicate Antennas".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        # Check if the pulse_sequence is not increasing, which would be an error
        if not list_tests.isincreasing(exp_slice['pulse_sequence']):
            error_dict[error_count] = "Slice {} pulse_sequence Not Increasing".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        # Check that pulse_len and mpinc make sense (values in us)
        if exp_slice['pulse_len'] > exp_slice['mpinc']:
            error_dict['error_count'] = "Slice {} Pulse Length Greater than MPINC".format(
                exp_slice['slice_id'])
            error_count = error_count + 1
        if exp_slice['pulse_len'] < self.options.minimum_pulse_length:
            error_dict[error_count] = "Slice {} Pulse Length Too Small".format(
                exp_slice['slice_id'])
            error_count = error_count + 1
        if exp_slice['mpinc'] < self.options.minimum_mpinc_length:
            error_dict[error_count] = "Slice {} Multi-Pulse Increment Too Small".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        # check intn and intt make sense given mpinc, and pulse_sequence.
        if exp_slice['pulse_sequence']:  # if not empty
            # Sequence length is length of pulse sequence plus the scope sync delay time.
            seq_len = exp_slice['mpinc'] * (exp_slice['pulse_sequence'][-1]) \
                      + (exp_slice['nrang'] + 19 + 10) * exp_slice['pulse_len']  # us

            if exp_slice['intt'] is None and exp_slice['intn'] is None:
                # both are None and we are not rx - only
                error_dict['error_count'] = "Slice {} Has Transmission but no Intt or IntN".format(
                    exp_slice['slice_id'])
                error_count = error_count + 1

            if exp_slice['intt'] is not None and exp_slice['intn'] is not None:
                error_dict['error_count'] = "Slice {} Choose Either Intn or Intt to be the Limit " \
                                            "for Number of Integrations in an Integration Period.".\
                    format(exp_slice['slice_id'])
                error_count = error_count + 1

            if exp_slice['intt'] is not None:
                if seq_len > (exp_slice['intt'] * 1000):  # in us
                    error_dict[error_count] = "Slice {} : Pulse Sequence is Too Long for Integration " \
                                         "Time Given".format(exp_slice['slice_id'])
                    error_count = error_count + 1

        if not exp_slice['pulse_sequence']:
            if exp_slice['txfreq']:
                error_dict[error_count] = "Slice {} Has Transmission Frequency but no" \
                                            "Pulse Sequence defined".format(
                    exp_slice['slice_id'])
                error_count = error_count + 1

        if list_tests.isduplicates(exp_slice['beam_angle']):
            error_dict[error_count] = "Slice {} Beam Angles Has Duplicate Antennas".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        if not list_tests.isincreasing(exp_slice['beam_angle']):
            error_dict[error_count] = "Slice {} beam_angle Not Increasing Clockwise (E of N " \
                                      "is positive)".format(exp_slice['slice_id'])
            error_count = error_count + 1

        # Check if the list of beams to transmit on is empty
        if not exp_slice['beam_order']:
            error_dict[error_count] = "Slice {} Beam Order Scan Empty".format(
                exp_slice['slice_id'])
            error_count = error_count + 1

        # Check that the beam numbers in the beam_order exist
        for bmnum in exp_slice['beam_order']:
            if bmnum >= len(exp_slice['beam_angle']):
                error_dict[error_count] = "Slice {} Scan Beam Number {} DNE".format(
                    exp_slice['slice_id'], bmnum)
                error_count = error_count + 1

        # check scan boundary not less than minimum required scan time.
        if exp_slice['scanboundflag']:
            if exp_slice['scanbound'] < (
                        len(exp_slice['beam_order']) * exp_slice['intt']):
                error_dict[error_count] = "Slice {} Beam Order Too Long for ScanBoundary".format(
                    exp_slice['slice_id'])
                error_count = error_count + 1

        # TODO other checks

        if exp_slice['wavetype'] != 'SINE':
            error_dict[error_count] = "Slice {} wavetype of {} currently not supported".format(
                exp_slice['slice_id'], exp_slice['wavetype'])
            error_count = error_count + 1

        return error_dict

    def check_interfacing(self):
        """
        Check that the keys in the interface are not NONE and are
        valid.
        """

        for key, interface_type in self.interface.items():
            if interface_type == "NONE":
                errmsg = 'Interfacing is still default, must set key {}'.format(key)
                sys.exit(
                    errmsg)  # TODO for error handling. Perhaps use exceptions instead. REPLY OK

        for num1, num2 in self.interface.keys():
            if num1 >= self.num_slices or num2 >= self.num_slices or num1 < 0 or num2 < 0:
                # This is required for how I have it set up. Avoids any confusion
                #  with keys [0,2] vs [2,0] for example. Because you could add your own keys I check it.
                errmsg = """Interfacing key ({}, {}) is not valid, all keys must refer to (slice_id1,
                slice_id2) where slice_id1 < slice_id2""".format(num1, num2)
                sys.exit(errmsg)  # TODO for error handling. Perhaps use exceptions instead.
            if self.interface[num1, num2] not in interface_types:
                errmsg = 'Interfacing Not Valid Type between Slice_id {} and Slice_id {}'.format(
                    num1, num2)
                sys.exit(errmsg)  # TODO for error handling. Perhaps use exceptions instead.

    def check_slice_ids(self):
        checker_list = []
        for slc in self.slice_list:
            checker_list.append(slc['slice_id'])
        if self.slice_ids != checker_list:
            print('ERROR Slice List and Slice IDs Are Inconsistent - Fatal Error')
            # TODO error handling.
