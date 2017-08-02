#!/usr/bin/env python

# Copyright 2017 SuperDARN Canada  #TODO check this is standardized across all files.
#
# Marci Detwiller
#

from __future__ import print_function

"""
The base class for an experiment. 
"""

import sys
import copy

from experiment_exception import ExperimentException

import list_tests

# TODO: Set up python path in scons sys.path.append(BOREALIS_PATH)
from utils.experiment_options.experimentoptions import ExperimentOptions
# from radar_control.scan_classes.scan_class_base import ScanClassBase
from experiments.scan_classes.scans import Scan

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


class ExperimentPrototype(object):
    """ A prototype experiment class composed of metadata, including experiment slices (exp_slice) 
    which are dictionaries of radar parameters. Basic, traditional experiments will be composed 
    of a single slice. More complicated experiments will be composed of multiple slices that 
    interface in one of four pre-determined ways, as described under interface_types.
    
    This class is used via inheritance to create experiments.
    
    Some variables shouldn't be changed by the experiment, and their properties do not have setters. 
    Some variables can be changed in the init of your experiment, and can also be modified 
    in-experiment by the class method 'update' in your experiment class. These variables have been
    given property setters.
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
txantennas: The antennas to transmit on, default is all main antennas given max number from config.
rx_main_antennas: The antennas to receive on in main array, default = all antennas given max number 
    from config.
rx_int_antennas : The antennas to receive on in interferometer array, default is all antennas given 
    max number from config.
pulse_sequence: The pulse sequence timing, given in quantities of mpinc, for example 
    normalscan = [0, 14, 22, 24, 27, 31, 42, 43]
mpinc: multi-pulse increment in us, Defines minimum space between pulses.
pulse_shift: Allows phase shifting between pulses, enabling encoding of pulses. Default all zeros 
    for all pulses in pulse_sequence.
pulse_len: length of pulse in us. Range gate size is also determined by this.
nrang: Number of range gates.
frang: first range gate, in km
intt: duration of an integration, in ms. (maximum)
intn: number of averages to make a single integration, if intt = None.
beam_angle: list of beam directions, in degrees off azimuth. Positive is E of N. Array 
    length = number of beams.
beam_order: beam numbers written in order of preference, one element in this list corresponds to 
    one integration period. Can have list within lists. a beam number of 0 in this list gives us 
    beam_angle[0] as a direction.
scanboundflag: flag for whether there is a scan boundary to wait for in order to start a new scan.
scanbound: time that is allotted for a scan before a new scan boundary occurs (ms).
clrfrqrange: range for clear frequency search, should be a list of length = 2, [min_freq, max_freq] 
    in kHz.
txfreq: transmit frequency, in kHz. Note if you specify clrfrqrange it won't be used.
rxfreq: receive frequency, in kHz. Note if you specify clrfrqrange or txfreq it won't be used. Only 
    necessary to specify if you want a receive-only slice.
acf: flag for rawacf and generation. Default True.
xcf: flag for cross-correlation data. Default True
acfint: flag for interferometer autocorrelation data. Default True.
wavetype: default SINE. Any others not currently supported but possible to add in at later date.
seqtimer: timing in us that this object's sequence will begin at, after the start of the sequence.

Should add: 

scanboundt : time past the hour to start a scan at ?  

Explanation of beam_order and beam_angle:
    Traditionally beams have been 3.24 degrees separated but we don't refer to them as beam 
    -19.64 degrees, we refer as beam 1, beam 2. This is like a mapping of beam number
    to beam direction off orthogonal to array. Then you can use the beam numbers in the beam_order 
    list so you can reuse beams within one scan, or use multiple beam numbers in a single 
    integration time, which would trigger an imaging integration. When we do imaging we will still 
    have to quantize the directions we are looking in to certain beam directions.

    """  # TODO add scanboundt?

    def __init__(self, cpid):
        """
        Base initialization for your experiment.
        :param cpid: unique id necessary for each control program (experiment)
        """

        try:
            assert isinstance(cpid, int)  # TODO add check for uniqueness
        except AssertionError:
            errmsg = 'CPID must be a unique int'
            raise ExperimentException(errmsg)

        self.__cpid = cpid

        self.__slice_dict = {}

        self.__new_slice_id = 0

        # Centre frequencies can be specified in your experiment class using the setter. TODO: make modifiable (with warning that it takes time. Get time estimate for this.
        self.__txctrfreq = 13000  # in kHz.
        self.__rxctrfreq = 13000  # in kHz.

        # Load the config, hardware, and restricted frequency data
        self.__options = ExperimentOptions()
        self.__txrate = self.__options.tx_sample_rate  # sampling rate, samples per sec.
        self.__rxrate = self.__options.rx_sample_rate  # sampling rate for rx in samples per sec
        # Transmitting is possible in the range of txctrfreq +/- (txrate/2) because we have iq data
        # Receiving is possible in the range of rxctrfreq +/- (rxrate/2)

        # The following are processing defaults. These can be set by the experiment using the setter
        #   upon instantiation. These are defaults for all slices, but these values are
        #   slice-specific so if the slice is added with these flags specified, that will override
        #   these values for the specific slice.
        self._xcf = True  # cross-correlation
        self._acf = True  # auto-correlation
        self._acfint = True  # interferometer auto-correlation.

        self._interface = {}  # setup_interfacing(self.num_slices)
        # Dictionary of how each exp_slice interacts with the other slices. Default is "NONE" for
        #  all, but must be modified in experiment. NOTE keys are as such: (0,1), (0,2), (1,2),
        # NEVER includes (2,0) etc. The only interface options are those specified in
        # interface_types.

        # The following are for internal use only, and should not be modified in the experimental
        #  class, but will be modified by the class method build_scans. For this reason they
        #  are private, with getters only, in case they are used for reference by the user.
        # These are used internally to build iterable objects out of the slice using the
        # interfacing specified.

        self.__slice_id_scan_lists = None
        self.__scan_objects = []

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
        return len(self.__slice_dict)

    @property
    def slice_keys(self):
        """
        Get the list of slice keys available. This cannot be updated.
        :return: the keys in the current ExperimentPrototype slice_keys dictionary (parameters 
         available for slices)
        """
        return self.__slice_keys

    @property
    def slice_dict(self):
        """
        Get the list of slices. The slice list can be updated in add_slice, edit_slice, and 
        del_slice.
        :return: the list of slice dictionaries in this experiment.
        """
        return self.__slice_dict

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
        return self.__slice_dict.keys()

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
        Get the cross-correlation flag status - note it's only a default for slices without 
        specification.
        :return: cross-correlation flag boolean.
        """
        return self._xcf

    @xcf.setter
    def xcf(self, value):
        """
        To set the cross-correlation flag default for new slices.
        :param value: boolean for cross-correlation processing flag.
        :return: 
        """
        if isinstance(value, bool):
            self._xcf = value
        else:
            pass  # TODO log no change

    @property
    def acf(self):
        """
        Get the auto-correlation flag status - note it's only a default for slices without 
        specification.
        :return: auto-correlation flag boolean.
        """
        return self._acf

    @acf.setter
    def acf(self, value):
        """
        To set the auto-correlation flag default for new slices.
        :param value: boolean for auto-correlation processing flag.
        :return: 
        """
        if isinstance(value, bool):
            self._acf = value
        else:
            pass  # TODO log no change

    @property
    def acfint(self):
        """
        To get the interferometer autocorrelation flag - note it's only a default for slices.
        :return: interferometer autocorrelation flag boolean.
        """
        return self._acfint

    @acfint.setter
    def acfint(self, value):
        """
        To set the interferometer autocorrelation flag default for new slices.
        :param value: boolean for interferometer autocorrelation processing flag.
        :return: 
        """
        if isinstance(value, bool):
            self._acfint = value
        else:
            pass  # TODO log no change

    @property
    def txrate(self):
        """
        To get the transmission sample rate to the DAC.
        :return: the transmission sample rate to the DAC (Hz).
        """
        return self.__txrate

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
            pass  # TODO errors / log no change

    @property
    def tx_maxfreq(self):
        """
        :return: the maximum tx frequency possible in this experiment (either maximum in our license
         or maximum given by the centre frequency and sampling rate).
        """
        max_freq = self.txctrfreq + (self.txrate/2.0)
        if max_freq < self.options.max_freq:
            return max_freq
        else:
            return self.options.max_freq

    @property
    def tx_minfreq(self):
        """
        :return: the minimum tx frequency possible in this experiment (either minimum in our license
         or minimum given by the centre frequency and sampling rate).
        """
        min_freq = self.txctrfreq - (self.txrate/2.0)
        if min_freq > self.options.min_freq:
            return min_freq
        else:
            return self.options.min_freq

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
    def rxrate(self):
        """
        :return: rx sampling rate in samples per sec. Comes from config file.
        """
        return self.__rxrate

    @property
    def rx_maxfreq(self):
        """
        :return: the maximum tx frequency possible in this experiment (either maximum in our license
         or maximum given by the centre frequency and sampling rate).
        """
        max_freq = self.rxctrfreq + (self.rxrate/2.0)
        if max_freq < self.options.max_freq:
            return max_freq
        else:
            return self.options.max_freq

    @property
    def rx_minfreq(self):
        """
        :return: the minimum tx frequency possible in this experiment (either minimum in our license
         or minimum given by the centre frequency and sampling rate).
        """
        min_freq = self.rxctrfreq - (self.rxrate/2.0)
        if min_freq > self.options.min_freq:
            return min_freq
        else:
            return self.options.min_freq

    @property
    def interface(self):
        """
        To get the list of interfacing for the experiment slices.  Interfacing should be set up 
        for any slice when it gets added, ie. in add_slice.
        :return:the list of interfacing defined as [(slice_id1, slice_id2) : INTERFACING_TYPE] for
                all current slice_ids. 
        """
        return self._interface

    @property
    def slice_id_scan_lists(self):
        """
        :return: the list of scan slice ids (a list of lists of slice_ids, organized by scan).
        """
        return self.__slice_id_scan_lists

    @property
    def scan_objects(self):
        """
        To get the list of scan_objects for use in radar_control. 
        :return: scan_objects, the list of instances of the Scan class.
        """
        return self.__scan_objects

    def add_slice(self, exp_slice, interfacing_dict=None):
        """
        Add a slice to the experiment.
        :param exp_slice: a slice (dictionary of slice_keys) to add to the experiment.
        :param interfacing_dict: dictionary of type {slice_id : INTERFACING , ... } that defines how
         this slice interacts with all the other slices currently in the experiment.
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
        self.__slice_dict[new_exp_slice['slice_id']] = new_exp_slice
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
        """
        Remove a slice from the experiment.
        :param remove_slice_id: the id of the slice you'd like to remove.
        :return: boolean True if successful
        """
        if isinstance(remove_slice_id, int) and remove_slice_id in self.slice_ids:
            del(self.slice_dict[remove_slice_id])
            for key1, key2 in self._interface.keys():
                if key1 == remove_slice_id or key2 == remove_slice_id:
                    del self._interface[(key1, key2)]
            return True
        else:
            return False
            # TODO log that it cannot be removed
            # errmsg = 'Cannot remove slice id {} : it does not exist'.format(remove_slice_id)
            # raise ExperimentException(errmsg)

    def edit_slice(self, edit_slice_id, param1, value1, param2=None, value2=None, param3=None,
                   value3=None):
        """
        A quick way to edit a slice. In reality this is actually adding a new slice and deleting 
        the old one. Useful for quick changes - if you have to change more than three parameters 
        then should do your own copy / add / delete.
        :param edit_slice_id: the slice id of the slice to be edited.
        :param param1: the slice_key that is wished to be changed.
        :param value1: the new value of the slice_key
        :param param2: the 2nd slice_key that is wished to be changed.
        :param value2: the new value of the 2nd slice_key
        :param param3: the 3rd slice_key that is wished to be changed.
        :param value3: the new value of the 3rd slice_key
        :return: the new slice id of the edited slice.
        """
        if isinstance(edit_slice_id, int) and edit_slice_id in self.slice_ids:
            if isinstance(param1, str) and param1 in self.slice_keys:
                edited_slice = self.slice_dict[edit_slice_id].copy()
                edited_slice[param1] = value1
                if param2 is not None:
                    edited_slice[param2] = value2
                if param3 is not None:
                    edited_slice[param3] = value3
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
            self.slice_dict, self.interface)
        return represent

    def build_scans(self):
        """ 
        Will be run by experiment handler, to build iterable objects for radar_control to use. 
        Creates scan_objects and slice_id_scan_lists in the experiment for identifying which 
        slices are in the scans.
        """

        # Check interfacing
        self.self_check()
        self.check_interfacing()

        # investigating how I might go about using this base class - TODO maybe make a new IterableExperiment class to inherit
        # from ScanClassBase ? Then could have the slice_combos_sorter function as a method of scanclassbase
        # iterable_experiment = ScanClassBase(self.slice_keys, self.slice_dict, self.interface, self.options)

        self.__slice_id_scan_lists = self.get_scans()
        print("All experiment slice ids: {}".format(self.slice_ids))
        print("Scan Slice Id list : {}".format(self.__slice_id_scan_lists))
        # Returns list of scan lists. Each scan list is a list of the slice_ids for the slices
        # included in that scan.
        for scan_list in self.__slice_id_scan_lists:
            slices_for_scan = {}
            for slice_id in scan_list:
                try:
                    slices_for_scan[slice_id] = self.slice_dict[slice_id]
                except KeyError:
                    errmsg = 'Error with slice list - slice {} cannot be found.'.format(slice_id)
                    raise ExperimentException(errmsg)

            # Create smaller interfacing dictionary for this scan specifically.
            # This dictionary will only include the slices in this scan, therefore it will not include any SCAN interfacing.
            scan_interface_keys = []
            for m in range(len(scan_list)):
                for n in range(m + 1, len(scan_list)):
                    scan_interface_keys.append(tuple([scan_list[m], scan_list[n]]))
            scan_interface = {}
            for k in scan_interface_keys:
                scan_interface[k] = self.interface[k]

            self.__scan_objects.append(Scan(scan_list, slices_for_scan, scan_interface,
                                            self.options))
            # Append a scan instance, passing in the list of slice ids to include in scan.

        if __debug__:
            print("Number of Scan types: {}".format(len(self.__scan_objects)))
            print("Number of AveragingPeriods in Scan #1: {}".format(len(self.__scan_objects[
                                                                             0].aveperiods)))
            print("Number of Sequences in Scan #1, Averaging Period #1: {}".format(
                len(self.__scan_objects[0].aveperiods[0].sequences)))
            print("Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1:"
                  " {}".format(len(self.__scan_objects[0].aveperiods[0].sequences[0].slice_dict)))

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
                scan_combos.append(list(k))

        print(scan_combos)

        combos = list_tests.slice_combos_sorter(scan_combos, self.slice_ids)

        return combos

    def check_slice_minimum_requirements(self, exp_slice):
        """
        Check for the minimum requirements of the slice. The following keys are always required:
        "pulse_sequence", "mpinc", "pulse_len", "nrang", "frang", (one of "intt" or "intn"), 
        "beam_angle", and "beam_order". This function may modify these keys. Ensure the values 
        make sense.
        :param exp_slice: slice to check.
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
                assert isinstance(exp_slice['intt'], float) or isinstance(exp_slice['intt'], int)
                exp_slice['intt'] = float(exp_slice['intt'])
            except AssertionError:
                errmsg = "intt must be an number"
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
        :param exp_slice: slice in which to set identifiers
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
        :param exp_slice: the slice to check, before adding to the experiment.
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
                assert (exp_slice['clrfrqrange'][1] * 1000) < self.tx_maxfreq
                assert (exp_slice['clrfrqrange'][0] * 1000) > self.tx_minfreq
                assert (exp_slice['clrfrqrange'][1] * 1000) < self.rx_maxfreq
                assert (exp_slice['clrfrqrange'][0] * 1000) > self.rx_minfreq
            except AssertionError:
                errmsg = """clrfrqrange must be between min and max tx frequencies {} and rx 
                            frequencies {} according to license and/or centre frequencies / sampling 
                            rates, and must have lower frequency first.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
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
                            exp_slice['clrfrqrange'][0] = freq_range[1] + 1
                            # outside of restricted range now.
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
                            exp_slice['clrfrqrange'][1] = freq_range[0] - 1
                            # outside of restricted range now.
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
                assert (exp_slice['rxfreq'] * 1000) < self.rx_maxfreq
                assert (exp_slice['rxfreq'] * 1000) > self.rx_minfreq
            except AssertionError:
                errmsg = """rxfreq must be a number (kHz) between rx min and max frequencies {} for
                            the radar license and be within range given centre frequency and 
                            sampling rate.""".format((self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)

        else:  # TX-specific mode , without a clear frequency search.
            # In this mode, txfreq is required along with the other requirements.
            try:
                assert isinstance(exp_slice['txfreq'], int) or isinstance(exp_slice['txfreq'],
                                                                          float)
                assert (exp_slice['txfreq'] * 1000) < self.tx_maxfreq
                assert (exp_slice['txfreq'] * 1000) > self.tx_minfreq
                assert (exp_slice['txfreq'] * 1000) < self.rx_maxfreq
                assert (exp_slice['txfreq'] * 1000) > self.rx_minfreq
            except AssertionError:
                errmsg = """txfreq must be a number (kHz) between tx min and max frequencies {} and
                            rx min and max frequencies {} for the radar license and be within range
                            given centre frequencies and sampling rates.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            for freq_range in self.options.restricted_ranges:
                try:
                    assert exp_slice['txfreq'] not in range(freq_range[0], freq_range[1])
                except AssertionError:
                    errmsg = """txfreq is within a restricted frequency range {}
                             """.format(freq_range)
                    raise ExperimentException(errmsg)

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
            slice_with_defaults['acf'] = self.acf
        if 'xcf' not in exp_slice:
            slice_with_defaults['xcf'] = self.xcf
        if 'acfint' not in exp_slice:
            slice_with_defaults['acfint'] = self.acfint
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
            assert 'rxfreq' in complete_slice.keys() or 'txfreq' in complete_slice.keys() \
                   or 'clrfrqrange' in complete_slice.keys()
        except AssertionError:
            errmsg = 'An rxfreq, txfreq, or clrfrqrange must be specified in a slice'
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

        # TODO: somehow check if self.cpid is not unique - incorporate known cpids from git repo?

        for a_slice in self.slice_ids:
            selferrs = self.check_slice(self.slice_dict[a_slice])
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
                elif param == 'clrfrqrange' and not exp_slice['clrfrqflag']:
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