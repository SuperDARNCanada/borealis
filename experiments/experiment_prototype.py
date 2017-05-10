#!/usr/bin/env python

# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# REVIEW #37 Best way to call interpreter? there's also "/usr/bin/env python" that allows you to
# have python installed anywhere REPLY sure
# REVIEW #7 We need some kind of license at top of all files - or a referral to the
# license/copyright/etc REPLY agreed we should discuss and standardize this for all our files

"""
The template for an experiment. 
"""

import json
import sys

# REPLY going to try making the scans into a separate package and then import because this is a relative path to
# the running directory TODO: Set up python path in scons
from utils.experiment_options.experimentoptions import ExperimentOptions
from radar_control.scan_classes import scans

# REPLY question : should all of this be inside the ExperimentPrototype class

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

MINIMUM_PULSE_LENGTH = 0.01  # us


def setup_interfacing(exp_slice_num):
    if_keys = []
    for exp_slice1 in range(exp_slice_num):
        for exp_slice2 in range(exp_slice1 + 1, exp_slice_num):
            if_keys.append((exp_slice1, exp_slice2))

    if_dict = dict((key, "NONE") for key in if_keys)

    return if_dict


def selfcheck_slice(exp_slice, options):  
    """
    This is the first test of the dictionary in the experiment done to ensure values in this slice make sense. This 
    is a self-check to ensure the parameters (for example, txfreq, antennas) are appropriate.
    """
    error_count = 0
    error_dict = {}

    # check none greater than number of transmitting antennas, no duplicates REVIEW #0 Since we decided for now that we would
    # receive on all antennas, the check for rxchannels should be equal to main+inter count. REPLY: I disagree this
    # is where you set up which antenna channels you want to include in your data so you can have less channels if you
    # want to! - but should include interferometer probably separately in another list
    if len(exp_slice['txantennas']) > options.main_antenna_count:
        error_dict[error_count] = "Slice {} Has Too Many Main TX Antenna Channels {} Greater than Config {}" \
            .format(exp_slice['slice_id'], len(exp_slice['txantennas']), options.main_antenna_count)
    if len(exp_slice['rx_main_antennas']) > options.main_antenna_count:
        # REVIEW #39 These errors could just be appended to a list instead of using a dictionary. - REPLY or I could
        # TODO use our own exception class for these errors.
        error_dict[error_count] = "Slice {} Has Too Many Main RX Antenna Channels {} Greater than Config {}" \
            .format(exp_slice['slice_id'], len(exp_slice['rx_main_antennas']), options.main_antenna_count)
        error_count = error_count + 1
    if len(exp_slice['rx_int_antennas']) > options.interferometer_antenna_count:
        error_dict[error_count] = "Slice {} Has Too Many RX Interferometer Antenna Channels {} Greater than Config {}" \
            .format(exp_slice['slice_id'], len(exp_slice['rx_int_antennas']), options.interferometer_antenna_count)
        error_count = error_count + 1
        
    # Check if the antenna identifier number is greater than the config file's 
    # maximum antennas for all three of tx antennas, rx antennas and rx int antennas
    # Also check for duplicates
    for i in range(len(exp_slice['txantennas'])):
        if exp_slice['txantennas'][i] >= options.main_antenna_count:
            error_dict[error_count] = "Slice {} Specifies Main Array Antenna Numbers Over Config Max {}" \
                .format(exp_slice['slice_id'], options.main_antenna_count)
            error_count = error_count + 1

    no_duplicates = set()
    for ant in exp_slice['txantennas']:
        if ant not in no_duplicates:
            no_duplicates.add(ant)
        else:
            error_dict[error_count] = "Slice {} TX Main Antennas Has Duplicate Antennas".format(exp_slice['slice_id'])
            error_count = error_count + 1

    for i in range(len(exp_slice['rx_main_antennas'])):
        if exp_slice['rx_main_antennas'][i] >= options.main_antenna_count:
            error_dict[error_count] = "Slice {} Specifies Main Array Antenna Numbers Over Config Max {}" \
                .format(exp_slice['slice_id'], options.main_antenna_count)
            error_count = error_count + 1

            # TODO use the same method as above
        for j in range(i + 1, len(exp_slice['rx_main_antennas'])):
            if exp_slice['rx_main_antennas'][i] == exp_slice['rx_main_antennas'][j]:
                error_dict[error_count] = "Slice {} RX Main Antennas Has Duplicate Antennas" \
                    .format(exp_slice['slice_id'])
                error_count = error_count + 1
    for i in range(len(exp_slice['rx_int_antennas'])):
        if exp_slice['rx_int_antennas'][i] >= options.interferometer_antenna_count:
            error_dict[error_count] = "Slice {} Specifies Interferometer Array Antenna Numbers Over Config Max {}" \
                .format(exp_slice['slice_id'], options.interferometer_antenna_count)
            error_count = error_count + 1

            # TODO use the same method as above
        for j in range(i + 1, len(exp_slice['rx_int_antennas'])):
            if exp_slice['rx_int_antennas'][i] == exp_slice['rx_int_antennas'][j]:
                error_dict[error_count] = "Slice {} RX Interferometer Antennas Has Duplicate Antennas" \
                    .format(exp_slice['slice_id'])
                error_count = error_count + 1

    # Check if the pulse_sequence is not increasing, which would be an error
    if not all(x < y for x, y in zip(exp_slice['pulse_sequence'], exp_slice['pulse_sequence'][1:])):
        error_dict[error_count] = "Slice {} pulse_sequence Not Increasing".format(exp_slice['slice_id'])
        error_count = error_count + 1

    # Check that pulse_len and mpinc make sense (values in us)
    if exp_slice['pulse_len'] > exp_slice['mpinc']:
        error_dict['error_count'] = "Slice {} Pulse Length Greater than MPINC".format(exp_slice['slice_id'])
        error_count = error_count + 1
    if exp_slice['pulse_len'] < MINIMUM_PULSE_LENGTH:
        error_dict[error_count] = "Slice {} Pulse Length Too Small".format(exp_slice['slice_id'])
        error_count = error_count + 1

    # check intn and intt make sense given mpinc, and pulse_sequence.
    seq_len = exp_slice['mpinc'] * (exp_slice['pulse_sequence'][-1] + 1)
    # REVIEW #0 Do you really need the +1 here? Also need to take into account the wait time at end of pulse_sequence
    # (ss_delay in current system) REPLY agree TODO will fix

    # TODO: Check these
    #        self.intt=3000 # duration of the direction, in ms
    #        self.intn=21 # max number of averages (aka full pulse_sequences transmitted)
    # in an integration time intt (default 3s)

    # Traditionally beams have been 3.75 degrees separated but we
    # don't refer to them as beam -22.75 degrees, we refer as beam 1, beam 2. This is like a mapping of beam number
    # to beam direction off azimuth . Then you can use the beam numbers in the scan list so you can reuse beams
    # within one scan, or use multiple beamnumbers in a single integration time. Open to revision. I imagine when we
    # do imaging we will still have to quantize the directions we are looking in to certain beam directions,
    # and if you wanted to add directions you could do so in experiment modification functions.
    
    # exp_slice['beamdir'] is going to be a list of possible beam directions for this
    # experiment slice in degrees off azimuth. It doesn't mean that these are the beam directions that will be used in this scan, those beams are in the scan list #TODO: update this comment with the new keys
    for i in range(len(exp_slice['beamdir'])): # TODO: Make methods 'check_for_duplicates(dict, key)' and 'check_for_proper_order(dict,key,increasing)'
        for j in range(i + 1, len(exp_slice['beamdir'])):
            if exp_slice['beamdir'][i] == exp_slice['beamdir'][j]:
                error_dict[error_count] = "Slice {} Beam Direction List Has Duplicates".format(exp_slice['slice_id'])
                error_count = error_count + 1
            if exp_slice['beamdir'][i] > exp_slice['beamdir'][j]:
                error_dict[error_count] = "Slice {} Beam Directions Not in Order \
                    Clockwise (E of N is positive)".format(exp_slice['slice_id'])
                error_count = error_count + 1

    # Check if the list of beams to transmit on is empty
    if not exp_slice['scan']:
        error_dict[error_count] = "Slice {} Scan Empty".format(exp_slice['slice_id'])
        error_count = error_count + 1

    # Check that the beam numbers in scan exist  # TODO: Update comment
    for bmnum in exp_slice['scan']:
        if bmnum >= len(exp_slice['beamdir']):
            error_dict[error_count] = "Slice {} Scan Beam Number {} DNE".format(exp_slice['slice_id'], bmnum)
            error_count = error_count + 1

    # check scan boundary not less than minimum required scan time. 
    if exp_slice['scanboundflag']: 
        if exp_slice['scanbound'] < (len(exp_slice['scan']) * exp_slice['intt']):  # REVIEW #5 TODO Need units
            # documented somewhere highly visible for scanbound, intt
            error_dict[error_count] = "Slice {} Scan Too Long for ScanBoundary".format(exp_slice['slice_id'])
            error_count = error_count + 1

    # TODO other checks

    # freq=12300 # in kHz
    # clrfrqf=1 # flag for clear frequency search
    # clrfrqrange=300 # clear frequency range if searching
    # receiving params
    # xcf=1 # flag for cross-correlation
    # acfint=1 # flag for getting lag-zero power of interferometer
    # wavetype='SINE'
    # seqtimer=0
    # cpid is a unique value? # REVIEW Talked about placing a file with all known cpids in git repo

    return error_dict


class ExperimentPrototype(object):
    """A prototype experiment class composed of metadata, including experiment slices (exp_slice) which are 
    dictionaries of radar parameters. Basic, traditional experiments will be composed of a single slice. More 
    complicated experiments will be composed of multiple slices that interface in one of four pre-determined ways, 
    as described in more detail below. 
    
    This class is used via inheritance to create experiments.

    :param cpid: unique id necessary for each control program (experiment)
    :param num_slices: number of experiment slices in this experiment.
    :
    """

    def __init__(self, cpid, num_slices):

        self.cpid = cpid
        self.num_slices = num_slices
        slice_list = []

        self.slice_keys = ["slice_id", "cpid", "txantennas", "rx_main_antennas", "rx_int_antennas", "pulse_sequence",
                           "pulse_shift", "mpinc", "pulse_len", "nrang", "frang", "intt", "intn", "beamdir", "scan",
                           "scanboundflag", "scanbound", "txfreq", "rxfreq", "clrfrqf", "clrfrqrange", "xcf", "acfint",
                           "wavetype", "seqtimer"]

        """ The slice keys are described as follows: """  # TODO Can we change the name of scan to 'beam_order' and perhaps 'beamdir' can be changed to 'beam_angle'

        for num in range(self.num_slices):
            exp_slice = {key: None for key in self.slice_keys}
            exp_slice["slice_id"] = num
            exp_slice["cpid"] = self.cpid
            slice_list.append(exp_slice)

        self.slice_list = slice_list

        # Load the config data
        self.options = ExperimentOptions()

        # Next some metadata that you can change, with defaults.
        # TODO: make default none and have a function that will
        #   calculate something appropriate if left blank.

        self.txctrfreq = 12000  # in kHz.
        self.txrate = 12000000  # sampling rate, samples per sec
        self.rxctrfreq = 12000  # in kHz.
        # NOTE: rx sampling rate is set in config.

        self.xcf = 1 # TODO: Make into booleans, we should
        # Get cross-correlation data in processing block.

        self.acfint = 1
        # Determine lag-zero interferometer power in fitacf.

        self.interface = setup_interfacing(self.num_slices) 
        # Dictionary of how each exp_slice interacts with the other slices. Default is "NONE" for all, but must be modified
        # in experiment. NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc. The only interface
        # options are: interface_types = frozenset(['SCAN', 'INTTIME', 'INTEGRATION', 'PULSE'])

        self.slice_id_scan_lists = None
        self.scan_objects = []
        # These are used internally by the radar_control block to build iterable objects out of the slice using the
        # interfacing specified.

    def __repr__(self):
        represent = 'self.cpid = {}\nself.num_slices = {}\nself.slice_keys = {}\nself.slice_list = {}\nself.options = \
        {}\nself.txctrfreq = {}\nself.txrate = {}\nself.rxctrfreq = {}\nself. xcf = {}\nself.acfint = {}\nself.i\
        nterface = {}\n'.format(self.cpid, self.num_slices, self.slice_keys, self.slice_list, self.options,
                                self.txctrfreq, self.txrate, self.rxctrfreq, self.xcf, self.acfint, self.interface)
        return represent

    def __str__(self):
        represent = 'CPID [cpid]: {}'.format(self.cpid)
        represent += '\nNum of Slices [num_slices]: {}'.format(self.num_slices)
        for exp_slice in self.slice_list:
            represent += '\nSlice {} : '.format(exp_slice['slice_id']) + exp_slice
        represent += '\nInterfacing [interface]: {}'.format(self.interface)
        return represent

    def build_scans(self):
        """ 
        Will be run by experiment handler, to build iterable objects for radar_control to use.
        """

        # Check interfacing
        self.self_check()
        self.check_interfacing()
        self.slice_id_scan_lists = self.get_scans()
        # Returns list of scan lists. Each scan list is a list of the slice_ids for the slices included in the scan.
        if self.slice_id_scan_lists:  # if list is not empty, can make scans
            self.scan_objects = [scans.Scan(self, scan_list) for scan_list in self.slice_id_scan_lists]
        # Append a scan instance, passing in the list of slice ids to include in scan.
        else:
            pass  # TODO error

    def self_check(self):
        """
        Check that the values in this experiment are valid
        """

        if self.num_slices < 1:
            errmsg = "Error: No slices in control program"
            sys.exit(errmsg)  # REVIEW 6 Add a todo for error handling. Perhaps use exceptions instead.

        # TODO: somehow check if self.cpid is not unique

        for slice in range(self.num_slices):
            selferrs = selfcheck_slice(self.slice_list[slice], self.options)
            if not selferrs:
                # If returned error dictionary is empty
                continue
            errmsg = "Self Check Errors Occurred with slice Number : {} \nSelf \
                Check Errors are : {}".format(slice, selferrs)
            sys.exit(errmsg)  #TODO error handling. Perhaps use exceptions instead.
        else:  # no break (exit)
            print "No Self Check Errors. Continuing..."
        return None

    # REVIEW #30 Consider splitting interface/error checking into a seperate class. It would decouple this behaviour
    # and could simplify the experiment significantly. The experiment handler would be able to make sure that the
    # experiment is error checked, and someone developing the experiment could just run their experiment through it
    # in the interpretter or something. REPLY - OK TODO: Make experiment_checker class that you can pass an experiment object - it would spit out pass/fail for each criteria and maybe suggest what is wrong? possible recovery from errors?
    def check_interfacing(self):
        """
        Check that the keys in the interface are not NONE and are
        valid.
        """

        for key, interface_type in self.interface.items():
            if interface_type == "NONE":
                errmsg = 'Interfacing is still default, must set key {}'.format(key)
                sys.exit(errmsg)  # REVIEW 6 Add a TODO for error handling. Perhaps use exceptions instead. REPLY OK

        for num1, num2 in self.interface.iterkeys():  
            if num1 >= self.num_slices or num2 >= self.num_slices or num1 < 0 or num2 < 0:
                # This is required for how I have it set up. Avoids any confusion
                #  with keys [0,2] vs [2,0] for example. Because you could add your own keys I check it.
                errmsg = """Interfacing key ({}, {}) is not valid, all keys must refer to (slice_id1,
                slice_id2) where slice_id1 < slice_id2""".format(num1, num2)
                sys.exit(errmsg)  # TODO for error handling. Perhaps use exceptions instead.
            if self.interface[num1, num2] not in interface_types:
                errmsg = 'Interfacing Not Valid Type between Slice_id {} and Slice_id {}'.format(num1, num2)
                sys.exit(errmsg)  # TODO for error handling. Perhaps use exceptions instead.

    def get_scans(self):
        """
        Take my own interfacing and get info on how many scans and which slices make which scans.
        :rtype list
        :return list of lists. The list has one element per scan. Each element is a list of slice_id's signifying which
         slices are combined inside that scan. The element could be a list of length 1, meaning only one slice_id is 
         included in that scan. The list returned could be of length 1, meaning only one scan is present in the 
         experiment.
        """
        scan_combos = []

        for num1, num2 in self.interface.iterkeys():  # REPLY: Ok but need to specify keys so we don't get key,value TODO: But you can just use k in place of num1, num2 everywhere here
            if self.interface[num1, num2] == "PULSE" or self.interface[num1, num2] == "INT_TIME" or \
                    self.interface[num1, num2] == "INTEGRATION":
                scan_combos.append([num1, num2])
                # Save the keys that are scan combos (that != SCAN, as that would mean they are in separate scans) # TODO: Just check self.interface[num1, num2] != "SCAN"

        scan_combos = sorted(scan_combos)
        # if [2,4] and [1,4], then also must be [1,2] in the scan_combos
        # Now we are going to modify the list of lists of length = 2 to be a list of length x so that if [1,2] and [2,4]
        # and [1,4] are in scan_combos, we want only one list element for this scan : [1,2,4] .
        scan_i = 0  # REVIEW #3 This needs a detailed explaination with examples. REPLY OK TODO
        while scan_i < len(scan_combos):  # i: element in scan_combos (representing one scan)
            slice_id_k = 0
            while slice_id_k < len(scan_combos[scan_i]):  # k: element in scan (representing a slice)
                scan_j = scan_i + 1  # j: iterates through the other elements of scan_combos, to combine them into
                # the first, i, if they are in fact part of the same scan.
                while scan_j < len(scan_combos):
                    if scan_combos[scan_i][slice_id_k] == scan_combos[scan_j][0]:  # if an element (slice_id) inside
                        # the i scan is the same as a slice_id in the j scan (somewhere further in the scan_combos),
                        # then we need to combine that j scan into the i scan. We only need to check the first element
                        # of the j scan because scan_combos has been sorted and we know the first slice_id in the scan
                        # is less than the second slice id.
                        add_n_slice_id = scan_combos[scan_j][1]  # the slice_id to add to the i scan from the j scan.
                        scan_combos[scan_i].append(add_n_slice_id)
                        # Combine the indices if there are 3+ slices combining in same scan
                        for m in range(0, len(scan_combos[scan_i]) - 1):  # if we have added z to scan_i, such that
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
                                    interface the same with exp_slice {}'.format(scan_combos[scan_i][m],
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
