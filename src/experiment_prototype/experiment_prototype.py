#!/usr/bin/env python3

"""
    experiment_prototype
    ~~~~~~~~~~~~~~~~~~~~
    This is the base module for all experiments. An experiment will only run if it inherits from
    this class.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
# built-in
import copy
import itertools
import math
import os

# third-party
import numpy as np
from pathlib import Path
import re
from scipy.constants import speed_of_light

# local
from utils.options import Options
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.scan_classes.scans import Scan
from experiment_prototype.scan_classes.scan_class_base import ScanClassBase
from experiment_prototype.decimation_scheme.decimation_scheme import create_default_scheme, DecimationScheme
from experiment_prototype import list_tests

BOREALISPATH = os.environ['BOREALISPATH']

interface_types = tuple(['SCAN', 'AVEPERIOD', 'SEQUENCE', 'CONCURRENT'])
""" The types of interfacing available for slices in the experiment.

Interfacing in this case refers to how two or more components are meant to be run together. The
following types of interfacing are possible:

1. SCAN.
The scan by scan interfacing allows for slices to run a scan of one slice, followed by a scan of the
second. The scan mode of interfacing typically means that the slice will cycle through all of its
beams before switching to another slice.

There are no requirements for slices interfaced in this manner.

2. AVEPERIOD.
This type of interfacing allows for one slice to run its averaging period (also known as integration
time or integration period), before switching to another slice's averaging period. This type of
interface effectively creates an interleaving scan where the scans for multiple slices are run 'at
the same time', by interleaving the averaging periods.

Slices which are interfaced in this manner must share:
    - the same SCANBOUND value.

3. SEQUENCE.
Sequence interfacing allows for pulse sequences defined in the slices to alternate between each
other within a single averaging period. It's important to note that data from a single slice is
averaged only with other data from that slice. So in this case, the averaging period is running two
slices and can produce two averaged datasets, but the sequences within the averaging period are
interleaved.

Slices which are interfaced in this manner must share:
    - the same SCANBOUND value.
    - the same INTT or INTN value.
    - the same BEAM_ORDER length (scan length)

4. CONCURRENT.
Concurrent interfacing allows for pulse sequences to be run together concurrently. Slices will have
their pulse sequences summed together so that the data transmits at the same time. For example,
slices of different frequencies can be mixed simultaneously, and slices of different pulse sequences
can also run together at the cost of having more blanked samples. When slices are interfaced in this
way the radar is truly transmitting and receiving the slices simultaneously.

Slices which are interfaced in this manner must share:
    - the same SCANBOUND value.
    - the same INTT or INTN value.
    - the same BEAM_ORDER length (scan length)
    - the same DECIMATION_SCHEME

"""

slice_key_set = frozenset([
    "acf",
    "acfint",
    "align_sequences",
    "averaging_method",
    "beam_angle",
    "clrfrqrange",
    "comment",
    "cpid",
    "first_range",
    "freq",
    "intn",
    "intt",
    "lag_table",
    "num_ranges",
    "pulse_len",
    "pulse_phase_offset",
    "pulse_sequence",
    "range_sep",
    "rx_beam_order",
    "rx_int_antennas",
    "rx_main_antennas",
    "scanbound",
    "seqoffset",
    "slice_id",
    "tau_spacing",
    "tx_antennas",
    "tx_antenna_pattern",
    "tx_beam_order",
    "wait_for_first_scanbound",
    "xcf",
    ])

hidden_key_set = frozenset(['rxonly', 'clrfrqflag', 'slice_interfacing'])
"""
These are used by the build_scans method (called from the experiment_handler every time the
experiment is run). If set by the user, the values will be overwritten and therefore ignored.
"""

possible_averaging_methods = frozenset(['mean', 'median'])
possible_scheduling_modes = frozenset(['common', 'special', 'discretionary'])
default_rx_bandwidth = 5.0e6
default_output_rx_rate = 10.0e3/3
transition_bandwidth = 750.0e3


class ExperimentPrototype(object):
    """
    The base class for all experiments. This class is used via inheritance to create experiments.

    A prototype experiment class composed of metadata, including experiment slices (exp_slice)
    which are dictionaries of radar parameters. Basic, traditional experiments will be composed of
    a single slice. More complicated experiments will be composed of multiple slices that
    interface in one of four pre-determined ways, as described under interface_types.

    Some variables shouldn't be changed by the experiment, and their properties do not have setters.
    Some variables can be changed in the init of your experiment, and can also be modified
    in-experiment by the class method 'update' in your experiment class. These variables have been
    given property setters.

    The following are the user-modifiable attributes of the ExperimentPrototype that are
    used to make an experiment. Other parameters are set in the init and cannot be modified after 
    instantiation.

    * xcf:          boolean for cross-correlation data. A default can be set here for slices, \
                    but any slice can override this setting with the xcf slice key.
    * acf:          boolean for auto-correlation data on main array. A default can be set here for \
                    slices, but any slice can override this setting with the acf slice key.
    * acfint:       boolean for auto-correlation data on interferometer array. A default can be set \
                    here for slices, but any slice can override this setting with the acfint slice \
                    key.
    * slice_dict:   modifiable only using the add_slice, edit_slice, and del_slice methods.
    * interface:    modifiable using the add_slice, edit_slice, and del_slice methods, or by \
                    updating the interface dict directly.

    :param  cpid:               Unique id necessary for each control program (experiment). Cannot be
                                changed after instantiation.
    :type   cpid:               int
    :param  output_rx_rate:     The desired output rate for the data, to be decimated to, in Hz.
                                Cannot be changed after instantiation. Default 3.333 kHz.
    :type  output_rx_rate:      float
    :param  rx_bandwidth:       The desired bandwidth for the experiment. Directly determines rx
                                sampling rate of the USRPs. Cannot be changed after instantiation.
                                Default 5.0 MHz.
    :type   rx_bandwidth:       float
    :param  tx_bandwidth:       The desired tx bandwidth for the experiment. Directly determines tx
                                sampling rate of the USRPs. Cannot be changed after instantiation.
                                Default 5.0 MHz.
    :type  tx_bandwidth:        float
    :param  txctrfreq:          Center frequency, in kHz, for the USRP to mix the samples with.
                                Since this requires tuning time to set, it cannot be modified after
                                instantiation.
    :type   txctrfreq:          float
    :param  rxctrfreq:          Center frequency, in kHz, used to mix to baseband. Since this
                                requires tuning time to set, it cannot be modified after
                                instantiation.
    :type   rxctrfreq:          float
    :param  comment_string:     Description of experiment for data files. This should be used to
                                describe your overall experiment design. Another comment string
                                exists for every slice added, to describe information that is
                                slice-specific.
    :type   comment_string:     str

    :raises ExperimentException:    if cpid is not an integer, cannot be represented by a 16-bit
                                    signed integer, is not unique, or is not a positive value
    :raises ExperimentException:    if output sample rate is too high
    :raises ExperimentException:    if transmit bandwidth is too large or not an integer multiple of
                                    USRP clock rate
    :raises ExperimentException:    if receive bandwidth is too large or not an integer multiple of
                                    USRP clock rate
    """

    def __init__(self, cpid, output_rx_rate=default_output_rx_rate, rx_bandwidth=default_rx_bandwidth,
                 tx_bandwidth=5.0e6, txctrfreq=12000.0, rxctrfreq=12000.0, comment_string=''):
        if not isinstance(cpid, int):
            errmsg = 'CPID must be a unique int'
            raise ExperimentException(errmsg)
        if cpid > np.iinfo(np.int16).max:
            errmsg = 'CPID must be representable by a 16-bit signed integer'
            raise ExperimentException(errmsg)
        # Quickly check for uniqueness with a search in the experiments directory first taking care
        # not to look for CPID in any experiments that are just tests (located in the testing
        # directory)
        experiment_files_list = list(Path(f"{BOREALISPATH}/src/borealis_experiments/").glob("*.py"))
        self.__experiment_name = self.__class__.__name__  
        # TODO use this to check the cpid is correct using pygit2, or __class__.__module__ for module name
        # TODO replace below cpid local uniqueness check with pygit2 or some reference
        #  to a database to to ensure CPID uniqueness and to ensure CPID is entered in the database
        #  for this experiment (this CPID is unique AND its correct given experiment name)
        cpid_list = []
        for experiment_file in experiment_files_list:
            with open(experiment_file) as file_to_search:
                for line in file_to_search:
                    # Find the name of the class in the file and break if it matches this class
                    experiment_class_name = re.findall("class.*\(ExperimentPrototype\):", line)
                    if experiment_class_name:
                        # Parse out just the name from the experiment, format will be like this:
                        # ['class IBCollabMode(ExperimentPrototype):']
                        atomic_class_name = experiment_class_name[0].split()[1].split('(')[0]
                        if self.__experiment_name == atomic_class_name:
                            break

                    # Find any lines that have 'cpid = [integer]'
                    existing_cpid = re.findall("cpid.?=.?[0-9]+", line)
                    if existing_cpid:
                        cpid_list.append(existing_cpid[0].split('=')[1].strip())

        if str(cpid) in cpid_list:
            errmsg = f'CPID must be unique. {cpid} is in use by another local experiment'
            raise ExperimentException(errmsg)
        if cpid <= 0:
            errmsg = 'The CPID should be a positive number in the experiment. Borealis'\
                     ' will determine if it should be negative based on the scheduling mode.'\
                     ' Only experiments run during discretionary time will have negative CPIDs.'
            raise ExperimentException(errmsg)

        self.__options = Options()      # Load the config, hardware, and restricted frequency data
        self.__cpid = cpid
        self.__scheduling_mode = 'unknown'
        self.__output_rx_rate = float(output_rx_rate)
        self.__comment_string = comment_string
        self.__slice_dict = {}
        self.__new_slice_id = 0

        if self.output_rx_rate > self.options.max_output_sample_rate:
            errmsg = f"Experiment's output sample rate is too high: {self.output_rx_rate} " \
                     f"greater than max {self.options.max_output_sample_rate}."
            raise ExperimentException(errmsg)

        self.__txrate = float(tx_bandwidth)  # sampling rate, samples per sec, Hz.
        self.__rxrate = float(rx_bandwidth)  # sampling rate for rx in samples per sec

        # Transmitting is possible in the range of txctrfreq +/- (txrate/2) because we have iq data
        # Receiving is possible in the range of rxctrfreq +/- (rxrate/2)
        if self.txrate > self.options.max_tx_sample_rate:
            errmsg = f"Experiment's transmit bandwidth is too large: {self.txrate} greater than " \
                     f"max {self.options.max_tx_sample_rate}."
            raise ExperimentException(errmsg)
        if self.rxrate > self.options.max_rx_sample_rate:
            errmsg = f"Experiment's receive bandwidth is too large: {self.rxrate} greater than " \
                     f"max {self.options.max_rx_sample_rate}."
            raise ExperimentException(errmsg)
        if round(self.options.usrp_master_clock_rate / self.txrate, 3) % 2.0 != 0.0:
            errmsg = f"Experiment's transmit bandwidth {self.txrate} is not possible as it must be an " \
                     f"integer divisor of USRP master clock rate {self.options.usrp_master_clock_rate}"
            raise ExperimentException(errmsg)
        if round(self.options.usrp_master_clock_rate / self.rxrate, 3) % 2.0 != 0.0:
            errmsg = f"Experiment's receive bandwidth {self.rxrate} is not possible as it must be an " \
                     f"integer divisor of USRP master clock rate {self.options.usrp_master_clock_rate}"
            raise ExperimentException(errmsg)

        # Note - txctrfreq and rxctrfreq are set here and modify the actual center frequency to a
        # multiple of the clock divider that is possible by the USRP - this default value set
        # here is not exact (center freq is never exactly 12 MHz).

        # convert from kHz to Hz to get correct clock divider. Return the result back in kHz.
        clock_multiples = self.options.usrp_master_clock_rate/2**32
        clock_divider = math.ceil(txctrfreq*1e3/clock_multiples)
        self.__txctrfreq = (clock_divider * clock_multiples)/1e3

        clock_divider = math.ceil(rxctrfreq*1e3/clock_multiples)
        self.__rxctrfreq = (clock_divider * clock_multiples)/1e3

        # This is experiment-wide transmit metadata necessary to build the pulses. This data
        # cannot change within the experiment and is used in the scan classes to pass information
        # to where the samples are built.
        self.__transmit_metadata = {
            'output_rx_rate':           self.output_rx_rate,
            'main_antennas':            self.options.main_antennas,
            'main_antenna_count':       self.options.main_antenna_count,
            'intf_antenna_count':       self.options.intf_antenna_count,
            'tr_window_time':           self.options.tr_window_time,
            'main_antenna_spacing':     self.options.main_antenna_spacing,
            'intf_antenna_spacing':     self.options.intf_antenna_spacing,
            'pulse_ramp_time':          self.options.pulse_ramp_time,
            'max_usrp_dac_amplitude':   self.options.max_usrp_dac_amplitude,
            'rx_sample_rate':           self.rxrate,
            'min_pulse_separation':     self.options.min_pulse_separation,
            'txctrfreq':                self.txctrfreq,
            'txrate':                   self.txrate,
            'intf_offset':              self.options.intf_offset
        }

        # Dictionary of how each exp_slice interacts with the other slices.
        # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.
        # The only interface options are those specified in interface_types.
        self.__interface = {}

        # The following are for internal use only, and should not be modified in the experimental
        # class, but will be modified by the class method build_scans. For this reason they
        # are private, with getters only, in case they are used for reference by the user.
        # These are used internally to build iterable objects out of the slice using the
        # interfacing specified.
        self.__scan_objects = []
        self.__scanbound = False
        self.__running_experiment = None  # this will be of ScanClassBase type

        # This is used for adding and editing slices
        self.__slice_restrictions = {
            'tx_bandwidth': self.tx_bandwidth,
            'rx_bandwidth': self.rx_bandwidth,
            'tx_minfreq': self.tx_minfreq,
            'tx_maxfreq': self.tx_maxfreq,
            'rx_minfreq': self.rx_minfreq,
            'rx_maxfreq': self.rx_maxfreq,
            'txctrfreq': self.txctrfreq,
            'rxctrfreq': self.rxctrfreq,
            'output_rx_rate': self.output_rx_rate,
        }

    __slice_keys = slice_key_set
    __hidden_slice_keys = hidden_key_set

    @property
    def cpid(self):
        """
        This experiment's CPID (control program ID, a term that comes from ROS).

        :returns:   cpid - read-only, only modified at runtime by set_scheduling_mode() to set to a
                    negative value during discretionary time
        :rtype:     int
        """
        return self.__cpid

    @property
    def experiment_name(self):
        """
        The experiment class name.

        :returns:   experiment_name
        :rtype:     str
        """
        return self.__experiment_name

    @property
    def output_rx_rate(self):
        """
        The output receive rate of the data, Hz.

        :returns:   output_rx_rate - read-only
        :rtype:     float
        """
        return self.__output_rx_rate

    @property
    def tx_bandwidth(self):
        """
        The transmission sample rate to the DAC (Hz), and the transmit bandwidth.

        :returns:   tx_bandwidth - read-only
        :rtype:     float
        """
        return self.__txrate

    @property
    def txrate(self):
        """
        The transmission sample rate to the DAC (Hz).

        :returns:   txrate - read-only
        :rtype:     float
        """
        return self.__txrate

    @property
    def rx_bandwidth(self):
        """
        The receive bandwidth for this experiment, in Hz.

        :returns:   rx_bandwidth - read-only
        :rtype:     float
        """
        return self.__rxrate

    @property
    def rxrate(self):
        """
        The receive bandwidth for this experiment, or the receive sampling rate (of I and Q samples)
        In Hz.

        :returns:   rxrate - read-only
        :rtype:     float
        """
        return self.__rxrate

    @property
    def comment_string(self):
        """
        A string related to the experiment, to be placed in the experiment's files.

        :returns:   comment_string - read-only
        :rtype:     str
        """
        return self.__comment_string

    @property
    def num_slices(self):
        """
        The number of slices currently in the experiment.

        Will change after methods add_slice or del_slice are called.

        :returns:   num_slices
        :rtype:     int
        """
        return len(self.__slice_dict)

    @property
    def slice_keys(self):
        """
        The list of slice keys available.

        This cannot be updated. These are the keys in the current ExperimentPrototype slice_keys
        dictionary (the parameters available for slices).

        :returns:   slice_keys
        :rtype:     frozenset
        """
        return self.__slice_keys

    @property
    def slice_dict(self):
        """
        The dictionary of slices.

        The slice dictionary can be updated in add_slice, edit_slice, and del_slice. The slice
        dictionary is a dictionary of dictionaries that looks like:

        { slice_id1 : {slice_key1 : x, slice_key2 : y, ...},
        slice_id2 : {slice_key1 : x, slice_key2 : y, ...},
        ...}

        :returns:   slice_dict
        :rtype:     dict
        """
        return self.__slice_dict

    @property
    def new_slice_id(self):
        """
        The next unique slice id that is available to this instance of the experiment.

        This gets incremented each time it is called to ensure it returns a unique ID each time.

        :returns:   new_slice_id
        :rtype:     int
        """
        self.__new_slice_id += 1
        return self.__new_slice_id - 1

    @property
    def slice_ids(self):
        """
        The list of slice ids that are currently available in this experiment.

        This can change when add_slice, edit_slice, and del_slice are called.

        :returns:   slice_ids
        :rtype:     list
        """
        return list(self.__slice_dict.keys())

    @property
    def options(self):
        """
        The config options for running this experiment.

        These cannot be set or removed, but are specified in the config.ini, hdw.dat, and
        restrict.dat files.

        :returns:   options
        :rtype:     :py:class:`Options`
        """
        return self.__options

    @property
    def transmit_metadata(self):
        """
        A dictionary of config options and experiment-set values that cannot change in the
        experiment, that will be used to build pulse sequences.

        :returns:   transmit_metadata
        :rtype:     dict
        """
        return self.__transmit_metadata

    @property
    def txctrfreq(self):
        """
        The transmission center frequency that USRP is tuned to (kHz).

        :returns:   txctrfreq
        :rtype:     float
        """
        return self.__txctrfreq

    @property
    def tx_maxfreq(self):
        """
        The maximum transmit frequency.

        This is the maximum tx frequency possible in this experiment (either maximum in our license
        or maximum given by the center frequency, and sampling rate). The maximum is slightly less
        than that allowed by the center frequency and txrate, to stay away from the edges of the
        possible transmission band where the signal is distorted.

        :returns:   tx_maxfreq
        :rtype:     float
        """
        max_freq = self.txctrfreq * 1000 + (self.txrate/2.0) - transition_bandwidth
        if max_freq < self.options.max_freq:
            return max_freq
        else:
            # TODO log warning that wave_freq should not exceed options.max_freq - ctrfreq
            #  (possible to transmit above licensed band)
            return self.options.max_freq

    @property
    def tx_minfreq(self):
        """
        The minimum transmit frequency.

        This is the minimum tx frequency possible in this experiment (either minimum in our license
        or minimum given by the center frequency and sampling rate). The minimum is slightly more
        than that allowed by the center frequency and txrate, to stay away from the edges of the
        possible transmission band where the signal is distorted.

        :returns:   tx_minfreq
        :rtype:     float
        """
        min_freq = self.txctrfreq * 1000 - (self.txrate/2.0) + transition_bandwidth
        if min_freq > self.options.min_freq:
            return min_freq
        else:
            # TODO log warning that wave_freq should not go below ctrfreq - options.minfreq
            #  (possible to transmit below licensed band)
            return self.options.min_freq

    @property
    def rxctrfreq(self):
        """
        The receive center frequency that USRP is tuned to (kHz).

        :returns:   rxctrfreq
        :rtype:     float
        """
        return self.__rxctrfreq

    @property
    def rx_maxfreq(self):
        """
        The maximum receive frequency.

        This is the maximum tx frequency possible in this experiment (maximum given by the center
        frequency and sampling rate), as license doesn't matter for receiving. The maximum is
        slightly less than that allowed by the center frequency and rxrate, to stay away from the
        edges of the possible receive band where the signal may be distorted.

        :returns:   rx_maxfreq
        :rtype:     float
        """
        max_freq = self.rxctrfreq * 1000 + (self.rxrate/2.0) - transition_bandwidth
        return max_freq

    @property
    def rx_minfreq(self):
        """
        The minimum receive frequency.

        This is the minimum rx frequency possible in this experiment (minimum given by the center
        frequency and sampling rate) - license doesn't restrict receiving. The minimum is
        slightly more than that allowed by the center frequency and rxrate, to stay away from the
        edges of the possible receive band where the signal may be distorted.

        :returns:   rx_minfreq
        :rtype:     float
        """
        min_freq = self.rxctrfreq * 1000 - (self.rxrate/2.0) + transition_bandwidth
        if min_freq > 1000:     # Hz
            return min_freq
        else:
            return 1000         # Hz

    @property
    def interface(self):
        """
        The dictionary of interfacing for the experiment slices.

        Interfacing should be set up for any slice when it gets added, ie. in add_slice,
        except for the first slice added. The dictionary of interfacing is setup as:
        [(slice_id1, slice_id2) : INTERFACING_TYPE,
        (slice_id1, slice_id3) : INTERFACING_TYPE,
        ...]
        for all current slice_ids.

        :returns:   interface
        :rtype:     dict
        """
        return self.__interface

    @property
    def scan_objects(self):
        """
        The list of instances of class Scan for use in radar_control.

        These cannot be modified by the user, but are created using the slice dictionary.

        :returns:   scan_objects
        :rtype:     list
        """
        return self.__scan_objects

    @property
    def scheduling_mode(self):
        """
        Return the scheduling mode time type that this experiment is running in. Types are listed in
        possible_scheduling_modes. Initialized to 'unknown' until set by the experiment handler.

        :returns:   scheduling_mode
        :rtype:     str
        """
        return self.__scheduling_mode

    def _set_scheduling_mode(self, scheduling_mode):
        """
        Set the scheduling mode if the provided mode is valid. Should only be called by the
        experiment handler after initializing the user's class.

        :param  scheduling_mode:    scheduling mode to be set
        :type   scheduling_mode:    str

        :raises ExperimentException: if scheduling mode not valid
        """
        if scheduling_mode in possible_scheduling_modes:
            self.__scheduling_mode = scheduling_mode
            if scheduling_mode == 'discretionary':
                self.__cpid = -1 * self.__cpid
        else:
            errmsg = f'Scheduling mode {scheduling_mode} set by experiment handler is not '\
                     f' a valid mode: {possible_scheduling_modes}'
            raise ExperimentException(errmsg)

    def slice_beam_directions_mapping(self, slice_id):
        """
        A mapping of the beam directions in the given slice id.

        :param      slice_id:   id of the slice to get beam directions for.
        :type       slice_id:   int

        :returns:   enumeration mapping dictionary of beam number to beam direction(s) in degrees off
                    boresight.
        :rtype:     dict
        """
        if slice_id not in self.slice_ids:
            return {}
        beam_directions = self.slice_dict[slice_id].beam_angle
        mapping = {}
        for beam_num, beam_dir in enumerate(beam_directions):
            mapping[beam_num] = beam_dir
        return mapping

    def check_new_slice_interfacing(self, interfacing_dict):
        """
        Checks that the new slice plays well with its siblings (has interfacing that is resolvable).
        If so, returns a new dictionary with all interfacing values set.

        The interfacing assumes that the interfacing_dict given by the user defines the closest
        interfacing of the new slice with a slice. For example, if the slice is to be 'CONCURRENT'
        combined with slice 0, the interfacing dict should provide this information. If only 'SCAN'
        interfacing with slice 1 is provided, then that will be assumed to be the closest and
        therefore the interfacing with slice 0 will also be 'SCAN'.

        If no interfacing_dict is provided for a slice, the default is to do 'SCAN' type interfacing
        for the new slice with all other slices.

        :param      interfacing_dict:   the user-provided interfacing dict, which may be empty or
                                        incomplete. If empty, all interfacing is assumed to be =
                                        'SCAN' type. If it contains something, we ensure that the
                                        interfacing provided makes sense with the values already
                                        known for its closest sibling.
        :type       interfacing_dict:   dict

        :returns:   full interfacing dictionary.
        :rtype:     dict

        :raises ExperimentException:    if invalid interface types provided or if interfacing can
                                        not be resolved.
        """
        for sibling_slice_id, interface_value in interfacing_dict.items():
            if interface_value not in interface_types:
                errmsg = f'Interface value with slice {sibling_slice_id}: {interface_value} not '\
                         f'valid. Types available are: {interface_types}'
                raise ExperimentException(errmsg)

        full_interfacing_dict = {}

        # if this is not the first slice we are setting up, set up interfacing.
        if len(self.slice_ids) != 0:
            if len(interfacing_dict.keys()) > 0:
                # the user provided some keys, so check that keys are valid.
                # To do this, get the closest interface type.
                # We assume that the user meant this to be the closest interfacing
                # for this slice.
                for sibling_slice_id in interfacing_dict.keys():
                    if sibling_slice_id not in self.slice_ids:
                        errmsg = f'Cannot add slice: the interfacing_dict set interfacing to an unknown '\
                                 f'slice {sibling_slice_id} not in slice ids {self.slice_ids}'
                        raise ExperimentException(errmsg)
                try:
                    closest_sibling = max(interfacing_dict.keys(),
                                          key=lambda k: interface_types.index(
                                               interfacing_dict[k]))
                except ValueError as e:  # cannot find interface type in list
                    errmsg = f'Interface types must be of valid types {interface_types}.'
                    raise ExperimentException(errmsg) from e
                closest_interface_value = interfacing_dict[closest_sibling]
                closest_interface_rank = interface_types.index(closest_interface_value)
            else:
                # the user provided no keys. The default is therefore 'SCAN'
                # with all keys so the closest will be 'SCAN' (the furthest possible interface_type)
                closest_sibling = self.slice_ids[0]
                closest_interface_value = 'SCAN'
                closest_interface_rank = interface_types.index(closest_interface_value)

            # now populate a full_interfacing_dict based on the closest sibling's interface values
            # and knowing how we interface with that sibling. this is the only correct interfacing
            # given the closest interfacing.
            full_interfacing_dict[closest_sibling] = closest_interface_value
            for sibling_slice_id, siblings_interface_value in self.get_slice_interfacing(closest_sibling).items():
                if interface_types.index(siblings_interface_value) >= closest_interface_rank:
                    # in this case, the interfacing between the sibling and the closest sibling is
                    # closer than the closest interface for the new slice. Therefore, interface with
                    # this sibling should be equal to the closest interface. Or, if they are all at
                    # the same rank, then the interfacing should equal that rank. For example,
                    # slices 0 and 1 combined CONCURRENT. New slice 2 is added with closest
                    # interfacing SEQUENCE to slice 0. Slice 2 will therefore also be interfaced
                    # with slice 1 as SEQUENCE type, since both slices 0 and 1 are in a single
                    # SEQUENCE.
                    full_interfacing_dict[sibling_slice_id] = closest_interface_value
                else:  # the rank is less than the closest rank.
                    # in this case, the interfacing to this sibling should be the same as the
                    # closest sibling interface to this sibling. For example, slices 0 and 1 are
                    # combined SCAN and slice 2 is combined AVEPERIOD with slice 0 (closest).
                    # Therefore slice 2 should be combined SCAN with slice 1 since 0 and 2 are now
                    # within the same scan.
                    full_interfacing_dict[sibling_slice_id] = siblings_interface_value

            # now check everything provided by the user with the correct full_interfacing_dict
            # that was populated based on the closest sibling given by the user.
            for sibling_slice_id, interface_value in interfacing_dict.items():
                if interface_value != full_interfacing_dict[sibling_slice_id]:
                    siblings_interface_value = self.get_slice_interfacing(closest_sibling)[sibling_slice_id]
                    errmsg = f'The interfacing values of new slice cannot be reconciled. Interfacing '\
                             f'with slice {closest_sibling}: {closest_interface_value} and with '\
                             f'slice {sibling_slice_id}: {interface_value} does not make sense with '\
                             f'existing interface between slices of '\
                             f'{([sibling_slice_id, closest_sibling].sort())}: {siblings_interface_value}'
                    raise ExperimentException(errmsg)

        return full_interfacing_dict

    def __update_slice_interfacing(self):
        """
        Internal slice interfacing updater. This should only be used internally when slice
        dictionary is changed, to update all of the slices' interfacing dictionaries.
        """
        for slice_id in self.slice_ids:
            self.__slice_dict[slice_id].slice_interfacing = self.get_slice_interfacing(slice_id)

    def add_slice(self, exp_slice, interfacing_dict=None):
        """
        Add a slice to the experiment.

        :param      exp_slice:          a slice (dictionary of slice_keys) to add to the experiment.
        :type       exp_slice:          dict
        :param      interfacing_dict:   dictionary of type {slice_id : INTERFACING , ... } that
                                        defines how this slice interacts with all the other slices
                                        currently in the experiment.
        :type       interfacing_dict:   dict

        :returns:   the slice_id of the new slice that was just added.
        :rtype:     int

        :raises ExperimentException:    if slice is not a dictionary or if there are errors in
                                        setup_slice.
        """
        if not isinstance(exp_slice, dict):
            errmsg = f'Attempt to add a slice failed - {exp_slice} is not a dictionary of slice parameters'
            raise ExperimentException(errmsg)
            # TODO multiple types of Exceptions so they can be caught by the experiment in these
            #  add_slice, edit_slice, del_slice functions (and handled specifically)
        if interfacing_dict is None:
            interfacing_dict = {}

        add_slice_id = exp_slice['slice_id'] = self.new_slice_id
        # each added slice has a unique slice id, even if previous slices have been deleted.
        exp_slice['cpid'] = self.cpid

        # Now we setup the slice which will check minimum requirements and set defaults, and then
        # will complete a check_slice and raise any errors found.
        new_exp_slice = ExperimentSlice(exp_slice, self.options, **self.__slice_restrictions)

        # now check that the interfacing values make sense before appending.
        full_interfacing_dict = self.check_new_slice_interfacing(interfacing_dict)
        for sibling_slice_id, interface_value in full_interfacing_dict.items():
            # sibling_slice_id < new slice id so this maintains interface list requirement.
            self.__interface[(sibling_slice_id, exp_slice['slice_id'])] = interface_value

        # if there were no errors raised in setup_slice, we will add the slice to the slice_dict.
        self.__slice_dict[add_slice_id] = new_exp_slice

        # reset all slice_interfacing since a slice has been added.
        self.__update_slice_interfacing()

        return add_slice_id

    def del_slice(self, remove_slice_id):
        """
        Remove a slice from the experiment.

        :param      remove_slice_id:    the id of the slice you'd like to remove.
        :type       remove_slice_id:    int

        :returns:   a copy of the removed slice.
        :rtype:     dict

        :raises ExperimentException:    if remove_slice_id does not exist in the slice dictionary.
        """
        try:
            removed_slice = copy.deepcopy(self.slice_dict[remove_slice_id])
            del(self.slice_dict[remove_slice_id])
        except (KeyError, TypeError) as e:
            errmsg = f'Cannot remove slice id {remove_slice_id} : it does not exist in slice '\
                      'dictionary'
            raise ExperimentException(errmsg) from e

        remove_keys = []
        for key1, key2 in self.__interface.keys():
            if key1 == remove_slice_id or key2 == remove_slice_id:
                remove_keys.append((key1, key2))

        for keyset in remove_keys:
            del self.__interface[keyset]

        # reset all slice_interfacing since a slice has been removed.
        self.__update_slice_interfacing()

        return removed_slice

    def edit_slice(self, edit_slice_id, **kwargs):
        """
        Edit a slice.

        A quick way to edit a slice. In reality this is actually adding a new slice and deleting the
        old one. Useful for quick changes. Note that using this function will remove the slice_id
        that you are changing and will give it a new id. It will account for this in the interfacing
        dictionary.

        :param      edit_slice_id:  the slice id of the slice to be edited.
        :type       edit_slice_id:  int
        :param      kwargs:         slice parameter to slice values that you want to change.
        :type       kwargs:         dict

        :returns:   the new slice id of the edited slice, or the edit_slice_id if no change has
                    occurred due to failure of new slice parameters to pass experiment checks.
        :rtype:     int

        :raises ExperimentException:    if the edit_slice_id does not exist in slice dictionary or
                                        the params or values do not make sense.
        """
        slice_params_to_edit = dict(kwargs)

        try:
            edited_slice = self.slice_dict[edit_slice_id].copy()
        except (KeyError, TypeError):
            # the edit_slice_id is not an index in the slice_dict
            errmsg = f'Trying to edit {edit_slice_id} but it does not exist in Slice_IDs list.'
            raise ExperimentException(errmsg)

        for edit_slice_param, edit_slice_value in slice_params_to_edit.items():
            if edit_slice_param in self.slice_keys:
                setattr(edited_slice, edit_slice_param, edit_slice_value)
            else:
                errmsg = f'Cannot edit slice ID {edit_slice_id}: {edit_slice_param} is not a valid'\
                          ' slice parameter'
                raise ExperimentException(errmsg)

        # Get the interface values of the slice. These are not editable, if these are wished to be
        # changed add_slice must be used explicitly to interface a new slice.
        interface_values = self.get_slice_interfacing(edit_slice_id)

        removed_slice = self.del_slice(edit_slice_id)

        try:
            # checks are done on interfacing when slice is added.
            # interfacing between existing slice_ids cannot be changed after addition.
            new_slice_id = self.add_slice(edited_slice, interface_values)
            return new_slice_id

        except ExperimentException:
            # if any failure occurs when checking the slice, the slice has not been added to the
            # slice dictionary so we will revert to old slice
            self.__slice_dict[edit_slice_id] = removed_slice

            for key1, key1_interface in interface_values.items():
                if key1 < edit_slice_id:
                    self.__interface[(key1, edit_slice_id)] = key1_interface
                else:
                    self.__interface[(edit_slice_id, key1)] = key1_interface

            # reset all slice_interfacing back
            self.__update_slice_interfacing()

            # TODO: Log that this operation failed

            return edit_slice_id

    def __repr__(self):
        represent = f'self.cpid = {self.cpid}\n'\
                    f'self.num_slices = {self.num_slices}\n'\
                    f'self.slice_ids = {self.slice_ids}\n'\
                    f'self.slice_keys = {self.slice_keys}\n'\
                    f'self.options = {self.options.__str__()}\n'\
                    f'self.txctrfreq = {self.txctrfreq}\n'\
                    f'self.txrate = {self.txrate}\n'\
                    f'self.rxctrfreq = {self.rxctrfreq}\n'\
                    f'self.slice_dict = {self.slice_dict}\n'\
                    f'self.interface = {self.interface}\n'
        return represent

    def build_scans(self):
        """
        Build the scan information, which means creating the Scan, AveragingPeriod, and Sequence
        instances needed to run this experiment.

        Will be run by experiment handler, to build iterable objects for radar_control to use.
        Creates scan_objects in the experiment for identifying which slices are in the scans.
        """
        # Check interfacing and other experiment-wide settings.
        self.self_check()

        # TODO: investigating how I might go about using this base class - maybe make a new IterableExperiment class
        #  to inherit

        # TODO consider removing scan_objects from init and making a new Experiment class to inherit
        # from ScanClassBase and having all of this included in there. Then would only need to
        # pass the running experiment to the radar control (would be returned from build_scans)
        self.__running_experiment = ScanClassBase(self.slice_ids, self.slice_dict, self.interface,
                                                  self.transmit_metadata)

        self.__running_experiment.nested_slice_list = self.get_scan_slice_ids()

        self.__scan_objects = []
        for params in self.__running_experiment.prep_for_nested_scan_class():
            self.__scan_objects.append(Scan(*params))
        
        for scan in self.__scan_objects:
            if scan.scanbound is not None:
                self.__scanbound = True

        if self.__scanbound:
            try:
                self.__scan_objects = sorted(self.__scan_objects, key=lambda input_scan: input_scan.scanbound[0])
            except (IndexError, TypeError) as e:  # scanbound is None in some scans
                errmsg = 'If one slice has a scanbound, they all must to avoid up to minute-long downtimes.'
                raise ExperimentException(errmsg) from e

        max_num_concurrent_slices = 0
        for scan in self.__scan_objects:
            for aveperiod in scan.aveperiods:
                for seq in aveperiod.sequences:
                    if len(seq.slice_ids) > max_num_concurrent_slices:
                        max_num_concurrent_slices = len(seq.slice_ids)

        # TODO: Log appropriately
        print(f"Number of Scan types: {len(self.__scan_objects)}")
        print(f"Number of AveragingPeriods in Scan #1: {len(self.__scan_objects[0].aveperiods)}")
        print(f"Number of Sequences in Scan #1, Averaging Period #1: "
              f"{len(self.__scan_objects[0].aveperiods[0].sequences)}")
        print("Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1:"
              f" {len(self.__scan_objects[0].aveperiods[0].sequences[0].slice_dict)}")
        print(f"Max concurrent slices: {max_num_concurrent_slices}")

    def get_scan_slice_ids(self):
        """
        Organize the slice_ids by scan.

        Take my own interfacing and get info on how many scans and which slices make which scans.
        Return a list of lists where each inner list contains the slices that are in an
        averagingperiod that is inside this scan. ie. len(nested_slice_list) = # of averagingperiods
        in this scan, len(nested_slice_list[0]) = # of slices in the first averagingperiod, etc.

        :returns:   A list that has one element per scan. Each element is a list of slice_ids
                    signifying which slices are combined inside that scan. The list returned could
                    be of length 1, meaning only one scan is present in the experiment.
        :rtype:     list of lists
        """
        # TODO add this to ScanClassBase method by just passing in the current type (Experiment,
        # Scan, AvePeriod) which would allow you to determine which interfacing to pull out.
        scan_combos = []

        for k, interface_value in self.interface.items():
            if interface_value != "SCAN":
                scan_combos.append(list(k))

        combos = self.__running_experiment.slice_combos_sorter(scan_combos, self.slice_ids)

        return combos

    def get_slice_interfacing(self, slice_id):
        """
        Check the experiment's interfacing dictionary for all interfacing that pertains to a given
        slice, and return the interfacing information in a dictionary.

        :param      slice_id:   Slice ID to search the interface dictionary for.
        :type       slice_id:   int

        :returns:   interfacing dictionary for the slice.
        :rtype:     dict
        """
        slice_interface = {}
        for keys, interfacing_type in self.interface.items():
            num1 = keys[0]
            num2 = keys[1]
            if num1 == slice_id:
                slice_interface[num2] = interfacing_type
            elif num2 == slice_id:
                slice_interface[num1] = interfacing_type

        return slice_interface

    def self_check(self):
        """
        Check that the values in this experiment are valid. Checks all slices.

        :raises ExperimentException:    if any self check errors occur
        """
        if self.num_slices < 1:
            errmsg = "Invalid num_slices less than 1"
            raise ExperimentException(errmsg)

        # TODO: check if self.cpid is not unique - incorporate known cpids from git repo
        # TODO: use pygit2 for this

        # run check_slice on all slices. Check_slice is a full check and can be done on a slice at
        # any time after setup. We run it now in case the user has changed something
        # inappropriately (ie, any way other than using edit_slice, add_slice, or del_slice).
        # "Private" instance variables with leading underscores are not actually private in
        # python they just have a bit of a mangled name so they are not readily available but give
        # the user notice that they should be left alone. If the __slice_dict has been changed
        # improperly, we should check it for problems here.
        for a_slice in self.slice_ids:
            selferrs = self.slice_dict[a_slice].check_slice()
            if not selferrs:
                continue        # If returned error dictionary is empty
            errmsg = f"Self Check Errors Occurred with slice Number : {a_slice} \nSelf Check Errors are : {selferrs}"
            raise ExperimentException(errmsg)

        print("No Self Check Errors. Continuing...")    # TODO: Log this


class ExperimentSlice:
    """
    These are the keys that are set by the user when initializing a slice. Some are required, some can
    be defaulted, and some are set by the experiment and are read-only.

    **Slice Keys Required by the User**

    beam_angle *required*
        list of beam directions, in degrees off azimuth. Positive is E of N. The beam_angle list length
        = number of beams. Traditionally beams have been 3.24 degrees separated but we don't refer to
        them as beam -19.64 degrees, we refer as beam 1, beam 2. Beam 0 will be the 0th element in the
        list, beam 1 will be the 1st, etc. These beam numbers are needed to write the [rx|tx]_beam_order
        list. This is like a mapping of beam number (list index) to beam direction off boresight.
    clrfrqrange *required or freq required*
        range for clear frequency search, should be a list of length = 2, [min_freq, max_freq] in kHz.
        **Not currently supported.**
    first_range *required*
        first range gate, in km
    freq *required or clrfrqrange required*
        transmit/receive frequency, in kHz. Note if you specify clrfrqrange it won't be used.
    intt *required or intn required*
        duration of an averaging period (integration), in ms. (maximum)
    intn *required or intt required*
        number of averages to make a single averaging period (integration), only used if intt = None
    num_ranges *required*
        Number of range gates to receive for. Range gate time is equal to pulse_len and range gate
        distance is the range_sep, calculated from pulse_len.
    pulse_len *required*
        length of pulse in us. Range gate size is also determined by this.
    pulse_sequence *required*
        The pulse sequence timing, given in quantities of tau_spacing, for example normalscan = [0, 14,
        22, 24, 27, 31, 42, 43].
    rx_beam_order *required*
        beam numbers written in order of preference, one element in this list corresponds to one
        averaging period. Can have lists within the list, resulting in multiple beams running
        simultaneously in the averaging period, so imaging. A beam number of 0 in this list gives us the
        direction of the 0th element in the beam_angle list. It is up to the writer to ensure their beam
        pattern makes sense. Typically rx_beam_order is just in order (scanning W to E or E to W, ie.
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]. You can list numbers multiple times in
        the rx_beam_order list, for example [0, 1, 1, 2, 1] or use multiple beam numbers in a single
        averaging period (example [[0, 1], [3, 4]], which would trigger an imaging integration. When we
        do imaging we will still have to quantize the directions we are looking in to certain beam
        directions. It is up to the user to ensure that this field works well with the specified
        tx_beam_order or tx_antenna_pattern.
    tau_spacing *required*
        multi-pulse increment (mpinc) in us, Defines minimum space between pulses.

    **Defaultable Slice Keys**

    acf *defaults*
        flag for rawacf generation. The default is False. If True, the following fields are also used: -
        averaging_method (default 'mean') - xcf (default True if acf is True) - acfint (default True if
        acf is True) - lagtable (default built based on all possible pulse combos) - range_sep (will be
        built by pulse_len to verify any provided value)
    acfint *defaults*
        flag for interferometer autocorrelation data. The default is True if acf is True, otherwise
        False.
    align_sequences *defaults*
        flag for aligning the start of the first pulse in each sequence to tenths of a second. Default
        False.
    averaging_method *defaults*
        a string defining the type of averaging to be done. Current methods are 'mean' or 'median'. The
        default is 'mean'.
    comment *defaults*
        a comment string that will be placed in the borealis files describing the slice. Defaults to
        empty string.
    lag_table *defaults*
        used in acf calculations. It is a list of lags. Example of a lag: [24, 27] from 8-pulse
        normalscan. This defaults to a lagtable built by the pulse sequence provided. All combinations
        of pulses will be calculated, with both the first pulses and last pulses used for lag-0.
    pulse_phase_offset *defaults*
        a handle to a function that will be used to generate one phase per each pulse in the sequence.
        If a function is supplied, the beam iterator, sequence number, and number of pulses in the
        sequence are passed as arguments that can be used in this function. The default is None if no
        function handle is supplied.

        encode_fn(beam_iter, sequence_num, num_pulses):
            return np.ones(size=(num_pulses))

        The return value must be numpy array of num_pulses in size. The result is a single phase shift
        for each pulse, in degrees.

        Result is expected to be real and in degrees and will be converted to complex radians.
    range_sep *defaults*
        a calculated value from pulse_len. If already set, it will be overwritten to be the correct
        value determined by the pulse_len. Used for acfs. This is the range gate separation, in the
        radial direction (away from the radar), in km.
    rx_int_antennas *defaults*
        The antennas to receive on in interferometer array, default is all antennas given max number
        from config.
    rx_main_antennas *defaults*
        The antennas to receive on in main array, default is all antennas given max number from config.
    scanbound *defaults*
        A list of seconds past the minute for averaging periods in a scan to align to. Defaults to None,
        not required. If one slice in an experiment has a scanbound, they all must.
    seqoffset *defaults*
        offset in us that this slice's sequence will begin at, after the start of the sequence. This is
        intended for CONCURRENT interfacing, when you want multiple slice's pulses in one sequence you
        can offset one slice's sequence from the other by a certain time value so as to not run both
        frequencies in the same pulse, etc. Default is 0 offset.
    tx_antennas *defaults*
        The antennas to transmit on, default is all main antennas given max number from config.
    tx_antenna_pattern *defaults*
        experiment-defined function which returns a complex weighting factor of magnitude <= 1 for each
        tx antenna used in the experiment. The return value of the function must be an array of size
        [num_beams, main_antenna_count] with all elements having magnitude <= 1. This function is
        analogous to the beam_angle field in that it defines the transmission pattern for the array, and
        the tx_beam_order field specifies which "beam" to use in a given averaging period.
    tx_beam_order *defaults, but required if tx_antenna_pattern given*
        beam numbers written in order of preference, one element in this list corresponds to one
        averaging period. A beam number of 0 in this list gives us the direction of the 0th element in
        the beam_angle list. It is up to the writer to ensure their beam pattern makes sense. Typically
        tx_beam_order is just in order (scanning W to E or E to W, i.e. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15]. You can list numbers multiple times in the tx_beam_order list, for
        example [0, 1, 1, 2, 1], but unlike rx_beam_order, you CANNOT use multiple beam numbers in a
        single averaging period. In other words, this field MUST be a list of integers, as opposed to
        rx_beam_order, which can be a list of lists of integers. The length of this list must be equal
        to the length of the rx_beam_order list. If tx_antenna_pattern is given, the items in
        tx_beam_order specify which row of the return from tx_antenna_pattern to use to beamform a given
        transmission. Default is None, i.e. rx_only slice.
    wait_for_first_scanbound *defaults*
        A boolean flag to determine when an experiment starts running. True (default) means an
        experiment will wait until the first averaging period in a scan to start transmitting. False
        means an experiment will not wait for the first averaging period, but will instead start
        transmitting at the nearest averaging period. Note: for multi-slice experiments, the first slice
        is the only one impacted by this parameter.
    xcf *defaults*
        flag for cross-correlation data. The default is True if acf is True, otherwise False.

    **Read-only Slice Keys**

    clrfrqflag *read-only*
        A boolean flag to indicate that a clear frequency search will be done. **Not currently
        supported.**
    cpid *read-only*
        The ID of the experiment, consistent with existing radar control programs. This is actually an
        experiment-wide attribute but is stored within the slice as well. This is provided by the user
        but not within the slice, instead when the experiment is initialized.
    rx_only *read-only*
        A boolean flag to indicate that the slice doesn't transmit, only receives.
    slice_id *read-only*
        The ID of this slice object. An experiment can have multiple slices. This is not set by the user
        but instead set by the experiment when the slice is added. Each slice id within an experiment is
        unique. When experiments start, the first slice_id will be 0 and incremented from there.
    slice_interfacing *read-only*
        A dictionary of slice_id : interface_type for each sibling slice in the experiment at any given
        time.
    """
    def __init__(self, exp_slice_dict: dict, options: Options, tx_minfreq: float, tx_maxfreq: float, rx_minfreq: float,
                 rx_maxfreq: float, txctrfreq: float, rxctrfreq: float, tx_bandwidth: float, rx_bandwidth: float,
                 output_rx_rate: float):
        # These fields can be specified in exp_slice_dict, subject to some conditions.
        self.acf: bool = False
        self.acfint: bool = False
        self.align_sequences: bool = False
        self.averaging_method: str = 'mean'
        self.beam_angle = None
        self.clrfrqrange = None
        self.comment: str = ''
        self.cpid = None
        self.decimation_scheme = None
        self.first_range = None
        self.freq = None
        self.intn = None
        self.intt = None
        self.lag_table = None
        self.num_ranges = None
        self.pulse_len = None
        self.pulse_phase_offset = None
        self.pulse_sequence = None
        self.range_sep = None
        self.rx_beam_order = None
        self.rx_int_antennas = None
        self.rx_main_antennas = None
        self.scanbound = None
        self.seqoffset = None
        self.slice_id = None
        self.tau_spacing = None
        self.tx_antennas = None
        self.tx_antenna_pattern = None
        self.tx_beam_order = None
        self.wait_for_first_scanbound: bool = False
        self.xcf: bool = False

        # These fields are for checking the validity of the user-specified fields above, to ensure the slice is
        # compatible with the experiment settings.
        self.__tx_minfreq = tx_minfreq
        self.__tx_maxfreq = tx_maxfreq
        self.__rx_minfreq = rx_minfreq
        self.__rx_maxfreq = rx_maxfreq
        self.__txctrfreq = txctrfreq
        self.__rxctrfreq = rxctrfreq
        self.__tx_bandwidth = tx_bandwidth
        self.__rx_bandwidth = rx_bandwidth
        self.__output_rx_rate = output_rx_rate
        self.__options = options

        # Read-only fields that are set based on the user-specified fields.
        self.clrfrqflag: bool = False
        self.cpid: int
        self.rxonly: bool = False
        self.slice_id: int
        self.slice_interfacing: dict

        # put all values from the slice dictionary into attributes with the same names
        for k, v in exp_slice_dict.items():
            if k not in slice_key_set:
                raise ExperimentException(f'Invalid slice parameter {k}')
            setattr(self, k, v)

        # Default all non-specified fields as None.
        for k in slice_key_set:
            if hasattr(self, k):
                continue    # Attribute was present in exp_slice_dict
            else:
                setattr(self, k, None)  # Otherwise, set to None

        self.set_slice_identifiers()
        self.check_slice_specific_requirements()
        self.check_slice_minimum_requirements()
        self.set_slice_defaults()

        errors = self.check_slice()
        if errors:
            raise ExperimentException(errors)

    def check_slice_minimum_requirements(self):
        """
        Check the required slice keys.

        Check for the minimum requirements of the slice. The following keys are always required:
        "pulse_sequence", "tau_spacing", "pulse_len", "num_ranges", "first_range", (one of "intt" or
        "intn"), and "rx_beam_order". This function may modify the values in this slice dictionary
        to ensure that it is able to run and that the values make sense.

        :raises ExperimentException:    if any slice keys are invalid or missing
        """
        # TODO: add checks for values that make sense, not just check for types
        # TODO: make lists of operations to run and use if any() to shorten up this code!
        if self.pulse_sequence is None:
            errmsg = "Slice must specify pulse_sequence that must be a list of integers."
            raise ExperimentException(errmsg, self)
        elif not isinstance(self.pulse_sequence, list):
            errmsg = "Slice must specify pulse_sequence that must be a list of integers"
            raise ExperimentException(errmsg, self)
        for element in self.pulse_sequence:
            if not isinstance(element, int):
                errmsg = "Slice must specify pulse_sequence that must be a list of integers"
                raise ExperimentException(errmsg, self)

        if self.tau_spacing is None or not isinstance(self.tau_spacing, int):
            errmsg = "Slice must specify tau_spacing in us that must be an integer"
            raise ExperimentException(errmsg, self)

        # TODO may want to add a field for range_gate which could set this param.
        if not isinstance(self.pulse_len, int):
            errmsg = "Slice must specify pulse_len in us that must be an integer"
            raise ExperimentException(errmsg, self)

        if not isinstance(self.num_ranges, int):
            errmsg = "Slice must specify num_ranges that must be an integer"
            raise ExperimentException(errmsg, self)

        if self.first_range is None:
            errmsg = "Slice must specify first_range in km that must be a number"
            raise ExperimentException(errmsg, self)
        if not isinstance(self.first_range, (int, float)):
            errmsg = "Slice must specify first_range in km that must be a number"
            raise ExperimentException(errmsg, self)

        if self.intt is None:
            if self.intn is None:
                errmsg = f"Slice must specify either an intn (unitless) or intt in ms. Slice: {self}"
                raise ExperimentException(errmsg)
            elif not isinstance(self.intn, int):
                errmsg = f"intn must be an integer. Slice: {self}"
                raise ExperimentException(errmsg)
        else:
            if not isinstance(self.intt, (int, float)):
                errmsg = f"intt must be a number. Slice: {self}"
                raise ExperimentException(errmsg)
            else:
                if self.intn is None:
                    print('intn is set in experiment slice but will not be used due to intt')
                    # TODO Log warning intn will not be used
            self.intt = float(self.intt)

        # Check the validity of 'beam_angle' specified
        if self.beam_angle is None:
            errmsg = "Slice must specify beam_angle that must be a list of numbers (ints or floats)"\
                     f" which are angles of degrees off boresight (positive E of N). Slice: {self}"
            raise ExperimentException(errmsg)
        if not isinstance(self.beam_angle, list):
            errmsg = "Slice must specify beam_angle that must be a list of numbers (ints or floats) " \
                     f"which are angles of degrees off boresight (positive E of N). Slice: {self}"
            raise ExperimentException(errmsg)
        for element in self.beam_angle:
            if not isinstance(element, (int, float)):
                errmsg = "Slice must specify beam_angle that must be a list of numbers (ints or floats)" \
                         f" which are angles of degrees off boresight (positive E of N). Slice: {self}"
                raise ExperimentException(errmsg)

        # Check the validity of 'rx_beam_order' specified
        if self.rx_beam_order is None:
            errmsg = "Slice must specify rx_beam_order that must be a list of ints or lists (of ints)" \
                     f" corresponding to the order of the angles in the beam_angle list. Slice: {self}"
            raise ExperimentException(errmsg)
        if not isinstance(self.rx_beam_order, list):
            errmsg = "Slice must specify rx_beam_order that must be a list of ints or lists (of ints)" \
                     f" corresponding to the order of the angles in the beam_angle list. Slice: {self}"
            raise ExperimentException(errmsg)
        for element in self.rx_beam_order:
            if not isinstance(element, (int, list)):
                errmsg = "Slice must specify rx_beam_order that must be a list of ints or lists (of ints)" \
                         f" corresponding to the order of the angles in the beam_angle list. Slice: {self}"
                raise ExperimentException(errmsg)
            if isinstance(element, list):
                for beamnum in element:
                    if not isinstance(beamnum, int):
                        errmsg = "Slice must specify rx_beam_order that must be a list of ints " \
                                 "or lists (of ints) corresponding to the order of the angles " \
                                 f"in the beam_angle list. Slice: {self}"
                        raise ExperimentException(errmsg)
                    if beamnum >= len(self.beam_angle):
                        errmsg = f"Beam number {beamnum} could not index in beam_angle list of " \
                                 f"length {len(self.beam_angle)}. Slice: {self}"
                        raise ExperimentException(errmsg)
            else:
                if element >= len(self.beam_angle):
                    errmsg = f"Beam number {element} could not index in beam_angle list of length " \
                             f"{len(self.beam_angle)}. Slice: {self}"
                    raise ExperimentException(errmsg)

        if self.tx_beam_order is None:
            if not isinstance(self.tx_beam_order, list):
                errmsg = "tx_beam_order must be a list of ints corresponding to the order of the angles in " \
                         "the beam_angle list or an array of phases in the tx_antenna_pattern return. " \
                         f"Slice: {self}"
                raise ExperimentException(errmsg)
            if len(self.tx_beam_order) != len(self.rx_beam_order):
                errmsg = f"tx_beam_order does not have same length as rx_beam_order. Slice: {self}"
                raise ExperimentException(errmsg)
            for element in self.tx_beam_order:
                if not isinstance(element, int):
                    errmsg = "tx_beam_order must be a list of ints corresponding to the order of the angles in " \
                             "the beam_angle list or an array of phases in the tx_antenna_pattern return. " \
                             f"Slice: {self}"
                    raise ExperimentException(errmsg)
                if element >= len(self.beam_angle) and self.tx_antenna_pattern is None:
                    errmsg = f"Beam number {element} in tx_beam_order could not index in beam_angle " \
                             f"list of length {len(self.beam_angle)}. Slice: {self}"
                    raise ExperimentException(errmsg)

        if self.tx_antenna_pattern is not None and self.tx_beam_order is None:
            errmsg = f"tx_beam_order must be specified if tx_antenna_pattern specified. Slice {self}"
            raise ExperimentException(errmsg)

    def set_slice_identifiers(self):
        """
        Set the hidden slice keys to determine how to run the slice.

        This function sets up internal identifier flags 'clrfrqflag' and 'rxonly' in the slice so
        that we know how to properly set up the slice and know which keys in the slice must be
        specified and which are unnecessary. If these keys are ever written by the user, they will
        be rewritten here.

        :raises ExperimentException:    if clrfrqrange or freq not specified in slice
        """
        if self.clrfrqrange is not None:
            self.clrfrqflag = True
            self.rxonly = False

            if self.freq is not None and self.freq not in range(self.clrfrqrange[0], self.clrfrqrange[1]):
                # TODO: Log this appropriately
                print("Slice parameter 'freq' removed as 'clrfrqrange' takes precedence. If this is not desired,"
                      "remove 'clrfrqrange' parameter from experiment.")

        elif self.freq is not None:
            self.clrfrqflag = False
            if self.tx_beam_order is None:
                self.rxonly = True
            else:
                self.rxonly = False

        else:
            errmsg = 'A freq or clrfrqrange must be specified in a slice'
            raise ExperimentException(errmsg, self)

    def check_slice_specific_requirements(self):
        """
        Set the specific slice requirements depending.

        Check the requirements for the specific slice type as identified by the identifiers rxonly
        and clrfrqflag. The keys that need to be checked depending on these identifiers are "freq"
        and "clrfrqrange". This function may modify these keys.

        :raises ExperimentException:    if slice's specified frequency is invalid
        """
        if self.clrfrqflag:  # TX and RX mode with clear frequency search.
            # In this mode, clrfrqrange is required along with the other requirements.
            if not isinstance(self.clrfrqrange, list):
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)
            if len(self.clrfrqrange) != 2:
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)
            if not isinstance(self.clrfrqrange[0], int) or not isinstance(self.clrfrqrange[1], int):
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)

            errmsg = f"clrfrqrange must be between min and max tx frequencies " \
                     f"{(self.__tx_minfreq, self.__tx_maxfreq)} and rx frequencies " \
                     f"{(self.__rx_minfreq, self.__rx_maxfreq)} according to license and/or center frequencies / " \
                     f"sampling rates / transition bands, and must have lower frequency first."
            if self.clrfrqrange[0] >= self.clrfrqrange[1]:
                raise ExperimentException(errmsg)
            if (self.clrfrqrange[1] * 1000) >= self.__tx_maxfreq or (self.clrfrqrange[1] * 1000) >= self.__rx_maxfreq:
                raise ExperimentException(errmsg)
            if (self.clrfrqrange[0] * 1000) <= self.__tx_minfreq or (self.clrfrqrange[0] * 1000) <= self.__rx_minfreq:
                raise ExperimentException(errmsg)

            still_checking = True
            while still_checking:
                for freq_range in self.__options.restricted_ranges:
                    if freq_range[0] <= self.clrfrqrange[0] <= freq_range[1]:
                        if freq_range[0] <= self.clrfrqrange[1] <= freq_range[1]:
                            # the range is entirely within the restricted range.
                            raise ExperimentException(f'clrfrqrange is entirely within restricted range {freq_range}')
                        else:
                            print('Clrfrqrange will be modified because it is partially in a restricted range.')
                            # TODO Log warning, changing clrfrqrange because lower portion is in a restricted
                            #  frequency range.
                            self.clrfrqrange[0] = freq_range[1] + 1
                            # outside of restricted range now.
                            break  # we have changed the 'clrfrqrange' - must restart the
                            # check in case it's in another range.
                    else:
                        # lower end is not in restricted frequency range.
                        if freq_range[0] <= self.clrfrqrange[1] <= freq_range[1]:
                            print('Clrfrqrange will be modified because it is partially in a restricted range.')
                            # TODO Log warning, changing clrfrqrange because upper portion is in a
                            #  restricted frequency range.
                            self.clrfrqrange[1] = freq_range[0] - 1
                            # outside of restricted range now.
                            break  # we have changed the 'clrfrqrange' - must restart the for loop
                            # checking in case it's in another range.
                        else:  # neither end of clrfrqrange is inside the restricted range but
                            # we should check if the range is inside the clrfrqrange.
                            if self.clrfrqrange[0] <= freq_range[0] <= self.clrfrqrange[1]:
                                print('There is a restricted range within the clrfrqrange - '
                                      'STOP.')
                                # TODO Log a warning that there is a restricted range in the middle
                                #  of the clrfrqrange that will be avoided OR could make this an
                                #  Error. Still need to implement clear frequency searching.
                else:  # no break, so no changes to the clrfrqrange
                    still_checking = False

        elif self.rxonly:  # RX only mode.
            # In this mode, freq is required.
            freq_error = False
            if not isinstance(self.freq, int) and not isinstance(self.freq, float):
                freq_error = True
            elif (self.freq * 1000) >= self.__rx_maxfreq or (self.freq * 1000) <= self.__rx_minfreq:
                freq_error = True

            if freq_error:
                errmsg = "freq must be a number (kHz) between rx min and max frequencies "\
                        f"{(self.__rx_minfreq/1.0e3, self.__rx_maxfreq/1.0e3)} for the radar license "\
                        f"and be within range given center frequency {self.__rxctrfreq} kHz, " \
                        f"sampling rate {self.__rx_bandwidth/1.0e3} kHz, and transition band " \
                        f"{transition_bandwidth/1.0e3} kHz."
                raise ExperimentException(errmsg)

        else:  # TX-specific mode , without a clear frequency search.
            # In this mode, freq is required along with the other requirements.
            freq_error = False
            if not isinstance(self.freq, int) and not isinstance(self.freq, float):
                freq_error = True
            elif (self.freq * 1000) >= self.__tx_maxfreq or (self.freq * 1000) >= self.__rx_maxfreq:
                freq_error = True
            elif (self.freq * 1000) <= self.__tx_minfreq or (self.freq * 1000) <= self.__rx_minfreq:
                freq_error = True
            
            if freq_error:
                errmsg = "freq must be a number (kHz) between tx min and max frequencies "\
                        f"{(self.__tx_minfreq/1.0e3, self.__tx_maxfreq/1.0e3)} and rx min and max "\
                        f"frequencies {(self.__rx_minfreq/1.0e3, self.__rx_maxfreq/1.0e3)} for the "\
                        f"radar license and be within range given center frequencies "\
                        f"(tx: {self.__txctrfreq} kHz, rx: {self.__rxctrfreq} kHz), sampling rates "\
                        f"(tx: {self.__tx_bandwidth/1.0e3} kHz, rx: {self.__rx_bandwidth/1.0e3} kHz), "\
                        f"and transition band ({transition_bandwidth/1.0e3} kHz)."
                raise ExperimentException(errmsg)

            for freq_range in self.__options.restricted_ranges:
                if freq_range[0] <= self.freq <= freq_range[1]:
                    errmsg = f"freq is within a restricted frequency range {freq_range}"
                    raise ExperimentException(errmsg)

    def set_slice_defaults(self):
        """
        Set up defaults in case of some parameters being left blank.

        :raises ExperimentException:    if any slice parameters are invalid
        """
        default_slice_values = {
            'acf': False,
            'acfint': False,
            'align_sequences': False,
            'averaging_method': 'mean',
            'comment': '',
            'decimation_scheme': create_default_scheme(),
            'intn': None,
            'intt': None,
            'pulse_phase_offset': None,
            'rx_main_antennas': [i for i in self.__options.main_antennas],
            'rx_int_antennas': [i for i in self.__options.intf_antennas],
            'scanbound': None,
            'seqoffset': 0,
            # TODO future proof this by specifying tx_main and tx_int ?? or give spatial information in config
            'tx_antennas': [i for i in self.__options.main_antennas],
            'tx_antenna_pattern': None,
            'tx_beam_order': None,
            'wait_for_first_scanbound': True,
            'xcf': False,
        }

        if self.acf is True:
            if self.xcf is None:
                default_slice_values['xcf'] = True
            if self.acfint is None:
                default_slice_values['acfint'] = True
        elif self.acf is False:
            # TODO log that no xcf or acfint will happen if acfs are not set.
            default_slice_values['xcf'] = False
            default_slice_values['acfint'] = False
        # else they default to False

        if default_slice_values['acf']:
            correct_range_sep = default_slice_values['pulse_len'] * 1.0e-9 * speed_of_light / 2.0    # km
            if self.range_sep is not None:
                if not math.isclose(default_slice_values['range_sep'], correct_range_sep, abs_tol=0.01):
                    errmsg = f"range_sep = {default_slice_values['range_sep']} was set incorrectly. "\
                            f"range_sep will be overwritten based on pulse_len, which must be equal to "\
                            f"1/rx_rate. The new range_sep = {correct_range_sep}"
                    # TODO change to logging
                    print(errmsg)

            default_slice_values['range_sep'] = correct_range_sep
            # This is the distance travelled by the wave in the length of the pulse, divided by
            # two because it's an echo (travels there and back). In km.

            # The below check is an assumption that is made during acf calculation
            # (1 output received sample = 1 range separation)
            if not math.isclose(self.pulse_len * 1.0e-6, (1/self.__output_rx_rate), abs_tol=0.000001):
                errmsg = "For an experiment slice with real-time acfs, pulse length must be equal "\
                        f"(within 1 us) to 1/output_rx_rate to make acfs valid. Current pulse length is "\
                        f"{self.pulse_len} us, output rate is {self.__output_rx_rate} Hz."
                raise ExperimentException(errmsg)

            if self.averaging_method is not '':
                if self.averaging_method not in possible_averaging_methods:
                    errmsg = f"Averaging method {self.averaging_method} not a valid method. "\
                             f"Possible methods are {possible_averaging_methods}"
                    raise ExperimentException(errmsg)

            if self.lag_table is not None:
                # Check that lags are valid
                for lag in self.lag_table:
                    if not set(np.array(lag).flatten()).issubset(set(self.pulse_sequence)):
                        errmsg = f'Lag {lag} not valid; One of the pulses does not exist in the sequence'
                        raise ExperimentException(errmsg)
            else:
                # build lag table from pulse_sequence
                lag_table = list(itertools.combinations(default_slice_values['pulse_sequence'], 2))
                lag_table.append([default_slice_values['pulse_sequence'][0],
                                  default_slice_values['pulse_sequence'][0]])  # lag 0
                # sort by lag number
                lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
                lag_table.append([default_slice_values['pulse_sequence'][-1],
                                  default_slice_values['pulse_sequence'][-1]])  # alternate lag 0
                default_slice_values['lag_table'] = lag_table

        else:
            # TODO log range_sep, lag_table, xcf, acfint, and averaging_method will not be used
            print('range_sep, lag_table, xcf, acfint, and averaging_method will not be used because acf is not True.')
            default_slice_values['range_sep'] = default_slice_values['pulse_len'] * 1.0e-9 * speed_of_light / 2.0
            default_slice_values['lag_table'] = []
            default_slice_values['averaging_method'] = None

        # Update self with the default fields, if necessary
        for k, v in default_slice_values.items():
            if getattr(self, k) is None:    # Hasn't been specified by user, so should default
                setattr(self, k, v)

    def check_slice(self):
        """
        Check the slice for errors.

        This is the first test of the dictionary in the experiment done to ensure values in this
        slice make sense. This is a self-check to ensure the parameters (for example, freq,
        antennas) are appropriate. All fields should be full at this time (whether filled by the
        user or given default values in set_slice_defaults). This was built to be usable at any
        time after setup.

        :raises ExperimentException:    When necessary parameters do not exist or equal None (would
                                        have to have been overridden by the user for this, as
                                        defaults all set when this runs).
        """
        error_list = []
        options = self.__options

        for param in slice_key_set:
            if getattr(self, param) is None:
                if param == 'freq' and (self.clrfrqflag or self.rxonly):
                    pass
                elif param == 'clrfrqrange' and not self.clrfrqflag:
                    pass
                else:
                    # TODO: I don't think this test can be tested by an experiment file, seems to be superseded by
                    #  other tests for necessary params
                    errmsg = f"Slice {self.slice_id} is missing necessary parameter {param}"
                    raise ExperimentException(errmsg)

        # Check that the slice only contains allowed values (excluding python built-in attributes)
        slice_attrs = [k for k in dir(self) if not k.startswith('__')]
        for k in slice_attrs:
            if k not in slice_key_set and k not in hidden_key_set:
                error_list.append(f"Slice {self.slice_id} has a parameter that is not used: {k} = {getattr(self, k)}")

        # TODO : tau_spacing needs to be an integer multiple of pulse_len in ros - is there a max ratio
        #  allowed for pulse_len/tau_spacing ? Add this check and add check for each slice's tx duty-cycle
        #  and make sure we aren't transmitting the entire time after combination with all slices

        if len(self.tx_antennas) > options.main_antenna_count:
            error_list.append(f"Slice {self.slice_id} has too many main TX antenna channels"
                              f" {len(self.tx_antennas)} greater than config {options.main_antenna_count}")
        if len(self.rx_main_antennas) > options.main_antenna_count:
            error_list.append(f"Slice {self.slice_id} has too many main RX antenna channels"
                              f" {len(self.rx_main_antennas)} greater than config {options.main_antenna_count}")
        if len(self.rx_int_antennas) > options.intf_antenna_count:
            error_list.append(f"Slice {self.slice_id} has too many RX interferometer antenna channels "
                              f"{len(self.rx_int_antennas)} greater than config {options.intf_antenna_count}")

        # Check if the antenna identifier number is greater than the config file's
        # maximum antennas for all three of tx antennas, rx antennas and rx int antennas
        # Also check for duplicates
        if max(self.tx_antennas) >= options.main_antenna_count:
            error_list.append(f"Slice {self.slice_id} specifies TX main array antenna numbers over config max "
                              f"{options.main_antenna_count}")
        for i in range(len(self.rx_main_antennas)):
            if self.rx_main_antennas[i] >= options.main_antenna_count:
                error_list.append(f"Slice {self.slice_id} specifies RX main array antenna numbers over config "
                                  f"max {options.main_antenna_count}")
        for i in range(len(self.rx_int_antennas)):
            if self.rx_int_antennas[i] >= options.intf_antenna_count:
                error_list.append(f"Slice {self.slice_id} specifies interferometer array antenna numbers over "
                                  f"config max {options.intf_antenna_count}")
        if list_tests.has_duplicates(self.tx_antennas):
            error_list.append(f"Slice {self.slice_id} TX main antennas has duplicate antennas")
        if list_tests.has_duplicates(self.rx_main_antennas):
            error_list.append(f"Slice {self.slice_id} RX main antennas has duplicate antennas")
        if list_tests.has_duplicates(self.rx_int_antennas):
            error_list.append(f"Slice {self.slice_id} RX interferometer antennas has duplicate antennas")

        # Check if the pulse_sequence is not increasing, which would be an error
        if not list_tests.is_increasing(self.pulse_sequence):
            error_list.append(f"Slice {self.slice_id} pulse_sequence not increasing")

        # Check that pulse_len and tau_spacing make sense (values in us)
        if self.pulse_len > self.tau_spacing:
            error_list.append(f"Slice {self.slice_id} pulse length greater than tau_spacing")
        if self.pulse_len < options.min_pulse_length and \
                self.pulse_len <= 2 * options.pulse_ramp_time * 1.0e6:
            error_list.append(f"Slice {self.slice_id} pulse length too small")
        if self.tau_spacing < options.min_tau_spacing_length:
            error_list.append(f"Slice {self.slice_id} multi-pulse increment too small")
        if not math.isclose((self.tau_spacing * self.__output_rx_rate % 1.0), 0.0, abs_tol=0.0001):
            error_list.append(f"Slice {self.slice_id} correlation lags will be off because tau_spacing "
                              f"{self.tau_spacing} us is not a multiple of the output rx sampling period "
                              f"(1/output_rx_rate {self.__output_rx_rate} Hz).")

        # check intn and intt make sense given tau_spacing, and pulse_sequence.
        if self.pulse_sequence:  # if not empty
            # Sequence length is length of pulse sequence plus the scope sync delay time.
            # TODO this is an old check and seqtime now set in sequences class, update.
            seq_len = self.tau_spacing * (self.pulse_sequence[-1]) + (self.num_ranges + 19 + 10) * self.pulse_len  # us

            if self.intt is None and self.intn is None:
                # both are None and we are not rx - only
                error_list.append(f"Slice {self.slice_id} has transmission but no intt or intn")
            if self.intt is not None and self.intn is not None:
                error_list.append(f"Slice {self.slice_id} choose either intn or intt to be the limit for number"
                                  f" of integrations in an integration period.")
            if self.intt is not None:
                if seq_len > (self.intt * 1000):  # seq_len in us, intt in ms
                    error_list.append(f"Slice {self.slice_id} : pulse sequence is too long for integration time given")
        else:
            if self.tx_beam_order:
                error_list.append(f"Slice {self.slice_id} has transmission defined but no pulse sequence defined")

        if self.pulse_phase_offset:
            num_pulses = len(self.pulse_sequence)

            # Test the encoding fn with beam iterator of 0 and sequence num of 0. test the user's
            # phase encoding function on first beam (beam_iterator = 0) and first sequence
            # (sequence_number = 0)
            phase_encoding = self.pulse_phase_offset(0, 0, num_pulses)

            if not isinstance(phase_encoding, np.ndarray):
                error_list.append(f"Slice {self.slice_id} Phase encoding return is not numpy array")
            else:
                if len(phase_encoding.shape) > 1:
                    error_list.append(f"Slice {self.slice_id} Phase encoding return must be 1 dimensional")
                else:
                    if phase_encoding.shape[0] != num_pulses:
                        error_list.append(f"Slice {self.slice_id} Phase encoding return dimension must be equal"
                                          f" to number of pulses")

        # Initialize to None for tests below
        antenna_pattern = None
        if self.tx_antenna_pattern:
            if not callable(self.tx_antenna_pattern):
                error_list.append(f"Slice {self.slice_id} tx antenna pattern must be a function")
            else:
                tx_freq_khz = self.freq
                antenna_spacing = options.main_antenna_spacing
                antenna_pattern = self.tx_antenna_pattern(tx_freq_khz, self.tx_antennas, antenna_spacing)

                if not isinstance(antenna_pattern, np.ndarray):
                    error_list.append(f"Slice {self.slice_id} tx antenna pattern return is not a numpy array")
                else:
                    if len(antenna_pattern.shape) != 2:
                        error_list.append(f"Slice {self.slice_id} tx antenna pattern return shape "
                                          f"{antenna_pattern.shape} must be 2-dimensional")
                    elif antenna_pattern.shape[1] != options.main_antenna_count:
                        error_list.append(f"Slice {self.slice_id} tx antenna pattern return 2nd dimension "
                                          f"({antenna_pattern.shape[1]}) must be equal to number of main antennas "
                                          f"({options.main_antenna_count})")
                    antenna_pattern_mag = np.abs(antenna_pattern)
                    if np.argwhere(antenna_pattern_mag > 1.0).size > 0:
                        error_list.append(f"Slice {self.slice_id} tx antenna pattern return must not have any "
                                          f"values with a magnitude greater than 1")

        if self.beam_angle:
            if list_tests.has_duplicates(self.beam_angle):
                error_list.append(f"Slice {self.slice_id} beam angles has duplicate directions")
            if not list_tests.is_increasing(self.beam_angle):
                error_list.append(f"Slice {self.slice_id} beam_angle not increasing clockwise (E of N is "
                                  f"positive)")

        # Check if the list of beams to transmit on is empty
        if self.beam_angle and not self.rx_beam_order:
            error_list.append(f"Slice {self.slice_id} rx beam order scan empty")

        # Check that the beam numbers in the tx_beam_order exist
        if self.tx_beam_order:
            if self.tx_antenna_pattern and isinstance(antenna_pattern, np.ndarray):
                num_beams = antenna_pattern.shape[0]
            else:
                num_beams = len(self.beam_angle)
            for bmnum in self.tx_beam_order:
                if bmnum >= num_beams:
                    error_list.append(f"Slice {self.slice_id} scan tx beam number {bmnum} DNE")

        # Check that the beam numbers in the rx_beam_order exist
        for bmnum in self.rx_beam_order:
            if isinstance(bmnum, int):
                if bmnum >= len(self.beam_angle):
                    error_list.append(f"Slice {self.slice_id} scan rx beam number {bmnum} DNE")
            elif isinstance(bmnum, list):
                for imaging_bmnum in bmnum:
                    if imaging_bmnum >= len(self.beam_angle):
                        error_list.append(f"Slice {self.slice_id} scan rx beam number {bmnum} DNE")

        # check scan boundary not less than minimum required scan time.
        if self.scanbound:
            if not self.intt:
                error_list.append(f"Slice {self.slice_id} must have intt enabled to use scanbound")
            elif any(i < 0 for i in self.scanbound):
                error_list.append(f"Slice {self.slice_id} scanbound times must be non-negative")
            elif len(self.scanbound) > 1 and not \
                    all(i < j for i, j in zip(self.scanbound, self.scanbound[1:])):
                error_list.append(f"Slice {self.slice_id} scanbound times must be increasing")
            else:
                # Check if any scanbound times are shorter than the intt.
                tolerance = 1e-9
                if len(self.scanbound) == 1:
                    if self.intt > (self.scanbound[0] * 1000 + tolerance):
                        error_list.append(f"Slice {self.slice_id} intt {self.intt}ms longer than "
                                          f"scanbound time {self.scanbound[0]}s")
                else:
                    for i in range(len(self.scanbound) - 1):
                        beam_time = (self.scanbound[i+1] - self.scanbound[i]) * 1000
                        if self.intt > beam_time + tolerance:
                            error_list.append(f"Slice {self.slice_id} intt {self.intt}ms longer than "
                                              f"one of the scanbound times")
                            break

        # Check wait_for_first_scanbound
        if type(self.wait_for_first_scanbound) is not bool:
            error_list.append(f"Slice {self.slice_id} wait_for_first_scanbound must be True or False, got "
                              f"{self.wait_for_first_scanbound} instead")

        if not isinstance(self.decimation_scheme, DecimationScheme):
            error_list.append(f"Slice {self.slice_id} decimation_scheme is not of type DecimationScheme")

        # TODO other checks

        return error_list
