#!/usr/bin/env python3

"""
    experiment_prototype
    ~~~~~~~~~~~~~~~~~~~~
    This is the base module for all experiments. An experiment will only run if it
    inherits from this class.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

from __future__ import print_function
import sys
import copy
import operator
import os
import math
import numpy as np
import itertools
from scipy.constants import speed_of_light

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_exception import ExperimentException
from sample_building.sample_building import get_wavetables
from experiment_prototype import list_tests

from utils.experiment_options.experimentoptions import ExperimentOptions
from experiment_prototype.scan_classes.scans import Scan, ScanClassBase
from experiment_prototype.decimation_scheme.decimation_scheme import DecimationScheme, DecimationStage, \
    create_default_scheme

interface_types = tuple(['SCAN', 'INTTIME', 'INTEGRATION', 'PULSE'])
""" The types of interfacing available for slices in the experiment.

Interfacing in this case refers to how two or more components are 
meant to be run together. The following types of interfacing are possible:

1. SCAN.
The scan by scan interfacing allows for slices to run a scan of one slice, 
followed by a scan of the second. The scan mode of interfacing typically 
means that the slice will cycle through all of its beams before switching 
to another slice.

There are no requirements for slices interfaced in this manner.

2. INTTIME.
This type of interfacing allows for one slice to run its integration period 
(also known as integration time or averaging period), before switching to 
another slice's integration period. This type of interface effectively creates
an interleaving scan where the scans for multiple slices are run 'at the same
time', by interleaving the integration times.

Slices which are interfaced in this manner must share:
    - the same SCANBOUND value.

3. INTEGRATION.
Integration interfacing allows for pulse sequences defined in the slices to 
alternate between each other within a single integration period. It's important 
to note that data from a single slice is averaged only with other data from that 
slice. So in this case, the integration period is running two slices and can 
produce two averaged datasets, but the sequences (integrations) within the 
integration period are interleaved.

Slices which are interfaced in this manner must share:
    - the same SCANBOUND value.
    - the same INTT or INTN value.
    - the same BEAM_ORDER length (scan length)

4. PULSE.
Pulse interfacing allows for pulse sequences to be run together concurrently. 
Slices will have their pulse sequences summed together so that the 
data transmits at the same time. For example, slices of different frequencies 
can be mixed simultaneously, and slices of different pulse sequences can also 
run together at the cost of having more blanked samples. When slices are 
interfaced in this way the radar is truly transmitting and receiving the 
slices simultaneously.

Slices which are interfaced in this manner must share:
    - the same SCANBOUND value.
    - the same INTT or INTN value.
    - the same BEAM_ORDER length (scan length)

"""

slice_key_set = frozenset(["slice_id", "cpid", "tx_antennas", "rx_main_antennas",
                    "rx_int_antennas", "pulse_sequence", "pulse_phase_offset", "tau_spacing",
                    "pulse_len", "num_ranges", "first_range", "intt", "intn", "beam_angle",
                    "beam_order", "scanbound", "txfreq", "rxfreq",
                    "clrfrqrange", "averaging_method", "acf", "xcf", "acfint",
                    "wavetype", "seqoffset", "iwavetable", "qwavetable",
                    "comment", "range_sep", "lag_table"])

"""
These are the keys that are set by the user when initializing a slice. Some
are required, some can be defaulted, and some are set by the experiment
and are read-only.

**Slice Keys Required by the User**

pulse_sequence *required*
    The pulse sequence timing, given in quantities of tau_spacing, for example
    normalscan = [0, 14, 22, 24, 27, 31, 42, 43].

tau_spacing *required*
    multi-pulse increment (mpinc) in us, Defines minimum space between pulses.

pulse_len *required*
    length of pulse in us. Range gate size is also determined by this.

num_ranges *required*
    Number of range gates.

first_range *required*
    first range gate, in km

intt *required or intn required*
    duration of an integration, in ms. (maximum)

intn *required or intt required*
    number of averages to make a single integration, only used if intt = None.

beam_angle *required*
    list of beam directions, in degrees off azimuth. Positive is E of N. The beam_angle list
    length = number of beams. Traditionally beams have been 3.24 degrees separated but we
    don't refer to them as beam -19.64 degrees, we refer as beam 1, beam 2. Beam 0 will
    be the 0th element in the list, beam 1 will be the 1st, etc. These beam numbers are
    needed to write the beam_order list. This is like a mapping of beam number (list
    index) to beam direction off boresight.

beam_order *required*
    beam numbers written in order of preference, one element in this list corresponds to
    one integration period. Can have lists within the list, resulting in multiple beams
    running simultaneously in the averaging period, so imaging. A beam number of 0 in
    this list gives us the direction of the 0th element in the beam_angle list. It is
    up to the writer to ensure their beam pattern makes sense. Typically beam_order is
    just in order (scanning W to E or E to W, ie. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15]. You can list numbers multiple times in the beam_order list,
    for example [0, 1, 1, 2, 1] or use multiple beam numbers in a single
    integration time (example [[0, 1], [3, 4]], which would trigger an imaging
    integration. When we do imaging we will still have to quantize the directions we
    are looking in to certain beam directions.

clrfrqrange *required or txfreq or rxfreq required*
    range for clear frequency search, should be a list of length = 2, [min_freq, max_freq]
    in kHz. **Not currently supported.**

txfreq *required or clrfrqrange or rxfreq required*
    transmit frequency, in kHz. Note if you specify clrfrqrange it won't be used.

rxfreq *required or clrfrqrange or txfreq required*
    receive frequency, in kHz. Note if you specify clrfrqrange or txfreq it won't be used. Only
    necessary to specify if you want a receive-only slice.


**Defaultable Slice Keys**

acf *defaults*
    flag for rawacf and generation. The default is False. If True, the following fields are
    also used:
    - averaging_method (default 'mean')
    - xcf (default True if acf is True)
    - acfint (default True if acf is True)
    - lagtable (default built based on all possible pulse combos)
    - range_sep (will be built by pulse_len to verify any provided value)

acfint *defaults*
    flag for interferometer autocorrelation data. The default is True if acf is True, otherwise
    False.

averaging_method *defaults*
    a string defining the type of averaging to be done. Current methods are 'mean' or 'median'.
    The default is 'mean'.

comment *defaults*
    a comment string that will be placed in the borealis files describing the slice. Defaults
    to empty string.

lag_table *defaults*
    used in acf calculations. It is a list of lags. Example of a lag: [24, 27] from
    8-pulse normalscan. This defaults to a lagtable built by the pulse sequence
    provided. All combinations of pulses will be calculated, with both the first pulses
    and last pulses used for lag-0.

pulse_phase_offset *defaults*
    Allows phase shifting of pulses, enabling encoding of pulses. Default all
    zeros for all pulses in pulse_sequence. Pulses can be shifted with a single 
    phase shift for each pulse or with a phase shift specified for each sample 
    in the pulses of the slice.

range_sep *defaults*
    a calculated value from pulse_len. If already set, it will be overwritten to be the correct
    value determined by the pulse_len. Used for acfs. This is the range gate separation,
    in azimuthal direction, in km.

rx_int_antennas *defaults*
    The antennas to receive on in interferometer array, default is all
    antennas given max number from config.

rx_main_antennas *defaults*
    The antennas to receive on in main array, default is all antennas
    given max number from config.

scanbound *defaults*
    A list of seconds past the minute for integration times in a scan to align to. Defaults
    to None, not required. If one slice in an experiment has a scanbound, they all 
    must.

seqoffset *defaults*
    offset in us that this slice's sequence will begin at, after the start of the sequence.
    This is intended for PULSE interfacing, when you want multiple slice's pulses in one sequence
    you can offset one slice's sequence from the other by a certain time value so as to not run both
    frequencies in the same pulse, etc. Default is 0 offset.

tx_antennas *defaults*
    The antennas to transmit on, default is all main antennas given max
    number from config.

xcf *defaults*
    flag for cross-correlation data. The default is True if acf is True, otherwise False.


**Read-only Slice Keys**

clrfrqflag *read-only*
    A boolean flag to indicate that a clear frequency search will be done.
    **Not currently supported.**

cpid *read-only*
    The ID of the experiment, consistent with existing radar control programs.
    This is actually an experiment-wide attribute but is stored within the
    slice as well. This is provided by the user but not within the slice,
    instead when the experiment is initialized.

rx_only *read-only*
    A boolean flag to indicate that the slice doesn't transmit, only receives.

slice_id *read-only*
    The ID of this slice object. An experiment can have multiple slices. This
    is not set by the user but instead set by the experiment when the
    slice is added. Each slice id within an experiment is unique. When experiments
    start, the first slice_id will be 0 and incremented from there.

slice_interfacing *read-only*
    A dictionary of slice_id : interface_type for each sibling slice in the
    experiment at any given time.


**Not currently supported and will be removed**

wavetype *defaults*
    string for wavetype. The default is SINE. **Not currently supported.**

iwavetable *defaults*
    a list of numeric values to sample from. The default is None. Not currently supported
    but could be set up (with caution) for non-SINE. **Not currently supported.**

qwavetable *defaults*
    a list of numeric values to sample from. The default is None. Not currently supported
    but could be set up (with caution) for non-SINE. **Not currently supported.**

"""

hidden_key_set = frozenset(['rxonly', 'clrfrqflag', 'slice_interfacing'])
"""
These are used by the build_scans method (called from the experiment_handler every
time the experiment is run). If set by the user, the values will be overwritten and
therefore ignored.
"""

possible_averaging_methods = frozenset(['mean', 'median'])
possible_scheduling_modes = frozenset(['common', 'special', 'discretionary'])
default_rx_bandwidth = 5.0e6
default_output_rx_rate = 10.0e3/3
transition_bandwidth = 750.0e3

class ExperimentPrototype(object):
    """
    The base class for all experiments.

    A prototype experiment class composed of metadata, including experiment slices (exp_slice)
    which are dictionaries of radar parameters. Basic, traditional experiments will be composed of
    a single slice. More complicated experiments will be composed of multiple slices that
    interface in one of four pre-determined ways, as described under interface_types.

    This class is used via inheritance to create experiments.

    Some variables shouldn't be changed by the experiment, and their properties do not have setters.
    Some variables can be changed in the init of your experiment, and can also be modified
    in-experiment by the class method 'update' in your experiment class. These variables have been
    given property setters.

    The following are the user-modifiable attributes of the ExperimentPrototype that are
    used to make an experiment:

    * xcf: boolean for cross-correlation data. A default can be set here for slices,
      but any slice can override this setting with the xcf slice key. 
    * acf: boolean for auto-correlation data on main array. A default can be set here for slices,
      but any slice can override this setting with the acf slice key. 
    * acfint: boolean for auto-correlation data on interferometer array. A default can be set here for slices,
      but any slice can override this setting with the acfint slice key. 

    * slice_dict: modifiable only using the add_slice, edit_slice, and del_slice
      methods.
    * interface: modifiable using the add_slice, edit_slice, and del_slice
      methods, or by updating the interface dict directly.

    Other parameters are set in the init and cannot be modified after instantiation.
    """

    def __init__(self, cpid, output_rx_rate=default_output_rx_rate,
                 rx_bandwidth=default_rx_bandwidth, tx_bandwidth=5.0e6, txctrfreq=12000.0,
                 rxctrfreq=12000.0,
                 decimation_scheme=create_default_scheme(),
                 comment_string=''):
        """
        Initialization for your experiment. Sets experiment-wide settings including cpid,
        center frequencies, sampling rates, and decimation and filtering schemes.
        :param cpid: unique id necessary for each control program (experiment). Cannot be
         changed after instantiation.
        :param output_rx_rate: The desired output rate for the data, to be decimated to, in Hz.
         Cannot be changed after instantiation. Default 3.333 kHz.
        :param rx_bandwidth: The desired bandwidth for the experiment. Directly determines rx
        sampling rate of the USRPs. Cannot be changed after instantiation. Default 5.0 MHz.
        :param rx_bandwidth: The desired tx bandwidth for the experiment. Directly determines tx
        sampling rate of the USRPs. Cannot be changed after instantiation. Default 5.0 MHz.
        :param txctrfreq: center frequency, in kHz, for the USRP to mix the samples with. 
         Since this requires tuning time to set, it cannot be modified after instantiation.
        :param rxctrfreq: center frequency, in kHz, used to mix to baseband.
         Since this requires tuning time to set, it cannot be modified after instantiation.
        :param decimation_scheme: an object defining the decimation and filtering stages for the
        signal processing module. If you would like something other than the default, you will
        need to build an object of the DecimationScheme type before initiating your experiment.
        This cannot be changed after instantiation.
        :param comment_string: description of experiment for data files. This should be 
         used to describe your overall experiment design. Another comment string exists 
         for every slice added, to describe information that is slice-specific.
        """

        if not isinstance(cpid, int):  # TODO add check for uniqueness
            errmsg = 'CPID must be a unique int'
            raise ExperimentException(errmsg)
        if cpid <= 0:
            errmsg = 'The CPID should be a positive number in the experiment. Borealis'\
                     ' will determine if it should be negative based on the scheduling mode.'\
                     ' Only experiments run during discretionary time will have negative CPIDs.'
            raise ExperimentException(errmsg)            

        self.__options = ExperimentOptions()
        self.__experiment_name = self.__class__.__name__  # TODO use this to check the cpid is correct using pygit2, or __class__.__module__ for module name

        self.__cpid = cpid

        self.__scheduling_mode = 'unknown'

        self.__output_rx_rate = float(output_rx_rate)

        if self.output_rx_rate > self.options.max_output_sample_rate:
            errmsg = "Experiment's output sample rate is too high: {} greater than max " \
                     "{}.".format(self.output_rx_rate, self.options.max_output_sample_rate)
            raise ExperimentException(errmsg)

        self.__txrate = float(tx_bandwidth)  # sampling rate, samples per sec, Hz.
        self.__rxrate = float(rx_bandwidth) # sampling rate for rx in samples per sec
        # Transmitting is possible in the range of txctrfreq +/- (txrate/2) because we have iq data
        # Receiving is possible in the range of rxctrfreq +/- (rxrate/2)

        if self.txrate > self.options.max_tx_sample_rate:
            errmsg = "Experiment's transmit bandwidth is too large: {} greater than max " \
                     "{}.".format(self.txrate, self.options.max_tx_sample_rate)
            raise ExperimentException(errmsg)

        if self.rxrate > self.options.max_rx_sample_rate:
            errmsg = "Experiment's receive bandwidth is too large: {} greater than max " \
                     "{}.".format(self.rxrate, self.options.max_rx_sample_rate)
            raise ExperimentException(errmsg)

        if round(self.options.usrp_master_clock_rate / self.txrate, 3) % 2.0 != 0.0:
            errmsg = "Experiment's transmit bandwidth {} is not possible as it must be an integer divisor of " \
                     " USRP master clock rate {}".format(self.txrate, self.options.usrp_master_clock_rate)
            raise ExperimentException(errmsg)

        if round(self.options.usrp_master_clock_rate / self.rxrate, 3) % 2.0 != 0.0:
            errmsg = "Experiment's receive bandwidth {} is not possible as it must be an integer divisor of " \
                     " USRP master clock rate {}".format(self.rxrate, self.options.usrp_master_clock_rate)
            raise ExperimentException(errmsg)

        self.__decimation_scheme = decimation_scheme

        self.__comment_string = comment_string

        self.__slice_dict = {}

        self.__new_slice_id = 0

        # Note - txctrfreq and rxctrfreq are set here and modify the actual center frequency to a
        # multiple of the clock divider that is possible by the USRP - this default value set
        # here is not exact (center freq is never exactly 12 MHz).

        # convert from kHz to Hz to get correct clock divider. Return the result back in kHz.
        clock_multiples = self.options.usrp_master_clock_rate/2**32
        clock_divider = math.ceil(txctrfreq*1e3/clock_multiples)
        self.__txctrfreq = (clock_divider * clock_multiples)/1e3

        clock_divider = math.ceil(rxctrfreq*1e3/clock_multiples)
        self.__rxctrfreq = (clock_divider * clock_multiples)/1e3

        # Load the config, hardware, and restricted frequency data

        # This is experiment-wide transmit metadata necessary to build the pulses. This data
        # cannot change within the experiment and is used in the scan classes to pass information
        # to where the samples are built.
        self.__transmit_metadata = {
            'output_rx_rate': self.output_rx_rate,
            'main_antenna_count': self.options.main_antenna_count,
            'tr_window_time': self.options.tr_window_time,
            'main_antenna_spacing': self.options.main_antenna_spacing,
            'pulse_ramp_time': self.options.pulse_ramp_time,
            'max_usrp_dac_amplitude': self.options.max_usrp_dac_amplitude,
            'rx_sample_rate': self.rxrate,
            'minimum_pulse_separation': self.options.minimum_pulse_separation,
            'txctrfreq': self.txctrfreq,
            'txrate': self.txrate
        }

        # The following are processing defaults. These can be set by the experiment using the setter
        #   upon instantiation. These are defaults for all slices, but these values are
        #   slice-specific so if the slice is added with these flags specified, that will override
        #   these values for the specific slice.
        self._xcf = False  # cross-correlation
        self._acf = False  # auto-correlation
        self._acfint = False  # interferometer auto-correlation.

        self.__interface = {}
        # Dictionary of how each exp_slice interacts with the other slices.
        # NOTE keys are as such: (0,1), (0,2), (1,2),
        # NEVER includes (2,0) etc. The only interface options are those specified in
        # interface_types.

        # The following are for internal use only, and should not be modified in the experimental
        #  class, but will be modified by the class method build_scans. For this reason they
        #  are private, with getters only, in case they are used for reference by the user.
        # These are used internally to build iterable objects out of the slice using the
        # interfacing specified.

        self.__scan_objects = []
        self.__scanbound = False
        self.__running_experiment = None  # this will be of ScanClassBase type

    __slice_keys = slice_key_set
    __hidden_slice_keys = hidden_key_set
    __default_output_rx_rate = default_output_rx_rate
    __default_rx_bandwidth = default_rx_bandwidth

    @property
    def cpid(self):
        """
        This experiment's CPID (control program ID, a term that comes from ROS).

        The CPID is read-only once established in instantiation. It may be 
        modified at runtime by the set_scheduling_mode function, to set it to
        a negative value during discretionary time.
        """

        return self.__cpid

    @property
    def experiment_name(self):
        """
        The experiment class name.
        """
        return self.__experiment_name

    @property
    def output_rx_rate(self):
        """
        The output receive rate of the data, Hz.

        This is read-only once established in instantiation.
        """

        return self.__output_rx_rate

    @property
    def tx_bandwidth(self):
        """
        The transmission sample rate to the DAC (Hz), and the transmit bandwidth.

        This is read-only once established in instantiation.
        """

        return self.__txrate

    @property
    def txrate(self):
        """
        The transmission sample rate to the DAC (Hz).

        This is read-only once established in instantiation.
        """

        return self.__txrate

    @property
    def rx_bandwidth(self):
        """
        The receive bandwidth for this experiment, in Hz.

        This is read-only once established in instantiation.
        """

        return self.__rxrate

    @property
    def rxrate(self):
        """
        The receive bandwidth for this experiment, or the receive sampling rate (of I and Q samples)
        In Hz.

        This is read-only once established in instantiation.
        """

        return self.__rxrate

    @property
    def decimation_scheme(self):
        """
        The decimation scheme, of type DecimationScheme from the filtering module. Includes all
        filtering and decimating information for the signal processing module.

        This is read-only once established in instantiation.
        """
        return self.__decimation_scheme

    @property
    def comment_string(self):
        """
        A string related to the experiment, to be placed in the experiment's files.

        This is read-only once established in instantiation.
        """

        return self.__comment_string

    @property
    def num_slices(self):
        """
        The number of slices currently in the experiment.

        Will change after methods add_slice or del_slice are called.
        """

        return len(self.__slice_dict)

    @property
    def slice_keys(self):
        """
        The list of slice keys available.

        This cannot be updated. These are the keys in the current ExperimentPrototype
        slice_keys dictionary (the parameters available for slices).
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
        """

        return self.__slice_dict

    @property
    def new_slice_id(self):
        """
        The next unique slice id that is available to this instance of the experiment.

        This gets incremented each time it is called to ensure it returns
        a unique ID each time.
        """

        self.__new_slice_id += 1
        return self.__new_slice_id - 1

    @property
    def slice_ids(self):
        """
        The list of slice ids that are currently available in this experiment.

        This can change when add_slice, edit_slice, and del_slice are called.
        """

        return list(self.__slice_dict.keys())

    @property
    def options(self):
        """
        The config options for running this experiment.

        These cannot be set or removed, but are specified in the config.ini, hdw.dat, and
        restrict.dat files.
        """

        return self.__options

    @property
    def transmit_metadata(self):
        """
        A dictionary of config options and experiment-set values that cannot change in the
        experiment, that will be used to build pulse sequences.
        """

        return self.__transmit_metadata


    @property
    def xcf(self):
        """
        The default cross-correlation flag boolean.

        This provides the default for slices where this key isn't specified.
        """

        return self._xcf

    @xcf.setter
    def xcf(self, value):
        """
        Set the cross-correlation flag default for new slices.

        :param value: boolean for cross-correlation processing flag.
        """

        if isinstance(value, bool):
            self._xcf = value
        else:
            pass  # TODO log no change - throw an exception perhaps

    @property
    def acf(self):
        """
        The default auto-correlation flag boolean.

        This provides the default for slices where this key isn't specified.
        """

        return self._acf

    @acf.setter
    def acf(self, value):
        """
        Set the auto-correlation flag default for new slices.

        :param value: boolean for auto-correlation processing flag.
        """

        if isinstance(value, bool):
            self._acf = value
        else:
            pass  # TODO log no change

    @property
    def acfint(self):
        """
        The default interferometer autocorrelation boolean.

        This provides the default for slices where this key isn't specified.
        """

        return self._acfint

    @acfint.setter
    def acfint(self, value):
        """
        Set the interferometer autocorrelation flag default for new slices.

        :param value: boolean for interferometer autocorrelation processing flag.
        """

        if isinstance(value, bool):
            self._acfint = value
        else:
            pass  # TODO log no change


    @property
    def txctrfreq(self):
        """
        The transmission center frequency that USRP is tuned to (kHz).
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
        """
        max_freq = self.txctrfreq * 1000 + (self.txrate/2.0) - transition_bandwidth
        if max_freq < self.options.max_freq:
            return max_freq
        else:
            # TODO log warning that wave_freq should not exceed options.max_freq - ctrfreq (possible to transmit above licensed band)
            return self.options.max_freq

    @property
    def tx_minfreq(self):
        """
        The minimum transmit frequency.

        This is the minimum tx frequency possible in this experiment (either minimum in our license
        or minimum given by the center frequency and sampling rate). The minimum is slightly more
        than that allowed by the center frequency and txrate, to stay away from the edges of the
        possible transmission band where the signal is distorted.
        """
        min_freq = self.txctrfreq * 1000 - (self.txrate/2.0) + transition_bandwidth
        if min_freq > self.options.min_freq:
            return min_freq
        else:
            # TODO log warning that wave_freq should not go below ctrfreq - options.minfreq (possible to transmit below licensed band)
            return self.options.min_freq

    @property
    def rxctrfreq(self):
        """
        The receive center frequency that USRP is tuned to (kHz).
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
        """
        min_freq = self.rxctrfreq * 1000 - (self.rxrate/2.0) + transition_bandwidth
        if min_freq > 1000: #Hz
            return min_freq
        else:
            return 1000 # Hz

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

        """
        return self.__interface

    @property
    def scan_objects(self):
        """
        The list of instances of class Scan for use in radar_control.

        These cannot be modified by the user, but are created using the slice dictionary.
        """
        return self.__scan_objects

    @property
    def scheduling_mode(self):
        """
        Return the scheduling mode time type that this experiment is running
        in. Types are listed in possible_scheduling_modes. Initialized to
        'unknown' until set by the experiment handler.
        """
        return self.__scheduling_mode

    def _set_scheduling_mode(self, scheduling_mode):
        """
        Set the scheduling mode if the provided mode is valid. Should only
        be called by the experiment handler after initializing the user's
        class.
        """
        if scheduling_mode in possible_scheduling_modes:
            self.__scheduling_mode = scheduling_mode
            if scheduling_mode == 'discretionary':
                self.__cpid = -1 * self.__cpid
        else:
            errmsg = 'Scheduling mode {} set by experiment handler is not '\
                     ' a valid mode: {}'.format(scheduling_mode,
                            possible_scheduling_modes)
            raise ExperimentException(errmsg)
    
    def printing(self, msg):
        EXPERIMENT_P = "\033[34m" + self.__class__.__name__ + " : " + "\033[0m"
        sys.stdout.write(EXPERIMENT_P + msg + "\n")

    def slice_beam_directions_mapping(self, slice_id):
        """
        A mapping of the beam directions in the given slice id.

        :param slice_id: id of the slice to get beam directions for.
        :returns mapping: enumeration mapping dictionary of beam number to beam
         direction(s) in degrees off boresight.
        """
        if slice_id not in self.slice_ids:
            return {}
        beam_directions = self.slice_dict[slice_id]['beam_angle']
        mapping = {}
        for beam_num, beam_dir in enumerate(beam_directions):
            mapping[beam_num] = beam_dir
        return mapping

    def check_new_slice_interfacing(self, interfacing_dict):
        """
        Checks that the new slice plays well with its siblings (has interfacing
        that is resolvable). If so, returns a new dictionary with all interfacing
        values set.

        The interfacing assumes that the interfacing_dict given by the user defines
        the closest interfacing of the new slice with a slice. For example,
        if the slice is to be PULSE combined with slice 0, the interfacing dict
        should provide this information. If only 'SCAN' interfacing with slice 1
        is provided, then that will be assumed to be the closest and therefore
        the interfacing with slice 0 will also be 'SCAN'.

        If no interfacing_dict is provided for a slice, the default
        is to do 'SCAN' type interfacing for the new slice with all other slices.

        :param interfacing_dict: the user-provided interfacing dict, which may
         be empty or incomplete. If empty, all interfacing is assumed to be =
         'SCAN' type. If it contains something, we ensure that the interfacing provided
         makes sense with the values already known for its closest sibling.
        :returns: full interfacing dictionary.
        :raises: ExperimentException if invalid interface types provided
         or if interfacing can not be resolved.
        """

        for sibling_slice_id, interface_value in interfacing_dict.items():
            if interface_value not in interface_types:
                errmsg = 'Interface value with slice {}: {} not valid. Types available are:'\
                         '{}'.format(sibling_slice_id, interface_value, interface_types)
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
                        errmsg = 'Cannot add slice: the interfacing_dict set interfacing to an unknown slice'\
                                 '{} not in slice ids {}'.format(sibling_slice_id, self.slice_ids)
                        raise ExperimentException(errmsg)
                try:
                    closest_sibling = max(interfacing_dict.keys(),
                                          key=lambda k: interface_types.index(
                                               interfacing_dict[k]))
                except ValueError as e:  # cannot find interface type in list
                    errmsg = 'Interface types must be of valid types {}.'\
                             ''.format(interface_types)
                    raise ExperimentException(errmsg) from e
                closest_interface_value = interfacing_dict[closest_sibling]
                closest_interface_rank = interface_types.index(closest_interface_value)
            else:
                # the user provided no keys. The default is therefore 'SCAN'
                # with all keys so the closest will be 'SCAN' (the furthest possible interface_type)
                closest_sibling = self.slice_ids[0]
                closest_interface_value = 'SCAN'
                closest_interface_rank = interface_types.index(closest_interface_value)

            # now populate a full_interfacing_dict based on the closest sibling's
            # interface values and knowing how we interface with that sibling.
            # this is the only correct interfacing given the closest interfacing.
            full_interfacing_dict[closest_sibling] = closest_interface_value
            for sibling_slice_id, siblings_interface_value in self.get_slice_interfacing(closest_sibling).items():
                if interface_types.index(siblings_interface_value) >= closest_interface_rank:
                    # in this case, the interfacing between the sibling
                    # and the closest sibling is closer than the closest interface for the new slice.
                    # therefore interface with this sibling should be equal to the closest interface.
                    # Or if they are all at the same rank, then the interfacing should equal that rank.
                    # For example, slices 0 and 1 combined PULSE. New slice 2 is
                    # added with closest interfacing INTEGRATION to slice 0. Slice
                    # 2 will therefore also be interfaced with slice 1 as INTEGRATION
                    # type, since both slices 0 and 1 are in a single INTEGRATION.
                    full_interfacing_dict[sibling_slice_id] = closest_interface_value
                else:  # the rank is less than the closest rank.
                    # in this case, the interfacing to this sibling should be the same as the
                    # closest sibling interface to this sibling.
                    # For example, slices 0 and 1 are combined SCAN and
                    # slice 2 is combined INTTIME with slice 0 (closest). Therefore slice 2
                    # should be combined SCAN with slice 1 since 0 and 2 are now
                    # within the same scan.
                    full_interfacing_dict[sibling_slice_id] = siblings_interface_value

            # now check everything provided by the user with the correct full_interfacing_dict
            # that was populated based on the closest sibling given by the user.
            for sibling_slice_id, interface_value in interfacing_dict.items():
                if interface_value != full_interfacing_dict[sibling_slice_id]:
                    siblings_interface_value = self.get_slice_interfacing(closest_sibling)[sibling_slice_id]
                    errmsg = 'The interfacing values of new slice cannot be reconciled. '\
                             'Interfacing with slice {closest}: {interface1} and with slice '\
                             '{other}: {interface2} does not make sense with existing interface between '\
                             ' slices of {sibling_other}: {interface3}'.format(closest=closest_sibling,
                                interface1=closest_interface_value, other=sibling_slice_id,
                                interface2=interface_value,
                                sibling_other=([sibling_slice_id, closest_sibling].sort()),
                                interface3=siblings_interface_value)
                    raise ExperimentException(errmsg)

        return full_interfacing_dict

    def __update_slice_interfacing(self):
        """
        Internal slice interfacing updater. This
        should only be used internally when slice dictionary is
        changed, to update all of the slices' interfacing dictionaries.
        """
        for slice_id in self.slice_ids:
            self.__slice_dict[slice_id]['slice_interfacing'] = \
                self.get_slice_interfacing(slice_id)

    def add_slice(self, exp_slice, interfacing_dict={}):
        """
        Add a slice to the experiment.

        :param exp_slice: a slice (dictionary of slice_keys) to add to the experiment.
        :param interfacing_dict: dictionary of type {slice_id : INTERFACING , ... } that
         defines how this slice interacts with all the other slices currently in the
         experiment.
        :raises: ExperimentException if slice is not a dictionary or if there are
         errors in setup_slice.
        :return: the slice_id of the new slice that was just added.
        """

        if not isinstance(exp_slice, dict):
            errmsg = 'Attempt to add a slice failed - {} is not a dictionary of slice' \
                     'parameters'.format(exp_slice)
            raise ExperimentException(errmsg)
            # TODO multiple types of Exceptions so they can be caught by the experiment in these
            # add_slice, edit_slice, del_slice functions (and handled specifically)

        add_slice_id = exp_slice['slice_id'] = self.new_slice_id
        # each added slice has a unique slice id, even if previous slices have been deleted.
        exp_slice['cpid'] = self.cpid

        # Now we setup the slice which will check minimum requirements and set defaults, and then
        # will complete a check_slice and raise any errors found.
        new_exp_slice = self.setup_slice(exp_slice)

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

        :param remove_slice_id: the id of the slice you'd like to remove.
        :returns: a copy of the removed slice.
        :raises: exception if remove_slice_id does not exist in the slice dictionary.
        """
        try:
            removed_slice = copy.deepcopy(self.slice_dict[remove_slice_id])
            del(self.slice_dict[remove_slice_id])
        except (IndexError, TypeError) as e:
            errmsg = 'Cannot remove slice id {} : it does not exist in slice dictionary'.format(remove_slice_id)
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

        A quick way to edit a slice. In reality this is actually adding a new slice and
        deleting the old one. Useful for quick changes. Note that using this function
        will remove the slice_id that you are changing and will give it a new id. It will
        account for this in the interfacing dictionary.

        :param edit_slice_id: the slice id of the slice to be edited.
        :param kwargs: dictionary of slice parameter to slice value that you want to
         change.
        :returns new_slice_id: the new slice id of the edited slice, or the edit_slice_id
         if no change has occurred due to failure of new slice parameters to pass experiment
         checks.
        :raises: exceptions if the edit_slice_id does not exist in slice dictionary or
         the params or values do not make sense.
        """
        slice_params_to_edit = dict(kwargs)

        try:
            edited_slice = self.slice_dict[edit_slice_id].copy()
        except (IndexError, TypeError):
            # the edit_slice_id is not an index in the slice_dict
            errmsg = 'Trying to edit {} but it does not exist in Slice_IDs' \
                     ' list.'.format(edit_slice_id)
            raise ExperimentException(errmsg)

        for edit_slice_param, edit_slice_value in slice_params_to_edit.items():
            if edit_slice_param in self.slice_keys:
                edited_slice[edit_slice_param] = edit_slice_value
            else:
                errmsg = 'Cannot edit slice: {} not a valid slice parameter'.format(edit_slice_param)
                raise ExperimentException(errmsg)

        # Get the interface values of the slice. These are not editable, if
        # these are wished to be changed add_slice must be used explicitly
        # to interface a new slice.
        interface_values = self.get_slice_interfacing(edit_slice_id)

        removed_slice = self.del_slice(edit_slice_id)

        try:
            # checks are done on interfacing when slice is added.
            # interfacing between existing slice_ids cannot be changed after addition.
            new_slice_id = self.add_slice(edited_slice, interface_values)
            return new_slice_id

        except ExperimentException:
            # if any failure occurs when checking the slice, the slice has
            # not been added to the slice dictionary so we will
            # revert to old slice
            self.__slice_dict[edit_slice_id] = removed_slice

            readd_keys = []
            for key1, key1_interface in interface_values.items():
                if key1 < edit_slice_id:
                    self.__interface[(key1, edit_slice_id)] = key1_interface
                else:
                    self.__interface[(edit_slice_id, key1)] = key1_interface

            # reset all slice_interfacing back
            self.__update_slice_interfacing()

            return edit_slice_id

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
        Build the scan information, which means creating the Scan, AveragingPeriod, and
        Sequence instances needed to run this experiment.

        Will be run by experiment handler, to build iterable objects for radar_control to
        use. Creates scan_objects in the experiment for identifying which slices are in the scans.
        """

        # Check interfacing and other experiment-wide settings.
        self.self_check()

        # investigating how I might go about using this base class - TODO maybe make a new IterableExperiment class to inherit

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
            if scan.scanbound != None:
                self.__scanbound = True

        if self.__scanbound:
            try:
                self.__scan_objects = sorted(self.__scan_objects, key=lambda scan: scan.scanbound[0])
            except IndexError as e:  # scanbound is None in some scans
                errmsg = 'If one slice has a scanbound, they all must to avoid up to minute-long downtimes.'
                raise ExperimentException(errmsg) from e

        if __debug__:
            print("Number of Scan types: {}".format(len(self.__scan_objects)))
            print("Number of AveragingPeriods in Scan #1: {}".format(len(self.__scan_objects[
                                                                             0].aveperiods)))
            print("Number of Sequences in Scan #1, Averaging Period #1: {}".format(
                len(self.__scan_objects[0].aveperiods[0].sequences)))
            print("Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1:"
                  " {}".format(len(self.__scan_objects[0].aveperiods[0].sequences[0].slice_dict)))

    def get_scan_slice_ids(self):
        # TODO add this to ScanClassBase method by just passing in the current type (Experiment, Scan, AvePeriod)
        # which would allow you to determine which interfacing to pull out.
        """
        Organize the slice_ids by scan.

        Take my own interfacing and get info on how many scans and which slices make which
        scans. Return a list of lists where each inner list contains the slices that
        are in an averagingperiod that is inside this scan. ie. len(nested_slice_list)
        = # of averagingperiods in this scan, len(nested_slice_list[0]) = # of slices
        in the first averagingperiod, etc.

        :return list of lists. The list has one element per scan. Each element is a list
        of slice_ids signifying which slices are combined inside that scan. The list
        returned could be of length 1, meaning only one scan is present in the experiment.
        """
        scan_combos = []

        for k, interface_value in self.interface.items():
            if interface_value != "SCAN":
                scan_combos.append(list(k))

        combos = self.__running_experiment.slice_combos_sorter(scan_combos, self.slice_ids)

        return combos

    def check_slice_minimum_requirements(self, exp_slice):
        """
        Check the required slice keys.

        Check for the minimum requirements of the slice. The following keys are always required:
        "pulse_sequence", "tau_spacing", "pulse_len", "num_ranges", "first_range", (one of "intt" or "intn"),
        "beam_angle", and "beam_order". This function may modify the values in this slice dictionary
        to ensure that it is able to be run and that the values make sense.

        :param exp_slice: slice to check.
        """

        # TODO: add checks for values that make sense, not just check for types
        # TODO: make lists of operations to run and use if any() to shorten up this code!
        if not isinstance(exp_slice['pulse_sequence'], list):
            errmsg = "Slice must specify pulse_sequence that must be a list of integers"
            raise ExperimentException(errmsg, exp_slice)
        for element in exp_slice['pulse_sequence']:
            if not isinstance(element, int):
                errmsg = "Slice must specify pulse_sequence that must be a list of integers"
                raise ExperimentException(errmsg, exp_slice)

        if 'tau_spacing' not in exp_slice.keys() or not isinstance(exp_slice['tau_spacing'], int):
            errmsg = "Slice must specify tau_spacing that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        # TODO may want to add a field for range_gate which could set this param.
        if 'pulse_len' not in exp_slice.keys() or not isinstance(exp_slice['pulse_len'], int):
            errmsg = "Slice must specify pulse_len that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        if 'num_ranges' not in exp_slice.keys() or not isinstance(exp_slice['num_ranges'], int):
            errmsg = "Slice must specify num_ranges that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        if 'first_range' not in exp_slice.keys() or not isinstance(exp_slice['first_range'], int):
            errmsg = "Slice must specify first_range in km that must be an integer"
            raise ExperimentException(errmsg, exp_slice)

        if 'intt' not in exp_slice.keys():
            if 'intn' not in exp_slice.keys():
                errmsg = "Slice must specify either an intn or intt"
                raise ExperimentException(errmsg, exp_slice)
            elif not isinstance(exp_slice['intn'], int):
                errmsg = "intn must be an integer"
                raise ExperimentException(errmsg, exp_slice)
        else:
            if not isinstance(exp_slice['intt'], float) and not isinstance(exp_slice['intt'], int):
                errmsg = "intt must be an number"
                raise ExperimentException(errmsg, exp_slice)
            else:
                if 'intn' in exp_slice.keys():
                    if __debug__:
                        print('intn is set in experiment slice but will not be used due to intt')
                    # TODO Log warning intn will not be used
                    exp_slice.pop('intn')
            exp_slice['intt'] = float(exp_slice['intt'])

        if 'beam_angle' not in exp_slice.keys(): # "beam_angle" is a required key
            errmsg = """Slice must specify beam_angle that must be a list of numbers (ints or
                floats) which are angles of degrees off boresight (positive E of N)"""
            raise ExperimentException(errmsg, exp_slice)
        if not isinstance(exp_slice['beam_angle'], list):
            errmsg = """Slice must specify beam_angle that must be a list of numbers (ints or
                floats) which are angles of degrees off boresight (positive E of N)"""
            raise ExperimentException(errmsg, exp_slice)
        for element in exp_slice['beam_angle']:
            if not isinstance(element, float) and not isinstance(element, int):
                errmsg = """Slice must specify beam_angle that must be a list of numbers (ints or
                    floats) which are angles of degrees off boresight (positive E of N)"""
                raise ExperimentException(errmsg, exp_slice)
            if isinstance(element, int):
                element = float(element)

        if 'beam_order' not in exp_slice.keys():
            errmsg = """Slice must specify beam_order that must be a list of ints or lists (of ints)
                     corresponding to the order of the angles in the beam_angle list."""
            raise ExperimentException(errmsg, exp_slice)
        if not isinstance(exp_slice['beam_order'], list):
            errmsg = """Slice must specify beam_order that must be a list of ints or lists (of ints)
                     corresponding to the order of the angles in the beam_angle list."""
            raise ExperimentException(errmsg, exp_slice)
        for element in exp_slice['beam_order']:
            if not isinstance(element, int) and not isinstance(element, list):
                errmsg = """Slice must specify beam_order that must be a list of ints or lists (of ints)
                         corresponding to the order of the angles in the beam_angle list."""
                raise ExperimentException(errmsg, exp_slice)
            if isinstance(element, list):
                for beamnum in element:
                    if not isinstance(beamnum, int):
                        errmsg = """Slice must specify beam_order that must be a list of ints or lists (of ints)
                                 corresponding to the order of the angles in the beam_angle list."""
                        raise ExperimentException(errmsg, exp_slice)
                    if beamnum >= len(exp_slice['beam_angle']):
                        errmsg = """Slice must specify beam_order that must be a list of ints or lists (of ints)
                                 corresponding to the order of the angles in the beam_angle list."""
                        raise ExperimentException(errmsg, exp_slice)
            else:
                if element >= len(exp_slice['beam_angle']):
                    errmsg = """Slice must specify beam_order that must be a list of ints or lists (of ints)
                             corresponding to the order of the angles in the beam_angle list."""
                    raise ExperimentException(errmsg, exp_slice)

    @staticmethod
    def set_slice_identifiers(exp_slice):
        """
        Set the hidden slice keys to determine how to run the slice.

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
                pass  # TODO log a warning. Txfreq is removed as clrfrqrange takes precedence but
                # we may not be doing as you intended.

            rxfreq = exp_slice.pop('rxfreq', None)
            if rxfreq is not None and rxfreq not in \
                    range(exp_slice['clrfrqrange'][0],
                          exp_slice['clrfrqrange'][1]):
                pass  # TODO log a warning. Rxfreq is removed as clrfrqrange takes precedence
                # but we may not be doing as you intended.

        elif 'txfreq' in exp_slice.keys():
            exp_slice['clrfrqflag'] = False
            exp_slice['rxonly'] = False
            rxfreq = exp_slice.pop('rxfreq', None)
            if rxfreq is not None and rxfreq != exp_slice['txfreq']:
                pass  # TODO log a warning. Rxfreq is removed as txfreq takes precedence but we may
                # not be doing as you intended.
        elif 'rxfreq' in exp_slice.keys():
            exp_slice['rxonly'] = True
            exp_slice['clrfrqflag'] = False
        else:
            errmsg = 'An rxfreq, txfreq, or clrfrqrange must be specified in a slice'
            raise ExperimentException(errmsg, exp_slice)

    def check_slice_specific_requirements(self, exp_slice):
        """
        Set the specific slice requirements depending.

        Check the requirements for the specific slice type as identified by the
        identifiers rxonly and clrfrqflag. The keys that need to be checked depending
        on these identifiers are "txfreq", "rxfreq", and "clrfrqrange". This function
        may modify these keys.

        :param exp_slice: the slice to check, before adding to the experiment.
        """
        if exp_slice['clrfrqflag']:  # TX and RX mode with clear frequency search.
            # In this mode, clrfrqrange is required along with the other requirements.
            if not isinstance(exp_slice['clrfrqrange'], list):
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)
            if len(exp_slice['clrfrqrange']) != 2:
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)
            if not isinstance(exp_slice['clrfrqrange'][0], int) or not isinstance(exp_slice[\
                    'clrfrqrange'][1], int):
                errmsg = 'clrfrqrange must be an integer list of length = 2'
                raise ExperimentException(errmsg)

            if exp_slice['clrfrqrange'][0] >= exp_slice['clrfrqrange'][1]:
                errmsg = """clrfrqrange must be between min and max tx frequencies {} and rx
                            frequencies {} according to license and/or center frequencies / sampling
                            rates / transition bands, and must have lower frequency first.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            if (exp_slice['clrfrqrange'][1] * 1000) >= self.tx_maxfreq or \
                    (exp_slice['clrfrqrange'][1] * 1000) >= self.rx_maxfreq:
                errmsg = """clrfrqrange must be between min and max tx frequencies {} and rx
                            frequencies {} according to license and/or center frequencies / sampling
                            rates / transition bands, and must have lower frequency first.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            if (exp_slice['clrfrqrange'][0] * 1000) <= self.tx_minfreq or \
                    (exp_slice['clrfrqrange'][0] * 1000) <= self.rx_minfreq:
                errmsg = """clrfrqrange must be between min and max tx frequencies {} and rx
                            frequencies {} according to license and/or center frequencies / sampling
                            rates / transition bands, and must have lower frequency first.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)

            still_checking = True
            while still_checking:
                for freq_range in self.options.restricted_ranges:
                    if exp_slice['clrfrqrange'][0] in range(freq_range[0], freq_range[1]):
                        if exp_slice['clrfrqrange'][1] in range(freq_range[0], freq_range[1]):
                            # the range is entirely within the restricted range.
                            raise ExperimentException('clrfrqrange is entirely within restricted '
                                                      'range {}'.format(freq_range))
                        else:
                            if __debug__:
                                print('Clrfrqrange will be modified because it is partially in a ' +
                                'restricted range.')
                            # TODO Log warning, changing clrfrqrange because lower portion is in a
                            # restricted frequency range.
                            exp_slice['clrfrqrange'][0] = freq_range[1] + 1
                            # outside of restricted range now.
                            break  # we have changed the 'clrfrqrange' - must restart the
                            # check in case it's in another range.
                    else:
                        # lower end is not in restricted frequency range.
                        if exp_slice['clrfrqrange'][1] in range(freq_range[0], freq_range[1]):
                            if __debug__:
                                print('Clrfrqrange will be modified because it is partially in a ' +
                                'restricted range.')
                            # TODO Log warning, changing clrfrqrange because upper portion is in a
                            # restricted frequency range.
                            exp_slice['clrfrqrange'][1] = freq_range[0] - 1
                            # outside of restricted range now.
                            break  # we have changed the 'clrfrqrange' - must restart the for loop
                            # checking in case it's in another range.
                        else:  # neither end of clrfrqrange is inside the restricted range but
                            # we should check if the range is inside the clrfrqrange.
                            if freq_range[0] in range(exp_slice['clrfrqrange'][0],
                                                                  exp_slice['clrfrqrange'][1]):
                                if __debug__:
                                    print('There is a restricted range within the clrfrqrange - '
                                          'STOP.')
                                # TODO Log a warning that there is a restricted range in the middle
                                # of the
                                # clrfrqrange that will be avoided OR could make this an Error.
                                # Still need to implement clear frequency searching.
                else:  # no break, so no changes to the clrfrqrange
                    still_checking = False

        elif exp_slice['rxonly']:  # RX only mode.
            # In this mode, rxfreq is required.
            if not isinstance(exp_slice['rxfreq'], int) and not isinstance(exp_slice['rxfreq'],
                                                                      float):
                errmsg = """rxfreq must be a number (kHz) between rx min and max frequencies {} for
                            the radar license and be within range given center frequency, sampling 
                            rate and transition band.""".format((self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            if (exp_slice['rxfreq'] * 1000) >= self.rx_maxfreq or (exp_slice['rxfreq'] *
                                                                   1000) <= self.rx_minfreq:
                errmsg = """rxfreq must be a number (kHz) between rx min and max frequencies {} for
                            the radar license and be within range given center frequency, sampling
                            rate and transition band.""".format((self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)

        else:  # TX-specific mode , without a clear frequency search.
            # In this mode, txfreq is required along with the other requirements.
            if not isinstance(exp_slice['txfreq'], int) and not isinstance(exp_slice['txfreq'],
                                                                          float):
                errmsg = """txfreq must be a number (kHz) between tx min and max frequencies {} and
                            rx min and max frequencies {} for the radar license and be within range
                            given center frequencies, sampling rates and transition band.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            if (exp_slice['txfreq'] * 1000) >= self.tx_maxfreq or (exp_slice['txfreq'] * 1000) >= \
                    self.rx_maxfreq:
                errmsg = """txfreq must be a number (kHz) between tx min and max frequencies {} and
                            rx min and max frequencies {} for the radar license and be within range
                            given center frequencies, sampling rates and transition band.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            if (exp_slice['txfreq'] * 1000) <= self.tx_minfreq or (exp_slice['txfreq'] * 1000) <= \
                    self.rx_minfreq:
                errmsg = """txfreq must be a number (kHz) between tx min and max frequencies {} and
                            rx min and max frequencies {} for the radar license and be within range
                            given center frequencies, sampling rates and transition band.
                            """.format((self.tx_minfreq, self.tx_maxfreq),
                                       (self.rx_minfreq, self.rx_maxfreq))
                raise ExperimentException(errmsg)
            for freq_range in self.options.restricted_ranges:
                if exp_slice['txfreq'] in range(freq_range[0], freq_range[1]):
                    errmsg = """txfreq is within a restricted frequency range {}
                             """.format(freq_range)
                    raise ExperimentException(errmsg)

    def set_slice_defaults(self, exp_slice):
        """
        Set up defaults in case of some parameters being left blank.

        :param exp_slice: slice to set defaults of
        :returns slice_with_defaults: updated slice
        """

        slice_with_defaults = copy.deepcopy(exp_slice)

        # TODO future proof this by specifying tx_main and tx_int ?? or give spatial information in config
        if 'tx_antennas' not in exp_slice:
            slice_with_defaults['tx_antennas'] = [i for i in range(0,
                                                                  self.options.main_antenna_count)]
            # all possible antennas.
        if 'rx_main_antennas' not in exp_slice:
            slice_with_defaults['rx_main_antennas'] = [i for i in
                                                       range(0, self.options.main_antenna_count)]
        if 'rx_int_antennas' not in exp_slice:
            slice_with_defaults['rx_int_antennas'] = \
                [i for i in range(0, self.options.interferometer_antenna_count)]
        if 'pulse_phase_offset' not in exp_slice:
            slice_with_defaults['pulse_phase_offset'] = [0.0 for i in range(0, len(
                slice_with_defaults['pulse_sequence']))]
        if 'scanbound' not in exp_slice:
            slice_with_defaults['scanbound'] = None

        # we only have one of intn or intt because of slice checks already completed in
        # check_slice_minimum_requirements.
        if 'intt' in exp_slice:
            slice_with_defaults['intn'] = None
        elif 'intn' in exp_slice:
            slice_with_defaults['intt'] = None

        if 'acf' not in exp_slice:
            slice_with_defaults['acf'] = self.acf
            slice_with_defaults['xcf'] = self.xcf
            slice_with_defaults['acfint'] = self.acfint
        elif exp_slice['acf']:
            if 'xcf' not in exp_slice:
                slice_with_defaults['xcf'] = True
            if 'acfint' not in exp_slice:
                slice_with_defaults['acfint'] = True
        else:  # acf is False
            # TODO log that no xcf or acfint will happen if acfs are not set.
            slice_with_defaults['xcf'] = False
            slice_with_defaults['acfint'] = False

        if slice_with_defaults['acf']:
            correct_range_sep = slice_with_defaults['pulse_len'] * 1.0e-9 * speed_of_light / 2.0
            if 'range_sep' in exp_slice:
                if not math.isclose(slice_with_defaults['range_sep'], correct_range_sep, abs_tol=0.01): # range_sep in km
                    errmsg = 'range_sep = {} was set incorrectly. range_sep will be overwritten based on \
                        pulse_len, which must be equal to 1/rx_rate. The new range_sep = {}'.format(slice_with_defaults['range_sep'],
                            correct_range_sep)
                    if __debug__:  # TODO change to logging
                        print(errmsg)

            slice_with_defaults['range_sep'] = correct_range_sep
            # This is the distance travelled by the wave in the length of the pulse, divided by
            # two because it's an echo (travels there and back). In km.

            # The below check is an assumption that is made during acf calculation
            # (1 output received sample = 1 range separation)
            if not math.isclose(exp_slice['pulse_len'] * 1.0e-6, (1/self.output_rx_rate), abs_tol=0.000001):
                errmsg = 'For an experiment slice with real-time acfs, pulse length must be equal (within 1 us) to ' \
                '1/output_rx_rate to make acfs valid. Current pulse length is {} us, output rate is {}' \
                ' Hz.'.format(exp_slice['pulse_len'], self.output_rx_rate)
                raise ExperimentException(errmsg)

            if 'averaging_method' in exp_slice:
                if exp_slice['averaging_method'] in possible_averaging_methods:
                    slice_with_defaults['averaging_method'] = exp_slice['averaging_method']
                else:
                    errmsg = 'Averaging method {} not valid method. Possible methods are ' \
                             '{}'.format(exp_slice['averaging_method'], possible_averaging_methods)
                    raise ExperimentException(errmsg)
            else:
                slice_with_defaults['averaging_method'] = 'mean'

            if 'lag_table' in exp_slice:
                # Check that lags are valid
                for lag in exp_slice['lag_table']:
                    if not set(np.array(lag).flatten()).issubset(set(exp_slice['pulse_sequence'])):
                            errmsg = 'Lag {} not valid; One of the pulses does not exist in the ' \
                                     'sequence'.format(lag)
                            raise ExperimentException(errmsg)
            else:
                # build lag table from pulse_sequence
                lag_table = list(itertools.combinations(slice_with_defaults['pulse_sequence'], 2))
                lag_table.append([slice_with_defaults['pulse_sequence'][0], slice_with_defaults[
                    'pulse_sequence'][0]])  # lag 0
                # sort by lag number
                lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
                lag_table.append([slice_with_defaults['pulse_sequence'][-1], slice_with_defaults[
                    'pulse_sequence'][-1]])  # alternate lag 0
                slice_with_defaults['lag_table'] = lag_table

        else:
            # TODO log range_sep, lag_table, xcf, acfint, and averaging_method will not be used
            if __debug__:
                print('range_sep, lag_table, xcf, acfint, and averaging_method will not be used '
                              'because acf is not True.')
            if 'range_sep' not in exp_slice.keys():
                slice_with_defaults['range_sep'] = slice_with_defaults['pulse_len'] * 1.0e-9 * \
                                                      speed_of_light/2.0
            if 'lag_table' not in exp_slice.keys():
                slice_with_defaults['lag_table'] = []

            if 'averaging_method' not in exp_slice.keys():
                slice_with_defaults['averaging_method'] = None

        if 'wavetype' not in exp_slice:
            slice_with_defaults['wavetype'] = 'SINE'
        if 'seqoffset' not in exp_slice:
            slice_with_defaults['seqoffset'] = 0

        if 'comment' not in exp_slice:
            slice_with_defaults['comment'] = ''

        return slice_with_defaults

    def setup_slice(self, exp_slice):
        """
        Check slice for errors and set defaults of optional keys.

        Before adding the slice, ensure that the internal parameters are set, remove unnecessary
        keys and check values of keys that are needed, and set defaults of keys that are optional.

        The following are always able to be defaulted, so are optional:
        "tx_antennas", "rx_main_antennas", "rx_int_antennas", "pulse_phase_offset", "scanboundflag",
        "scanbound", "acf", "xcf", "acfint", "wavetype", "seqoffset", "averaging_method"

        The following are always required for processing acf, xcf, and acfint which we will assume
        we are always doing:
        "pulse_sequence", "tau_spacing", "pulse_len", "num_ranges", "first_range", "intt", "intn", "beam_angle",
        "beam_order"

        The following are required depending on slice type:
        "txfreq", "rxfreq", "clrfrqrange"

        :param: exp_slice: a slice to setup
        :return: complete_slice : a checked slice with all defaults
        """

        complete_slice = copy.deepcopy(exp_slice)

        remove_keys = []
        # None values are useless to us - if they do not exist we know they are None.
        for key, value in complete_slice.items():
            if value is None:
                remove_keys.append(key)

        for key in remove_keys:
            complete_slice.pop(key)

        self.set_slice_identifiers(complete_slice)
        self.check_slice_specific_requirements(complete_slice)
        self.check_slice_minimum_requirements(complete_slice)

        # set_slice_defaults will check for any missing values that should be given a default and
        # fill them.
        complete_slice = self.set_slice_defaults(complete_slice)

        # Wavetables are currently None for sine waves, instead just use a sampling freq in rads/sample.
        # wavetype = 'SINE' is set in set_slice_defaults if not given.
        complete_slice['iwavetable'], complete_slice['qwavetable'] = get_wavetables(complete_slice['wavetype'])

        errors = self.check_slice(complete_slice)

        if errors:
            raise ExperimentException(errors)

        return complete_slice

    def self_check(self):
        """
        Check that the values in this experiment are valid.

        Checks all slices.
        """

        if self.num_slices < 1:
            errmsg = "Invalid num_slices less than 1"
            raise ExperimentException(errmsg)

        # TODO: check if self.cpid is not unique - incorporate known cpids from git repo
        # TODO: use pygit2 for this

        # check that the number of slices can be accommodated by the decimation scheme.
        for stage in self.decimation_scheme.stages:
            power_2 = 0
            while (2 ** power_2 < len(stage.filter_taps)):
                power_2 += 1
            effective_length = 2 ** power_2
            if effective_length * self.num_slices > self.options.max_number_of_filter_taps_per_stage:
                errmsg = "Length of filter taps once zero-padded ({}) in decimation stage {} with \
                    this many slices ({}) is too large for GPU max {}".format(len(stage.filter_taps),
                        stage.stage_num, self.num_slices, self.options.max_number_of_filter_taps_per_stage)

        # run check_slice on all slices. Check_slice is a full check and can be done on a slice at
        # any time after setup. We run it now in case the user has changed something
        # inappropriately (ie, any way other than using edit_slice, add_slice, or del_slice).
        # "Private" instance variables with leading underscores are not actually private in
        # python they just have a bit of a mangled name so they are not readily available but give
        # the user notice that they should be left alone. If the __slice_dict has been changed
        # improperly, we should check it for problems here.
        for a_slice in self.slice_ids:
            selferrs = self.check_slice(self.slice_dict[a_slice])
            if not selferrs:
                # If returned error dictionary is empty
                continue
            errmsg = "Self Check Errors Occurred with slice Number : {} \nSelf \
                Check Errors are : {}".format(a_slice, selferrs)
            raise ExperimentException(errmsg)

        if __debug__:
            print("No Self Check Errors. Continuing...")

    def check_slice(self, exp_slice):
        """
        Check the slice for errors.

        This is the first test of the dictionary in the experiment done to ensure values in this
        slice make sense. This is a self-check to ensure the parameters (for example, txfreq,
        antennas) are appropriate. All fields should be full at this time (whether filled by the
        user or given default values in set_slice_defaults). This was built to be useable at
        any time after setup.
        :param: exp_slice: a slice to check
        :raise: ExperimentException: When necessary parameters do not exist or = None (would have
        to have been overridden by the user for this, as defaults all set when this runs).
        """
        error_list = []

        options = self.options

        for param in self.slice_keys:
            if param not in exp_slice.keys():
                if param == 'txfreq' and (exp_slice['clrfrqflag'] or exp_slice['rxonly']):
                    pass
                elif param == 'rxfreq' and not exp_slice['rxonly']:
                    pass
                elif param == 'clrfrqrange' and not exp_slice['clrfrqflag']:
                    pass
                else:
                    errmsg = "Slice {} is missing Necessary Parameter {}".format(
                        exp_slice['slice_id'], param)
                    raise ExperimentException(errmsg)
            if param is None:
                pass  # TODO may want to check certain params are not None

        for param in exp_slice.keys():
            if param not in self.slice_keys and param not in self.__hidden_slice_keys:
                error_list.append("Slice {} has A Parameter that is not Used: {} = {}". \
                    format(exp_slice['slice_id'], param, exp_slice[param]))

        # TODO : tau_spacing needs to be an integer multiple of pulse_len in ros - is there a max ratio
        # allowed for pulse_len/tau_spacing ? Add this check and add check for each slice's tx duty-cycle
        # and make sure we aren't transmitting the entire time after combination with all slices

        if len(exp_slice['tx_antennas']) > options.main_antenna_count:
            error_list.append("Slice {} Has Too Many Main TX Antenna Channels {} Greater than Config {}" \
                .format(exp_slice['slice_id'], len(exp_slice['tx_antennas']),
                        options.main_antenna_count))
        if len(exp_slice['rx_main_antennas']) > options.main_antenna_count:
            error_list.append("Slice {} Has Too Many Main RX Antenna Channels {} Greater than Config {}" \
                .format(exp_slice['slice_id'], len(exp_slice['rx_main_antennas']),
                        options.main_antenna_count))
        if len(exp_slice['rx_int_antennas']) > options.interferometer_antenna_count:
            error_list.append("Slice {} Has Too Many RX Interferometer Antenna Channels {} " \
                               "Greater than Config {}".format(
                                    exp_slice['slice_id'],
                                    len(exp_slice['rx_int_antennas']),
                                    options.interferometer_antenna_count))

        # Check if the antenna identifier number is greater than the config file's
        # maximum antennas for all three of tx antennas, rx antennas and rx int antennas
        # Also check for duplicates
        if max(exp_slice['tx_antennas']) >= options.main_antenna_count:
            error_list.append("Slice {} Specifies Main Array Antenna Numbers Over Config " \
                               "Max {}" .format(exp_slice['slice_id'],
                                                options.main_antenna_count))

        if list_tests.has_duplicates(exp_slice['tx_antennas']):
            error_list.append("Slice {} TX Main Antennas Has Duplicate Antennas".format(
                exp_slice['slice_id']))

        for i in range(len(exp_slice['rx_main_antennas'])):
            if exp_slice['rx_main_antennas'][i] >= options.main_antenna_count:
                error_list.append("Slice {} Specifies Main Array Antenna Numbers Over Config " \
                                   "Max {}" .format(exp_slice['slice_id'],
                                                    options.main_antenna_count))

        if list_tests.has_duplicates(exp_slice['rx_main_antennas']):
            error_list.append("Slice {} RX Main Antennas Has Duplicate Antennas".format(
                exp_slice['slice_id']))

        for i in range(len(exp_slice['rx_int_antennas'])):
            if exp_slice['rx_int_antennas'][i] >= options.interferometer_antenna_count:
                error_list.append("Slice {} Specifies Interferometer Array Antenna Numbers Over " \
                                   "Config Max {}".format(exp_slice['slice_id'],
                                                          options.interferometer_antenna_count))

        if list_tests.has_duplicates(exp_slice['rx_int_antennas']):
            error_list.append("Slice {} RX Interferometer Antennas Has Duplicate Antennas".format(
                exp_slice['slice_id']))

        # Check if the pulse_sequence is not increasing, which would be an error
        if not list_tests.is_increasing(exp_slice['pulse_sequence']):
            error_list.append("Slice {} pulse_sequence Not Increasing".format(
                exp_slice['slice_id']))

        # Check that pulse_len and tau_spacing make sense (values in us)
        if exp_slice['pulse_len'] > exp_slice['tau_spacing']:
            error_list.append("Slice {} Pulse Length Greater than tau_spacing".format(
                exp_slice['slice_id']))
        if exp_slice['pulse_len'] < self.options.minimum_pulse_length and \
                        exp_slice['pulse_len'] <= 2 * self.options.pulse_ramp_time * \
                        10.0e6:
            error_list.append("Slice {} Pulse Length Too Small".format(
                exp_slice['slice_id']))
        if exp_slice['tau_spacing'] < self.options.minimum_tau_spacing_length:
            error_list.append("Slice {} Multi-Pulse Increment Too Small".format(
                exp_slice['slice_id']))
        if not math.isclose((exp_slice['tau_spacing'] * self.output_rx_rate % 1.0), 0.0, abs_tol=0.0001):
            error_list.append('Slice {} Correlation lags will be off because tau_spacing {} us is not a '\
                'multiple of the output rx sampling period (1/output_rx_rate {} Hz).'.format(
                    exp_slice['slice_id'], exp_slice['tau_spacing'], self.output_rx_rate))

        # check intn and intt make sense given tau_spacing, and pulse_sequence.
        if exp_slice['pulse_sequence']:  # if not empty
            # Sequence length is length of pulse sequence plus the scope sync delay time.
            # TODO this is an old check and seqtime now set in sequences class, update.
            seq_len = exp_slice['tau_spacing'] * (exp_slice['pulse_sequence'][-1]) \
                      + (exp_slice['num_ranges'] + 19 + 10) * exp_slice['pulse_len']  # us

            if exp_slice['intt'] is None and exp_slice['intn'] is None:
                # both are None and we are not rx - only
                error_list.append("Slice {} Has Transmission but no Intt or IntN".format(
                    exp_slice['slice_id']))

            if exp_slice['intt'] is not None and exp_slice['intn'] is not None:
                error_list.append("Slice {} Choose Either Intn or Intt to be the Limit " \
                                            "for Number of Integrations in an Integration Period.".\
                    format(exp_slice['slice_id']))

            if exp_slice['intt'] is not None:
                if seq_len > (exp_slice['intt'] * 1000):  # seq_len in us, so multiply intt
                                                          # (ms) by 1000 to compare in us
                    error_list.append("Slice {} : Pulse Sequence is Too Long for Integration " \
                                         "Time Given".format(exp_slice['slice_id']))

        if not exp_slice['pulse_sequence']:
            if exp_slice['txfreq']:
                error_list.append("Slice {} Has Transmission Frequency but no" \
                                            "Pulse Sequence defined".format(
                    exp_slice['slice_id']))

        if list_tests.has_duplicates(exp_slice['beam_angle']):
            error_list.append("Slice {} Beam Angles Has Duplicate Directions".format(
                exp_slice['slice_id']))

        if not list_tests.is_increasing(exp_slice['beam_angle']):
            error_list.append("Slice {} beam_angle Not Increasing Clockwise (E of N " \
                                      "is positive)".format(exp_slice['slice_id']))

        # Check if the list of beams to transmit on is empty
        if not exp_slice['beam_order']:
            error_list.append("Slice {} Beam Order Scan Empty".format(
                exp_slice['slice_id']))

        # Check that the beam numbers in the beam_order exist
        for bmnum in exp_slice['beam_order']:
            if isinstance(bmnum, int):
                if bmnum >= len(exp_slice['beam_angle']):
                    error_list.append("Slice {} Scan Beam Number {} DNE".format(
                        exp_slice['slice_id'], bmnum))
            elif isinstance(bmnum, list):
                for imaging_bmnum in bmnum:
                    if imaging_bmnum >= len(exp_slice['beam_angle']):
                        error_list.append("Slice {} Scan Beam Number {} DNE".format(
                            exp_slice['slice_id'], bmnum))

        # check scan boundary not less than minimum required scan time.
        if exp_slice['scanbound']:
            if not exp_slice['intt']:
                error_list.append("Slice {} must have intt enabled to use scanbound".format(
                        exp_slice['slice_id']))
            elif any(i<0 for i in exp_slice['scanbound']):
                error_list.append("Slice {} scanbound times must be non-negative".format(
                        exp_slice['slice_id']))
            elif (len(exp_slice['scanbound']) > 1 and
                not all(i<j for i,j in zip(exp_slice['scanbound'], exp_slice['scanbound'][1:]))):
                error_list.append("Slice {} scanbound times must be increasing".format(
                        exp_slice['slice_id']))
            else:
                # Check if any scanbound times are shorter than the intt.
                if len(exp_slice['scanbound']) == 1:
                    if exp_slice['scanbound'][0] * 1000 < exp_slice['intt']:
                        error_list.append("Slice {} intt longer than "
                            "scanbound times".format(exp_slice['slice_id']))
                else:
                    for i in range(len(exp_slice['scanbound']) - 1):
                        if ((exp_slice['scanbound'][i+1] - exp_slice['scanbound'][i]) * 1000 <
                            exp_slice['intt']):
                            error_list.append("Slice {} intt longer than "
                                "scanbound times".format(exp_slice['slice_id']))
                            break

        # TODO other checks

        if exp_slice['wavetype'] != 'SINE':
            error_list.append("Slice {} wavetype of {} currently not supported".format(
                exp_slice['slice_id'], exp_slice['wavetype']))

        return error_list

    def get_slice_interfacing(self, slice_id):
        """
        Check the experiment's interfacing dictionary for all interfacing that pertains to a
        given slice, and return the interfacing information in a dictionary.
        :param slice_id: Slice ID to search the interface dictionary for.
        :return: interfacing dictionary for the slice.
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

