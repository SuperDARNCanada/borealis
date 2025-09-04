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
import inspect
import os
from pathlib import Path

# third-party
import numpy as np
import re
import structlog

# local
from utils.options import Options
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.experiment_slice import (
    ExperimentSlice,
    slice_key_set,
    hidden_key_set,
)
from experiment_prototype.interface_classes.scans import Scan
from experiment_prototype.interface_classes.interface_class_base import (
    InterfaceClassBase,
)

# Obtain the module name that imported this log_config
caller = Path(inspect.stack()[-1].filename)
module_name = caller.name.split(".")[0]
log = structlog.getLogger(module_name)

BOREALISPATH = os.environ["BOREALISPATH"]

interface_types = tuple(["SCAN", "AVEPERIOD", "SEQUENCE", "CONCURRENT"])
""" Interfacing in this case refers to how two or more slices are meant to be run together.
The following types of interfacing between slices are possible, arranged from highest level
of experiment building-block to the lowest level:

1. **SCAN**

    The scan-by-scan interfacing allows for slices to run a scan of one slice, followed by a scan of the
    second. The scan mode of interfacing typically means that the slice will cycle through all of its
    beams before switching to another slice.

    If any slice in the experiment specifies a value for ``scanbound``, all other slices must also specify
    a value for ``scanbound``. The values do not have to be the same, however.

2. **AVEPERIOD**

    This type of interfacing allows for one slice to run its averaging period (also known as integration
    time or integration period), before switching to another slice's averaging period. This type of
    interface effectively creates an interleaving scan where the scans for multiple slices are run 'at
    the same time', by interleaving the averaging periods.

    Slices which are interfaced in this manner must share:

    * the same ``scanbound`` value.

3. **SEQUENCE**

    Sequence interfacing allows for pulse sequences defined in the slices to alternate between each
    other within a single averaging period. It's important to note that data from a single slice is
    averaged only with other data from that slice. So in this case, the averaging period is running two
    slices and can produce two averaged datasets, but the sequences within the averaging period are
    interleaved.

    Slices which are interfaced in this manner must share:

    * the same ``scanbound`` value.
    * the same ``intt`` or ``intn`` value.
    * the same ``rx_beam_order`` length (scan length).
    * the same ``txctrfreq`` value.
    * the same ``rxctrfreq`` value.

4. **CONCURRENT**

    Concurrent interfacing allows for pulse sequences to be run together concurrently. Slices will have
    their pulse sequences summed together so that the data transmits at the same time. For example,
    slices of different frequencies can be mixed simultaneously, and slices of different pulse sequences
    can also run together at the cost of having more blanked samples. When slices are interfaced in this
    way the radar is truly transmitting and receiving the slices simultaneously.

    Slices which are interfaced in this manner must share:

    * the same ``scanbound`` value.
    * the same ``intt`` or ``intn`` value.
    * the same ``rx_beam_order`` length (scan length).
    * the same ``txctrfreq`` value.
    * the same ``rxctrfreq`` value.
    * the same ``decimation_scheme``.

"""

possible_scheduling_modes = frozenset(["common", "special", "discretionary"])
default_rx_bandwidth = 5.0e6
transition_bandwidth = 750.0e3


class ExperimentPrototype:
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

    * slice_dict:   modifiable only using the add_slice, edit_slice, and del_slice methods.
    * interface:    modifiable using the add_slice, edit_slice, and del_slice methods, or by
                    updating the interface dict directly.

    :param  cpid:               Unique id necessary for each control program (experiment). Cannot be
                                changed after instantiation.
    :type   cpid:               int
    :param  rx_bandwidth:       The desired bandwidth for the experiment. Directly determines rx
                                sampling rate of the USRPs. Cannot be changed after instantiation.
                                Default 5.0 MHz.
    :type   rx_bandwidth:       float
    :param  tx_bandwidth:       The desired tx bandwidth for the experiment. Directly determines tx
                                sampling rate of the USRPs. Cannot be changed after instantiation.
                                Default 5.0 MHz.
    :type  tx_bandwidth:        float
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

    def __init__(
        self,
        cpid,
        rx_bandwidth=default_rx_bandwidth,
        tx_bandwidth=5.0e6,
        comment_string="",
    ):
        if not isinstance(cpid, int):
            errmsg = "CPID must be a unique int"
            raise ExperimentException(errmsg)
        if cpid > np.iinfo(np.int16).max:
            errmsg = "CPID must be representable by a 16-bit signed integer"
            raise ExperimentException(errmsg)
        # Quickly check for uniqueness with a search in the experiments directory first taking care
        # not to look for CPID in any experiments that are just tests (located in the testing
        # directory)
        experiment_files_list = list(
            Path(f"{BOREALISPATH}/src/borealis_experiments/").glob("*.py")
        )
        self.__experiment_name = self.__class__.__name__
        # TODO use this to check the cpid is correct using pygit2, or __class__.__module__ for module name
        # TODO replace below cpid local uniqueness check with pygit2 or some reference
        #  to a database to to ensure CPID uniqueness and to ensure CPID is entered in the database
        #  for this experiment (this CPID is unique AND its correct given experiment name)
        cpid_list = {}
        for experiment_file in experiment_files_list:
            with open(experiment_file) as file_to_search:
                for line in file_to_search:
                    # Find the name of the class in the file and break if it matches this class
                    experiment_class_name = re.findall(
                        "class.*\(ExperimentPrototype\):", line
                    )
                    if experiment_class_name:
                        # Parse out just the name from the experiment, format will be like this:
                        # ['class IBCollabMode(ExperimentPrototype):']
                        atomic_class_name = (
                            experiment_class_name[0].split()[1].split("(")[0]
                        )
                        if self.__experiment_name == atomic_class_name:
                            break

                    # Find any lines that have 'cpid = [integer]'
                    existing_cpid = re.findall("cpid.?=.?[0-9]+", line)
                    if existing_cpid:
                        cpid_list[existing_cpid[0].split("=")[1].strip()] = (
                            experiment_file
                        )

        if str(cpid) in cpid_list.keys():
            errmsg = f"CPID must be unique. {cpid} is in use by another local experiment {cpid_list[str(cpid)]}"
            raise ExperimentException(errmsg)
        if cpid <= 0:
            errmsg = (
                "The CPID should be a positive number in the experiment. If the embargo"
                " flag is set, then borealis will configure the CPID to be negative to ."
                " indicate the data is to be embargoed for one year."
            )
            raise ExperimentException(errmsg)

        self.__options = (
            Options()
        )  # Load the config, hardware, and restricted frequency data
        self.__cpid = cpid
        self.__scheduling_mode = "unknown"
        if comment_string is None:
            comment_string = ""
        self.__comment_string = comment_string
        self.__slice_dict = {}
        self.__new_slice_id = 0

        self.__txrate = float(tx_bandwidth)  # sampling rate, samples per sec, Hz.
        self.__rxrate = float(rx_bandwidth)  # sampling rate for rx in samples per sec

        # Transmitting is possible in the range of txctrfreq +/- (txrate/2) because we have iq data
        # Receiving is possible in the range of rxctrfreq +/- (rxrate/2)
        if self.txrate > self.options.max_tx_sample_rate:
            errmsg = (
                f"Experiment's transmit bandwidth is too large: {self.txrate} greater than "
                f"max {self.options.max_tx_sample_rate}."
            )
            raise ExperimentException(errmsg)
        if self.rxrate > self.options.max_rx_sample_rate:
            errmsg = (
                f"Experiment's receive bandwidth is too large: {self.rxrate} greater than "
                f"max {self.options.max_rx_sample_rate}."
            )
            raise ExperimentException(errmsg)
        if round(self.options.usrp_master_clock_rate / self.txrate, 3) % 2.0 != 0.0:
            errmsg = (
                f"Experiment's transmit bandwidth {self.txrate} is not possible as it must be an "
                f"integer divisor of USRP master clock rate {self.options.usrp_master_clock_rate}"
            )
            raise ExperimentException(errmsg)
        if round(self.options.usrp_master_clock_rate / self.rxrate, 3) % 2.0 != 0.0:
            errmsg = (
                f"Experiment's receive bandwidth {self.rxrate} is not possible as it must be an "
                f"integer divisor of USRP master clock rate {self.options.usrp_master_clock_rate}"
            )
            raise ExperimentException(errmsg)

        # This is experiment-wide transmit metadata necessary to build the pulses. This data
        # cannot change within the experiment and is used in the scan classes to pass information
        # to where the samples are built.
        self.__transmit_metadata = {
            "tx_main_antennas": self.options.tx_main_antennas,
            "rx_main_antennas": self.options.rx_main_antennas,
            "rx_intf_antennas": self.options.rx_intf_antennas,
            "main_antenna_locations": self.options.main_antenna_locations,
            "intf_antenna_locations": self.options.intf_antenna_locations,
            "main_antenna_count": self.options.main_antenna_count,
            "intf_antenna_count": self.options.intf_antenna_count,
            "main_antenna_spacing": self.options.main_antenna_spacing,
            "intf_antenna_spacing": self.options.intf_antenna_spacing,
            "tr_window_time": self.options.tr_window_time,
            "pulse_ramp_time": self.options.pulse_ramp_time,
            "max_usrp_dac_amplitude": self.options.max_usrp_dac_amplitude,
            "rx_sample_rate": self.rxrate,
            "min_pulse_separation": self.options.min_pulse_separation,
            "rxrate": self.rxrate,
            "txrate": self.txrate,
            "intf_offset": self.options.intf_offset,
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
        self.__running_experiment = None  # this will be of InterfaceClassBase type

        # This is used for adding and editing slices
        self.__slice_restrictions = {
            "tx_bandwidth": self.tx_bandwidth,
            "rx_bandwidth": self.rx_bandwidth,
            "transition_bandwidth": transition_bandwidth,
        }

    __slice_keys = slice_key_set
    __hidden_slice_keys = hidden_key_set

    @property
    def cpid(self):
        """
        This experiment's CPID (control program ID, a term that comes from ROS).

        :returns:   cpid - read-only, only modified at runtime by set_scheduling_mode() to set to a
                    negative value if the embargo flag was set in the schedule
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

    def _embargo_files(self, embargo_flag: bool):
        """
        Sets the cpid negative, signifying that the data generated is embargoed for one year by
        the host institution. Should only be called by the experiment handler after initializing
        the user's class.

        :param embargo_flag:    Flag to embargo the files
        :type  embargo_flag:    bool
        """
        if embargo_flag:
            self.__cpid = -1 * self.__cpid

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
        else:
            errmsg = (
                f"Scheduling mode {scheduling_mode} set by experiment handler is not "
                f" a valid mode: {possible_scheduling_modes}"
            )
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
                errmsg = (
                    f"Interface value with slice {sibling_slice_id}: {interface_value} not "
                    f"valid. Types available are: {interface_types}"
                )
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
                        errmsg = (
                            f"Cannot add slice: the interfacing_dict set interfacing to an unknown "
                            f"slice {sibling_slice_id} not in slice ids {self.slice_ids}"
                        )
                        raise ExperimentException(errmsg)
                try:
                    closest_sibling = max(
                        interfacing_dict.keys(),
                        key=lambda k: interface_types.index(interfacing_dict[k]),
                    )
                except ValueError as e:  # cannot find interface type in list
                    errmsg = (
                        f"Interface types must be of valid types {interface_types}."
                    )
                    raise ExperimentException(errmsg) from e
                closest_interface_value = interfacing_dict[closest_sibling]
                closest_interface_rank = interface_types.index(closest_interface_value)
            else:
                # the user provided no keys. The default is therefore 'SCAN'
                # with all keys so the closest will be 'SCAN' (the furthest possible interface_type)
                closest_sibling = self.slice_ids[0]
                closest_interface_value = "SCAN"
                closest_interface_rank = interface_types.index(closest_interface_value)

            # now populate a full_interfacing_dict based on the closest sibling's interface values
            # and knowing how we interface with that sibling. this is the only correct interfacing
            # given the closest interfacing.
            full_interfacing_dict[closest_sibling] = closest_interface_value
            for (
                sibling_slice_id,
                siblings_interface_value,
            ) in self.get_slice_interfacing(closest_sibling).items():
                if (
                    interface_types.index(siblings_interface_value)
                    >= closest_interface_rank
                ):
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
                    siblings_interface_value = self.get_slice_interfacing(
                        closest_sibling
                    )[sibling_slice_id]
                    errmsg = (
                        f"The interfacing values of new slice cannot be reconciled. Interfacing "
                        f"with slice {closest_sibling}: {closest_interface_value} and with "
                        f"slice {sibling_slice_id}: {interface_value} does not make sense with "
                        f"existing interface between slices of "
                        f"{([sibling_slice_id, closest_sibling].sort())}: {siblings_interface_value}"
                    )
                    raise ExperimentException(errmsg)

        return full_interfacing_dict

    def __update_slice_interfacing(self):
        """
        Internal slice interfacing updater. This should only be used internally when slice
        dictionary is changed, to update all of the slices' interfacing dictionaries.
        """
        for slice_id in self.slice_ids:
            self.__slice_dict[slice_id].slice_interfacing = self.get_slice_interfacing(
                slice_id
            )

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
            errmsg = f"Attempt to add a slice failed - {exp_slice} is not a dictionary of slice parameters"
            raise ExperimentException(errmsg)
            # TODO multiple types of Exceptions so they can be caught by the experiment in these
            #  add_slice, edit_slice, del_slice functions (and handled specifically)
        if interfacing_dict is None:
            interfacing_dict = {}

        add_slice_id = exp_slice["slice_id"] = self.new_slice_id
        # each added slice has a unique slice id, even if previous slices have been deleted.
        exp_slice["cpid"] = self.cpid

        # Now we setup the slice which will check minimum requirements and set defaults, and then
        # will complete a check_slice and raise any errors found.
        new_exp_slice = ExperimentSlice(**exp_slice, **self.__slice_restrictions)

        # now check that the interfacing values make sense before appending.
        full_interfacing_dict = self.check_new_slice_interfacing(interfacing_dict)
        for sibling_slice_id, interface_value in full_interfacing_dict.items():
            # sibling_slice_id < new slice id so this maintains interface list requirement.
            self.__interface[(sibling_slice_id, exp_slice["slice_id"])] = (
                interface_value
            )

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
            del self.slice_dict[remove_slice_id]
        except (KeyError, TypeError) as e:
            errmsg = (
                f"Cannot remove slice id {remove_slice_id} : it does not exist in slice "
                "dictionary"
            )
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
            edited_slice = copy.deepcopy(self.slice_dict[edit_slice_id])
        except (KeyError, TypeError):
            # the edit_slice_id is not an index in the slice_dict
            errmsg = f"Trying to edit {edit_slice_id} but it does not exist in Slice_IDs list."
            raise ExperimentException(errmsg)

        for edit_slice_param, edit_slice_value in slice_params_to_edit.items():
            if edit_slice_param in self.slice_keys:
                setattr(edited_slice, edit_slice_param, edit_slice_value)
            else:
                errmsg = (
                    f"Cannot edit slice ID {edit_slice_id}: {edit_slice_param} is not a valid"
                    " slice parameter"
                )
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

        except ExperimentException as err:
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

            log.error("Slice has errors, unable to add to experiment", errors=err)

            return edit_slice_id

    def __repr__(self):
        represent = (
            f"self.cpid = {self.cpid}\n"
            f"self.num_slices = {self.num_slices}\n"
            f"self.slice_ids = {self.slice_ids}\n"
            f"self.slice_keys = {self.slice_keys}\n"
            f"self.options = {self.options.__str__()}\n"
            f"self.txrate = {self.txrate}\n"
            f"self.slice_dict = {self.slice_dict}\n"
            f"self.interface = {self.interface}\n"
        )
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

        # Set any unset center frequencies in the experiment
        self.set_center_frequencies()

        # TODO consider removing scan_objects from init and making a new Experiment class to inherit
        # from InterfaceClassBase and having all of this included in there. Then would only need to
        # pass the running experiment to the radar control (would be returned from build_scans)
        self.__running_experiment = InterfaceClassBase(
            self.slice_ids, self.slice_dict, self.interface, self.transmit_metadata
        )

        self.__scan_objects = []
        for params in self.__running_experiment.prep_for_nested_interface_class():
            self.__scan_objects.append(Scan(*params))

        for scan in self.__scan_objects:
            if scan.scanbound is not None:
                self.__scanbound = True

        if self.__scanbound:
            try:
                self.__scan_objects = sorted(
                    self.__scan_objects, key=lambda input_scan: input_scan.scanbound[0]
                )
            except (IndexError, TypeError) as e:  # scanbound is None in some scans
                errmsg = "If one slice has a scanbound, they all must to avoid up to minute-long downtimes."
                raise ExperimentException(errmsg) from e

        max_num_concurrent_slices = 0
        for scan in self.__scan_objects:
            for aveperiod in scan.aveperiods:
                for seq in aveperiod.sequences:
                    if len(seq.slice_ids) > max_num_concurrent_slices:
                        max_num_concurrent_slices = len(seq.slice_ids)

        log.verbose(f"Number of Scan types: {len(self.__scan_objects)}")
        log.verbose(
            f"Number of AveragingPeriods in Scan #1: {len(self.__scan_objects[0].aveperiods)}"
        )
        log.verbose(
            f"Number of Sequences in Scan #1, Averaging Period #1: "
            f"{len(self.__scan_objects[0].aveperiods[0].sequences)}"
        )
        log.verbose(
            f"Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1: "
            f"{len(self.__scan_objects[0].aveperiods[0].sequences[0].slice_dict)}"
        )
        log.verbose(f"Max concurrent slices: {max_num_concurrent_slices}")

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

        log.info("No Self Check Errors. Continuing...")

    @staticmethod
    def compare_freq_to_band(freq_list, center_freq_band, valid=True, down_shift=False):
        """
        Checks if a list of frequencies fall within a frequency band and returns a boolean
        indicating the result. Since the radar should not operate in a 50 kHz band centered
        around the center frequency, the center band is split into two bands above and below
        the center frequency. If the frequency list under test is a range (eg: cfs_range)
        then the function will also check if the frequency range overlaps with the 50kHz
        band around the center freq that should not be used.

        :param  freq_list:      List of frequencies to check
        :type   freq_list:      list, 1 or 2 elements long
        :param  center_freq_band:   The frequency bands a corresponding to a particular
                                    center frequency. Split into and upper and lower band
                                    because of the 50kHZ band around the center frequency
                                    that should not be used for operation.
        :type   center_freq_band:   list of lists [[band 1], [band 2]]
        :param  valid:          Tracks if the frequencies meet the conditions
        :type   valid:          boolean
        :param  down_shift:     Indicates if the freq_list encompasses a null in center_freq_band
        :type   down_shift:     boolean

        :return: valid, down_shift
        :rtype:  boolean, boolean
        """

        for freq in freq_list:
            if not (
                center_freq_band[0][0] <= freq <= center_freq_band[0][1]
                or center_freq_band[1][0] <= freq <= center_freq_band[1][1]
            ):
                valid = False
        if len(freq_list) == 2:
            if valid:
                if (
                    freq_list[0] <= center_freq_band[0][1]
                    and freq_list[1] >= center_freq_band[1][0]
                ):
                    down_shift = True
                    valid = False
        return valid, down_shift

    def calculate_center_freq(self, chosen_freq):
        """
        Calculates the closest actual center frequency based on the USRP
        device clock and the desired center frequency.

        :param  chosen_freq: the center frequency desired
        :type   chosen_freq: float

        :return actual_center_freq: valid center frequency for USRP device
        :rtype  actual_center_freq: float
        """
        clock_multiples = self.options.usrp_master_clock_rate / 2**32
        clock_divider = np.ceil(chosen_freq * 1e3 / clock_multiples)
        return (clock_divider * clock_multiples) / 1e3

    def set_center_frequencies(self):
        """
        Determines and sets a tx and rx center frequency for any slices the user did not set manually.
        First the slices with unset center frequencies will be determined and compiled. Then a while loop
        will be entered. It also creates a list of all CONCURRENT and SEQUENCE interfaces slices that must
        have the same center frequencies.

        While not all the slices in the experiment have center frequencies, the lowest
        unset slice frequency will be selected and a center frequency that just satisfies that slice will
        be picked. The loop then goes through all the other unset slices to find which other slices are also
        satisfied by the center frequency. In the event that a slice has a frequency range that overlaps
        with the band around the center frequency that should not be used for operation, instead of setting
        the center frequency for all slices, the loop will shift the center frequency down a bit and then
        go back through the loop again. Once no frequency ranges overlap with the center frequency, then
        all slices that are satisfied by the current selected center frequency will have the tx and rx
        center frequencies set.

        :raises ExperimentException:    if while loop continues for too long (10000 iterations)
        """
        # Initialize parameters
        slice_freq, slice_ctr_freq = dict(), dict()
        accounted_for, strict_slices = set(), set()

        tx_null_band = 50  # kHz
        lowest_freq_slice, counter, center_freq = 0, 0, 0  # initial value unused
        tune_bandwidth = (self.tx_bandwidth - transition_bandwidth * 2) / 1000  # kHz

        down_shift = False

        # Create list of slices that need center frequencies and the freq used by the slice
        for slice_id in self.slice_ids:
            if (
                self.slice_dict[slice_id].txctrfreq is None
                and self.slice_dict[slice_id].rxctrfreq is not None
            ):
                self.slice_dict[slice_id].txctrfreq = self.slice_dict[
                    slice_id
                ].rxctrfreq
                accounted_for.union({slice_id})
            elif (
                self.slice_dict[slice_id].txctrfreq is not None
                and self.slice_dict[slice_id].rxctrfreq is None
            ):
                self.slice_dict[slice_id].rxctrfreq = self.slice_dict[
                    slice_id
                ].txctrfreq
                accounted_for.union({slice_id})
            elif (
                self.slice_dict[slice_id].txctrfreq is None
                and self.slice_dict[slice_id].rxctrfreq is None
            ):
                if self.slice_dict[slice_id].freq is None:
                    slice_freq[slice_id] = self.slice_dict[slice_id].cfs_range
                else:
                    slice_freq[slice_id] = [self.slice_dict[slice_id].freq]
            else:
                # Both tx and rx center frequency are already set
                accounted_for.union({slice_id})

        # If no frequencies need to be set
        if len(slice_freq) == 0:
            return

        # determine which slices must have the same center frequencies
        for slice_id in self.slice_ids:
            interfacing_dict = self.get_slice_interfacing(slice_id)
            interfacing_dict[slice_id] = (
                "CONCURRENT"  # add current slice to interfacing dict
            )
            keys = list(interfacing_dict.keys())
            keys.append(slice_id)
            strict_slices.add(
                frozenset(
                    [
                        x
                        for x in keys
                        if interfacing_dict[x] == "CONCURRENT"
                        or interfacing_dict[x] == "SEQUENCE"
                    ]
                )
            )

        # Begin calculating the center frequencies
        while not len(self.slice_ids) == len(accounted_for):
            # if a new slice should be picked, pick the lowest freq slice that
            # has not had a center freq determined yet.
            down_shifting_slice = set()
            if not down_shift:
                unset_slices = {
                    k: slice_freq[k]
                    for k in slice_freq.keys()
                    if k not in accounted_for
                }
                lowest_freq_slice = min(unset_slices, key=unset_slices.get)

                # determine center freq using selected slice id freq
                center_freq = slice_freq[lowest_freq_slice][0] + tune_bandwidth / 2

            # track which slices work with chosen center freq
            current_attempt = {lowest_freq_slice}
            center_freq_band = [
                [center_freq - tune_bandwidth / 2, center_freq - tx_null_band / 2],
                [center_freq + tx_null_band / 2, center_freq + tune_bandwidth / 2],
            ]

            for slice_id in slice_freq.keys():
                # only test slices not already matched to a center freq
                if slice_id not in current_attempt:
                    concurrent = False
                    concurrent_set = []
                    for strict_set in strict_slices:
                        if slice_id in strict_set:
                            # If the slice id must have the same center freq as another slice
                            concurrent = True
                            concurrent_set = strict_set
                            break

                    # check if slice is satisfied by current proposed center freq band
                    # If a freq was in the 50 kHz null in the middle of the band, attempt downshifting
                    # the center freq
                    valid = True
                    down_shift_trigger = False
                    if not concurrent:
                        valid, down_shift_trigger = self.compare_freq_to_band(
                            slice_freq[slice_id], center_freq_band, valid
                        )
                        if valid:
                            current_attempt.add(slice_id)
                        if down_shift_trigger:
                            # Store slices that have freq range crossing over the
                            # 50 kHz band around the center freq.
                            down_shifting_slice.add(slice_id)
                            down_shift = True
                    else:
                        for con_slice in concurrent_set:
                            valid, down_shift_trigger = self.compare_freq_to_band(
                                slice_freq[con_slice],
                                center_freq_band,
                                valid,
                                down_shift,
                            )
                            if down_shift_trigger:
                                down_shifting_slice.add(slice_id)
                                down_shift = True
                        if valid:
                            for con_slice in concurrent_set:
                                current_attempt.add(con_slice)

            # Now that all slices have been checked for this lowest freq slice, set center freqs and then
            # continue the loop if not all slices have been accounted for
            if down_shifting_slice.issubset(current_attempt):
                # Stop downshifting the center freq if all down_shifting slices are in current attempt
                down_shift = False

            if (
                down_shift
                and center_freq <= slice_freq[lowest_freq_slice][0] - tune_bandwidth / 2
            ):
                # if downshifting has taken the band below the starting slice freq, stop shifting
                center_freq += tx_null_band * 2
                down_shift = False
                current_attempt = {lowest_freq_slice}

            if down_shift:
                # if a frequency range was found to overlap with the 50kHz band around the
                # center frequency, trying shifting the center freq down by the band and repeat
                # the while loop until the down_shift condition is no longer triggered.
                center_freq = center_freq - tx_null_band
            else:
                # update slices accounted for and assign center freq to slices in current attempt
                accounted_for = accounted_for.union(current_attempt)
                for slice_id in current_attempt:
                    if slice_id not in slice_ctr_freq.keys():
                        slice_ctr_freq[slice_id] = center_freq
                down_shift = False

            counter += 1
            if counter > 10000:
                # Abort if while loop goes for too long
                raise ExperimentException(
                    "Experiment handler failed to find valid center frequencies for "
                    "experiment. Consider changing operating frequencies or setting "
                    "center frequencies manually."
                )

        for slice_id in self.slice_ids:
            # set the center frequencies
            self.slice_dict[slice_id].txctrfreq = self.calculate_center_freq(
                slice_ctr_freq[slice_id]
            )
            self.slice_dict[slice_id].rxctrfreq = self.calculate_center_freq(
                slice_ctr_freq[slice_id]
            )
