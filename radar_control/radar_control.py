#!/usr/bin/env python3

"""
    radar_control process
    ~~~~~~~~~~~~~~~~~~~~~

    Radar_control is the process that runs the radar (sends pulses to the driver with
    timing information and sends processing information to the signal processing process).
    Experiment_handler provides the experiment for radar_control to run. It iterates
    through the scan_class_base objects to control the radar.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

import cmath
import sys
import time
from datetime import datetime, timedelta
import os
import zmq
import pickle
import threading
import math
import numpy as np

from functools import reduce

sys.path.append(os.environ["BOREALISPATH"])
from experiment_prototype.experiment_exception import ExperimentException
from utils.experiment_options.experimentoptions import ExperimentOptions
import utils.shared_macros.shared_macros as sm

if __debug__:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')
else:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')

from driverpacket_pb2 import DriverPacket
from sigprocpacket_pb2 import SigProcPacket
from datawritemetadata_pb2 import IntegrationTimeMetadata

from experiment_prototype.experiment_prototype import ExperimentPrototype

from utils.zmq_borealis_helpers import socket_operations

TIME_PROFILE = False

rad_ctrl_print = sm.MODULE_PRINT("radar control", "green")


def setup_driver(driverpacket, radctrl_to_driver, driver_to_radctrl_iden, txctrfreq, rxctrfreq,
                 txrate, rxrate):
    """ First packet sent to driver for setup.
        :param driverpacket: the protobuf packet to fill and pass over zmq
        :param radctrl_to_driver: the sender socket for sending the driverpacket
        :param driver_to_radctrl_iden: the receiver socket identity on the driver side
        :param txctrfreq: the transmit center frequency to tune to, kHz.
        :param rxctrfreq: the receive center frequency to tune to. With rx_sample_rate from config.ini file, this
            determines the received signal band, kHz.
        :param txrate: the tx sampling rate (Hz).
        :param rxrate: the rx sampling rate (Hz).
    """

    driverpacket.Clear()
    driverpacket.txcenterfreq = txctrfreq * 1000  # convert to Hz
    driverpacket.rxcenterfreq = rxctrfreq * 1000  # convert to Hz
    driverpacket.txrate = txrate
    driverpacket.rxrate = rxrate

    socket_operations.send_pulse(radctrl_to_driver, driver_to_radctrl_iden, driverpacket.SerializeToString())

    socket_operations.recv_data(radctrl_to_driver, driver_to_radctrl_iden, rad_ctrl_print)


def data_to_driver(driverpacket, radctrl_to_driver, driver_to_radctrl_iden, samples_array,
                   txctrfreq, rxctrfreq, txrate, rxrate, numberofreceivesamples, seqtime, SOB, EOB, timing,
                   seqnum, repeat=False):
    """ Place data in the driver packet and send it via zeromq to the driver.
        :param driverpacket: the protobuf packet to fill and pass over zmq
        :param radctrl_to_driver: the sender socket for sending the driverpacket
        :param driver_to_radctrl_iden: the reciever socket identity on the driver side
        :param samples_array: this is a list of length main_antenna_count from the config file. It contains one
            numpy array of complex values per antenna. If the antenna will not be transmitted on, it contains a
            numpy array of zeros of the same length as the rest. All arrays will have the same length according to
            the pulse length.
        :param txctrfreq: the transmit center frequency to tune to.
        :param rxctrfreq: the receive center frequency to tune to. With rx_sample_rate from config.ini file, this
            determines the received signal band.
        :param txrate: the tx sampling rate (Hz).
        :param rxrate: the rx sampling rate (Hz).
        :param numberofreceivesamples: number of samples to receive at the rx_sample_rate from config.ini file. This
            determines length of Scope Sync GPIO being high for this sequence.
        :param SOB: start of burst boolean, true for first pulse in sequence.
        :param EOB: end of burst boolean, true for last pulse in sequence.
        :param timing: in us, the time past timezero to send this pulse. Timezero is the start of the sequence.
        :param seqnum: the sequence number. This is a unique identifier for the sequence that is always increasing
            with increasing sequences while radar_control is running. It is only reset when program restarts.
        :param repeat: a boolean indicating whether the pulse is the exact same as the last pulse
        in the sequence, in which case we will save the time and not send the samples list and other
        params that will be the same.
    """

    driverpacket.Clear()
    driverpacket.timetosendsamples = timing
    driverpacket.SOB = SOB
    driverpacket.EOB = EOB
    driverpacket.sequence_num = seqnum
    driverpacket.numberofreceivesamples = numberofreceivesamples
    driverpacket.seqtime = seqtime

    if repeat:
        # antennas empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        if __debug__:
            msg = "REPEAT; TIMING: {0}; SOB: {1}; EOB: {2};".format(timing, SOB, EOB)
            rad_ctrl_print(msg)
    else:
        # SETUP data to send to driver for transmit.
        for ant_idx in range(samples_array.shape[0]):
            sample_add = driverpacket.channel_samples.add()
            # Add one Samples message for each channel possible in config.
            # Any unused channels will be sent zeros.
            # Protobuf expects types: int, long, or float, will reject numpy types and
            # throw a TypeError so we must convert the numpy arrays to lists
            sample_add.real.extend(samples_array[ant_idx,:].real.tolist())
            sample_add.imag.extend(samples_array[ant_idx,:].imag.tolist())
        driverpacket.txcenterfreq = txctrfreq * 1000  # convert to Hz
        driverpacket.rxcenterfreq = rxctrfreq * 1000  # convert to Hz
        driverpacket.txrate = txrate
        driverpacket.rxrate = rxrate
        driverpacket.numberofreceivesamples = numberofreceivesamples
        if __debug__:
            msg = "NOT A REPEAT; TIMING: {0}; SOB: {1}; EOB: {2};".format(timing, SOB, EOB)
            rad_ctrl_print(msg)

    socket_operations.send_pulse(radctrl_to_driver, driver_to_radctrl_iden, driverpacket.SerializeToString())

    del driverpacket.channel_samples[:]  # TODO find out - Is this necessary in conjunction with .Clear()?


def send_dsp_metadata(packet, radctrl_to_dsp, dsp_radctrl_iden, radctrl_to_brian,
                      brian_radctrl_iden, rxrate, output_sample_rate, seqnum, slice_ids,
                      slice_dict, beam_dict, sequence_time, first_rx_sample_start,
                      main_antenna_count, rxctrfreq, decimation_scheme=None):
    """ Place data in the receiver packet and send it via zeromq to the signal processing unit and brian.
        Happens every sequence.
        :param packet: the signal processing packet of the protobuf sigprocpacket type.
        :param radctrl_to_dsp: The sender socket for sending data to dsp
        :param dsp_radctrl_iden: The reciever socket identity on the dsp side
        :param rxrate: The receive sampling rate (Hz).
        :param output_sample_rate: The output sample rate desired for the output data (Hz).
        :param seqnum: the sequence number. This is a unique identifier for the sequence that is always increasing
             with increasing sequences while radar_control is running. It is only reset when program restarts.
        :param slice_ids: The identifiers of the slices that are combined in this sequence. These IDs tell us where to
             look in the beam dictionary and slice dictionary for frequency information and beam direction information
             about this sequence to give to the signal processing unit.
        :param slice_dict: The slice dictionary, which contains information about all slices and will be referenced for
             information about the slices in this sequence. Namely, we get the frequency we want to receive at, the
             number of ranges and the first range information.
        :param beam_dict: The dictionary containing beam directions for each slice.
        :param sequence_time: entire duration of sequence, including receive time after all
             transmissions.
        :param first_rx_sample_start: The sample where the first rx sample will start relative to the
             tx data.
        :param main_antenna_count: number of main array antennas, from the config file.
        :param rxctrfreq: the center frequency of receiving, to send the translation frequency from center to dsp.
        :param decimation_scheme: object of type DecimationScheme that has all decimation and
             filtering data.

    """

    # TODO: does the for loop below need to happen every time. Could be only updated
    # as necessary to make it more efficient.
    packet.Clear()
    packet.sequence_time = sequence_time
    packet.sequence_num = seqnum
    packet.offset_to_first_rx_sample = first_rx_sample_start
    packet.rxrate = rxrate
    packet.output_sample_rate = output_sample_rate

    if decimation_scheme is not None:
        for stage in decimation_scheme.stages:
            dm_stage_add = packet.decimation_stages.add()
            dm_stage_add.stage_num = stage.stage_num
            dm_stage_add.input_rate = stage.input_rate
            dm_stage_add.dm_rate = stage.dm_rate
            dm_stage_add.filter_taps.extend(stage.filter_taps)

    for num, slice_id in enumerate(slice_ids):
        chan_add = packet.rxchannel.add()
        chan_add.slice_id = slice_id
        chan_add.tau_spacing = slice_dict[slice_id]['tau_spacing']  # us
        # send the translational frequencies to dsp in order to bandpass filter correctly.
        if slice_dict[slice_id]['rxonly']:
            chan_add.rxfreq = (rxctrfreq * 1.0e3) - slice_dict[slice_id]['rxfreq'] * 1.0e3
        elif slice_dict[slice_id]['clrfrqflag']:
            pass  # TODO - get freq from clear frequency search.
        else:
            chan_add.rxfreq = (rxctrfreq * 1.0e3) - slice_dict[slice_id]['txfreq'] * 1.0e3
        chan_add.num_ranges = slice_dict[slice_id]['num_ranges']
        chan_add.first_range = slice_dict[slice_id]['first_range']
        chan_add.range_sep = slice_dict[slice_id]['range_sep']

        main_bms = beam_dict[slice_id]['main']
        intf_bms = beam_dict[slice_id]['intf']

        for i in range(main_bms.shape[0]):
            beam_add = chan_add.beam_directions.add()
            # Don't need to send channel numbers, will always send beamdir with length = total antennas.
            # Beam directions are formated e^i*phi so that a 0 will indicate not
            # to receive on that channel.

            temp_main = np.zeros_like(main_bms[i], main_bms[i].dtype)
            temp_intf = np.zeros_like(intf_bms[i], intf_bms[i].dtype)

            mains = slice_dict[slice_id]['rx_main_antennas']
            temp_main[mains] = main_bms[i][mains]

            intfs = slice_dict[slice_id]['rx_int_antennas']
            temp_intf[intfs] = intf_bms[i][intfs]

            for phase in temp_main:
                phase_add = beam_add.phase.add()
                phase_add.real_phase = phase.real
                phase_add.imag_phase = phase.imag

            for phase in temp_intf:
                phase_add = beam_add.phase.add()
                phase_add.real_phase = phase.real
                phase_add.imag_phase = phase.imag


        for lag in slice_dict[slice_id]['lag_table']:
            lag_add = chan_add.lags.add()
            lag_add.pulse_1 = lag[0]
            lag_add.pulse_2 = lag[1]
            lag_add.lag_num = int(lag[1] - lag[0])

    # Brian requests sequence metadata for timeouts
    if TIME_PROFILE:
        time_waiting = datetime.utcnow()

    request = socket_operations.recv_request(radctrl_to_brian, brian_radctrl_iden,
                                             rad_ctrl_print)

    if TIME_PROFILE:
        time_done = datetime.utcnow() - time_waiting
        rad_ctrl_print('Time waiting for Brian request to send metadata: {}'.format(time_done))

    if __debug__:
        request_output = "Brian requested -> {}".format(request)
        rad_ctrl_print(request_output)

    bytes_packet = packet.SerializeToString()

    socket_operations.send_obj(radctrl_to_brian, brian_radctrl_iden, bytes_packet)

    socket_operations.send_obj(radctrl_to_dsp, dsp_radctrl_iden, packet.SerializeToString())


def search_for_experiment(radar_control_to_exp_handler,
                          exphan_to_radctrl_iden,
                          status):
    """
    Check for new experiments from the experiment handler
    :param radar_control_to_exp_handler:
    :param radctrl_to_exphan_iden: The
    :param status: status string (EXP_NEEDED or NO_ERROR).
    :returns new_experiment_received: boolean (True for new experiment received)
    :returns experiment: experiment instance (or None if there is no new experiment)
    """

    try:
        socket_operations.send_request(radar_control_to_exp_handler, exphan_to_radctrl_iden, status)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]

    experiment = None
    new_experiment_received = False

    try:
        serialized_exp = socket_operations.recv_exp(radar_control_to_exp_handler,
                                                    exphan_to_radctrl_iden,
                                                    rad_ctrl_print)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]

    new_exp = pickle.loads(serialized_exp)  # protocol detected automatically

    if isinstance(new_exp, ExperimentPrototype):
        experiment = new_exp
        new_experiment_received = True
        if __debug__:
            rad_ctrl_print("NEW EXPERIMENT FOUND")
    elif new_exp is not None:
        if __debug__:
            rad_ctrl_print("RECEIVED AN EXPERIMENT NOT OF TYPE EXPERIMENT_PROTOTYPE. CANNOT RUN.")
    else:
        if __debug__:
            rad_ctrl_print("The experiment was not updated - continuing.")
        # TODO decide what to do here. I think we need this case if someone doesn't build their experiment
        # properly

    return new_experiment_received, experiment


def send_datawrite_metadata(packet, radctrl_to_datawrite, datawrite_radctrl_iden,
                            seqnum, num_sequences, scan_flag, inttime, sequences, beam_iter,
                            experiment_id, experiment_name, scheduling_mode, output_sample_rate, 
                            experiment_comment, filter_scaling_factors, rx_center_freq, 
                            debug_samples=None):
    """
    Send the metadata about this integration time to datawrite so that it can be recorded.
    :param packet: The IntegrationTimeMetadata protobuf packet.
    :param radctrl_to_datawrite: The socket to send the packet on.
    :param datawrite_radctrl_iden: Identity of datawrite on the socket.
    :param seqnum: The last sequence number (identifier) that is valid for this integration
    period. Used to verify and synchronize driver, dsp, datawrite.
    :param num_sequences: The number of sequences that were sent in this integration period. (number of
    sequences to average together).
    :param scan_flag: True if this integration period is the first in a scan.
    :param inttime: The time that expired during this integration period.
    :param sequences: The sequences of class Sequence for this integration period (AveragingPeriod).
    :param beam_iter: The beam iterator of this averaging period.
    :param experiment_id: the ID of the experiment that is running
    :param experiment_name: the experiment name to be placed in the data files.
    :param scheduling_mode: the type of scheduling mode running at this time, to write to file.
    :param output_sample_rate: The output sample rate of the output data, defined by the
    experiment, in Hz.
    :param experiment_comment: The comment string for the experiment, user-defined.
    :param filter_scaling_factors: The decimation scheme scaling factors used for the experiment,
    to get the scaling for the data for accurate power measurements between experiments.
    :param rx_center_freq: The receive center frequency (kHz)
    :param debug_samples: the debug samples for this integration period, to be written to the
    file if debug is set. This is a list of dictionaries for each Sequence in the
    AveragingPeriod. The dictionary is set up in the sample_building module function
    create_debug_sequence_samples. The keys are 'txrate', 'txctrfreq', 'pulse_timing',
    'pulse_sample_start', 'sequence_samples', 'decimated_sequence', 'dmrate_error', and 'dmrate'.
    The 'sequence_samples' and 'decimated_samples' values are themselves dictionaries, where the
    keys are the antenna numbers (there is a sample set for each transmit antenna).
    """

    packet.Clear()
    packet.experiment_id = experiment_id
    packet.experiment_name = experiment_name
    packet.experiment_comment = experiment_comment
    packet.rx_center_freq = rx_center_freq
    packet.num_sequences = num_sequences
    packet.last_seqn_num = seqnum
    packet.scan_flag = scan_flag
    packet.integration_time = inttime.total_seconds()
    packet.output_sample_rate = output_sample_rate
    packet.data_normalization_factor = reduce(lambda x, y: x * y, filter_scaling_factors)  # multiply all
    packet.scheduling_mode = scheduling_mode

    for sequence_index, sequence in enumerate(sequences):
        sequence_add = packet.sequences.add()
        sequence_add.blanks[:] = sequence.blanks
        if debug_samples:
            sequence_add.tx_data.txrate = debug_samples[sequence_index]['txrate']
            sequence_add.tx_data.txctrfreq = debug_samples[sequence_index]['txctrfreq']
            sequence_add.tx_data.pulse_timing_us[:] = debug_samples[sequence_index][
                'pulse_timing']
            sequence_add.tx_data.pulse_sample_start[:] = debug_samples[sequence_index][
                'pulse_sample_start']
            for ant, ant_samples in debug_samples[sequence_index]['sequence_samples'].items():
                tx_samples_add = sequence_add.tx_data.tx_samples.add()
                tx_samples_add.real[:] = ant_samples['real']
                tx_samples_add.imag[:] = ant_samples['imag']
                tx_samples_add.tx_antenna_number = ant
            sequence_add.tx_data.dmrate = debug_samples[sequence_index]['dmrate']
            sequence_add.tx_data.dmrate_error = debug_samples[sequence_index]['dmrate_error']
            for ant, ant_samples in debug_samples[sequence_index]['decimated_samples'].items():
                tx_samples_add = sequence_add.tx_data.decimated_tx_samples.add()
                tx_samples_add.real[:] = ant_samples['real']
                tx_samples_add.imag[:] = ant_samples['imag']
                tx_samples_add.tx_antenna_number = ant
        for slice_id in sequence.slice_ids:
            rxchan_add = sequence_add.rxchannel.add()
            rxchan_add.slice_id = slice_id
            rxchan_add.slice_comment = sequence.slice_dict[slice_id]['comment']
            rxchan_add.interfacing = '{}'.format(sequence.slice_dict[slice_id]['slice_interfacing'])
            rxchan_add.rx_only = sequence.slice_dict[slice_id]['rxonly']
            rxchan_add.pulse_len = sequence.slice_dict[slice_id]['pulse_len']
            rxchan_add.tau_spacing = sequence.slice_dict[slice_id]['tau_spacing']

            if sequence.slice_dict[slice_id]['rxonly']:
                rxchan_add.rxfreq = sequence.slice_dict[slice_id]['rxfreq']
            else:
                rxchan_add.rxfreq = sequence.slice_dict[slice_id]['txfreq']

            rxchan_add.ptab.pulse_position[:] = sequence.slice_dict[slice_id]['pulse_sequence']

            for encoding in sequence.output_encodings[slice_id]:
                rx_encode = rxchan_add.sequence_encodings.add()
                python_type = encoding.flatten().tolist()
                rx_encode.encoding_value[:] = python_type
            sequence.output_encodings[slice_id] = []


            rxchan_add.rx_main_antennas[:] = sequence.slice_dict[slice_id]['rx_main_antennas']
            rxchan_add.rx_intf_antennas[:] = sequence.slice_dict[slice_id]['rx_int_antennas']

            beams = sequence.slice_dict[slice_id]["beam_order"][beam_iter]
            if isinstance(beams, int):
                beams = [beams]

            for beam in beams:
                beam_add = rxchan_add.beams.add()
                beam_add.beamazimuth = sequence.slice_dict[slice_id]["beam_angle"][beam]
                beam_add.beamnum = beam

            rxchan_add.first_range = sequence.slice_dict[slice_id]['first_range']
            rxchan_add.num_ranges = sequence.slice_dict[slice_id]['num_ranges']
            rxchan_add.range_sep = sequence.slice_dict[slice_id]['range_sep']
            if sequence.slice_dict[slice_id]['acf']:
                rxchan_add.acf = sequence.slice_dict[slice_id]['acf']
                rxchan_add.xcf = sequence.slice_dict[slice_id]['xcf']
                rxchan_add.acfint = sequence.slice_dict[slice_id]['acfint']
                for lag in sequence.slice_dict[slice_id]['lag_table']:
                    lag_add = rxchan_add.ltab.lag.add()
                    lag_add.pulse_position[:] = lag
                    lag_add.lag_num = int(lag[1] - lag[0])
                rxchan_add.averaging_method = sequence.slice_dict[slice_id]['averaging_method']
            rxchan_add.slice_interfacing = '{}'.format(sequence.slice_dict[slice_id]['slice_interfacing'])

    if __debug__:
        rad_ctrl_print('Sending metadata to datawrite.')

    socket_operations.send_bytes(radctrl_to_datawrite, datawrite_radctrl_iden,
                                 packet.SerializeToString())


def round_up_time(dt=None, round_to=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    Modified: K.Kotyk 2019

    Will round to the nearest minute mark. Adds one minute if rounded down.
    """
    if dt is None:
        dt = datetime.utcnow()
    midnight = dt.replace(hour=0, minute=0, second=0)
    seconds = (dt.replace(tzinfo=None) - midnight).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    result = dt + timedelta(0, rounding-seconds, -dt.microsecond)

    if result < dt:
        result += timedelta(minutes=1)
    return result


def radar():
    """
    Run the radar with the experiment supplied by experiment_handler.

    Receives an instance of an experiment. Iterates through the Scans,
    AveragingPeriods, Sequences, and pulses of the experiment.

    For every pulse, samples and other control information are sent to the n200_driver.

    For every pulse sequence, processing information is sent to the signal processing
    block.

    After every integration time (AveragingPeriod), the experiment block is given the
    opportunity to change the experiment (not currently implemented). If a new
    experiment is sent, radar will halt the old one and begin with the new experiment.
    """

    # Initialize driverpacket.
    driverpacket = DriverPacket()
    sigprocpacket = SigProcPacket()
    integration_time_packet = IntegrationTimeMetadata()

    # Get config options.
    options = ExperimentOptions()

    # The socket identities for radar_control, retrieved from options
    ids = [options.radctrl_to_exphan_identity, options.radctrl_to_dsp_identity,
           options.radctrl_to_driver_identity, options.radctrl_to_brian_identity,
           options.radctrl_to_dw_identity]

    # Setup sockets.
    # Socket to send pulse samples over.
    # TODO test: need to make sure that we know that all sockets are set up after this try...except block.
    # TODO test: starting the programs in different orders.
    try:
        sockets_list = socket_operations.create_sockets(ids, options.router_address)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR Setting up sockets"
        raise [ExperimentException(errmsg), e]
    radar_control_to_exp_handler = sockets_list[0]
    radar_control_to_dsp = sockets_list[1]
    radar_control_to_driver = sockets_list[2]
    radar_control_to_brian = sockets_list[3]
    radar_control_to_dw = sockets_list[4]

    # seqnum is used as a identifier in all packets while
    # radar is running so set it up here.
    # seqnum will get increased by num_sequences (number of averages or sequences in the integration period)
    # at the end of every integration time.
    seqnum_start = 0

    #  Wait for experiment handler at the start until we have an experiment to run.
    new_experiment_waiting = False

    while not new_experiment_waiting:
        new_experiment_waiting, experiment = search_for_experiment(
            radar_control_to_exp_handler, options.exphan_to_radctrl_identity,
            'EXPNEEDED')

    new_experiment_waiting = False
    new_experiment_loaded = True

    # Send driver initial setup data - rates and center frequency from experiment.
    # Wait for acknowledgment that USRP object is set up.
    setup_driver(driverpacket, radar_control_to_driver, options.driver_to_radctrl_identity,
                 experiment.txctrfreq, experiment.rxctrfreq, experiment.txrate,
                 experiment.rxrate)

    first_integration = True
    next_scan_start = None
    decimation_scheme = experiment.decimation_scheme
    while True:
        # This loops through all scans in an experiment, or restarts this loop if a new experiment occurs.
        # TODO : further documentation throughout in comments (high level) and in separate documentation.
        # Iterate through Scans, AveragingPeriods, Sequences, Pulses.

        if new_experiment_waiting:  # start anew on first scan if we have a new experiment.
            try:
                experiment = new_experiment
            except NameError:
                # new_experiment does not exist, should never happen as flag only gets set when
                # there is a new experiment.
                errmsg = 'Experiment could not be found'
                raise ExperimentException(errmsg)
            new_experiment_waiting = False
            new_experiment = None
            new_experiment_loaded = True

        for scan_num, scan in enumerate(experiment.scan_objects):
            if __debug__:
                rad_ctrl_print("Scan number: {}".format(scan_num))
            # scan iter is the iterator through the scanbound or through the number of averaging periods in the scan.
            scan_iter = 0
            # if a new experiment was received during the last scan, it finished the integration period it was on and
            # returned here with new_experiment_waiting set to True. Break to load new experiment.
            if new_experiment_waiting:  # start anew on first scan if we have a new experiment.
                break

            if scan.scanbound:
                if scan.align_scan_to_beamorder:
                    for aveperiod in scan.aveperiods:
                        aveperiod.beam_iter = 0  # always align first beam at start of scan

                # find the start of the next scan with a scanbound so we can
                # determine time remaining for end of scan
                next_scanbound = None
                next_scan_num = scan_num
                while next_scanbound is None:
                    next_scan_num += 1
                    if next_scan_num == len(experiment.scan_objects):
                        next_scan_num = 0
                    next_scanbound = experiment.scan_objects[next_scan_num].scanbound

                if first_integration:
                    # on the very first integration of Borealis starting, calculate the start minute
                    # align scanbound reference time to find when to start
                    now = datetime.utcnow()
                    dt = now.replace(second=0, microsecond=0)

                    if dt + timedelta(seconds=scan.scanbound[scan_iter]) >= now:
                        start_minute = dt
                    else:
                        start_minute = round_up_time(now)
                else:  # At the start of a scan object that has scanbound, recalculate the start
                    # minute to the previously calculated next_scan_start
                    start_minute = next_scan_start.replace(second=0, microsecond=0)

                # find the modulus of the number of aveperiod times to run in the scan and the number of AvePeriod classes.
                # the classes will be alternated so we can determine which class will be running at the end of the scan.
                index_of_last_aveperiod_in_scan = (scan.num_aveperiods_in_scan + scan.aveperiod_iter) % len(scan.aveperiods)
                last_aveperiod_intt = scan.aveperiods[index_of_last_aveperiod_in_scan].intt
                # a scanbound necessitates intt
                end_of_scan = start_minute + timedelta(seconds=scan.scanbound[-1]) + timedelta(seconds=last_aveperiod_intt * 1e-3)
                end_minute = end_of_scan.replace(second=0, microsecond=0)

                if end_minute + timedelta(seconds=next_scanbound[0]) >= end_of_scan:
                    next_scan_start = end_minute + timedelta(seconds=next_scanbound[0])
                else:
                    next_scan_start = round_up_time(end_of_scan) + timedelta(seconds=next_scanbound[0])

            while scan_iter < scan.num_aveperiods_in_scan and not new_experiment_waiting:
                # If there are multiple aveperiods in a scan they are alternated (INTTIME interfaced)
                aveperiod = scan.aveperiods[scan.aveperiod_iter]
                if TIME_PROFILE:
                    time_start_of_aveperiod = datetime.utcnow()

                # get new experiment here, before starting a new integration.
                # If new_experiment_waiting is set here, implement new_experiment after this
                # integration period. There may be a new experiment waiting, or a new experiment.
                if not new_experiment_waiting and not new_experiment_loaded:
                    new_experiment_waiting, new_experiment = search_for_experiment(
                        radar_control_to_exp_handler,
                        options.exphan_to_radctrl_identity, 'NOERROR')
                elif new_experiment_loaded:
                    new_experiment_loaded = False

                if __debug__:
                    rad_ctrl_print("New AveragingPeriod")
                
                # all phases are set up for this averaging period for the beams required. 
                # Time to start averaging in the below loop.            
                if not scan.scanbound:
                    integration_period_start_time = datetime.utcnow()  # ms
                    rad_ctrl_print("Integration start time: {}".format(integration_period_start_time))
                if aveperiod.intt is not None:
                    intt_break = True

                    if scan.scanbound:
                        # calculate scan start time. First beam in the sequence will likely
                        # be ready to go if the first scan aligns directly to the minute. The
                        # rest will need to wait until their boundary time is up.
                        beam_scanbound = start_minute + timedelta(seconds=scan.scanbound[scan_iter])
                        time_diff = beam_scanbound - datetime.utcnow()
                        if time_diff.total_seconds() > 0:
                            if __debug__ or first_integration:
                                msg = "{}s until averaging period {} at time {}"
                                msg = msg.format(sm.COLOR("blue", time_diff.total_seconds()),
                                                 sm.COLOR("yellow", scan_iter),
                                                 sm.COLOR("red", beam_scanbound))
                                rad_ctrl_print(msg)
                            # TODO: reduce sleep if we want to use GPS timestamped transmissions
                            time.sleep(time_diff.total_seconds())
                        else:
                            if __debug__:
                                # TODO: This will be wrong if the start time is in the past. 
                                # maybe use datetime.utcnow() like below
                                # TODO: instead of  beam_scanbound, or change wording to 
                                # when the aveperiod should have started?
                                msg = "starting averaging period {} at time {}"
                                msg = msg.format(sm.COLOR("yellow", scan_iter),
                                                 sm.COLOR("red", beam_scanbound))
                                rad_ctrl_print(msg)

                        integration_period_start_time = datetime.utcnow()  # ms
                        msg = "Integration start time: {}"
                        msg = msg.format(sm.COLOR("red", integration_period_start_time))
                        rad_ctrl_print(msg)

                        # Here we find how much system time has elapsed to find the true amount
                        # of time we can integrate for this scan boundary. We can then see if
                        # we have enough time left to run the integration period.
                        time_elapsed = integration_period_start_time - start_minute
                        if scan_iter < len(scan.scanbound) - 1:
                            scanbound_time = scan.scanbound[scan_iter + 1]
                            # TODO: scanbound_time could be in the past if system has taken 
                            # too long, perhaps calculate which 'beam' (scan_iter) instead by 
                            # rewriting this code for an experiment-wide scanbound attribute instead
                            # of individual scanbounds inside the scan objects
                            # TODO: if scan_iter skips ahead, aveperiod.beam_iter may also need to 
                            # if scan.align_to_beamorder is True
                            bound_time_remaining = scanbound_time - time_elapsed.total_seconds()
                        else:
                            bound_time_remaining = next_scan_start - integration_period_start_time
                            bound_time_remaining = bound_time_remaining.total_seconds()

                        msg = "scan {} averaging period {}: bound_time_remaining {}s"
                        msg = msg.format(sm.COLOR("yellow", scan_num),
                                         sm.COLOR("yellow", scan_iter),
                                         sm.COLOR("blue", round(bound_time_remaining, 6)))
                        rad_ctrl_print(msg)

                        if bound_time_remaining < aveperiod.intt * 1e-3:
                            # reduce the integration period to only the time remaining
                            # until the next scan boundary.
                            # TODO: Check for bound_time_remaining > 0
                            # to be sure there is actually time to run this intt
                            # (if bound_time_remaining < 0, we need a solution to 
                            # reset)
                            integration_period_done_time = integration_period_start_time + \
                                            timedelta(milliseconds=bound_time_remaining * 1e3)
                        else:
                            integration_period_done_time = integration_period_start_time + \
                                            timedelta(milliseconds=aveperiod.intt)
                    else:  # no scanbound for this scan
                        integration_period_done_time = integration_period_start_time + \
                                            timedelta(milliseconds=aveperiod.intt)
                else:  # intt does not exist, therefore using intn
                    intt_break = False
                    ending_number_of_sequences = aveperiod.intn  # this will exist

                msg = "AvePeriod slices and beam numbers: {}".format(
                    {x: y[aveperiod.beam_iter] for x, y in aveperiod.slice_to_beamorder.items()})
                rad_ctrl_print(msg)
                
                if TIME_PROFILE:
                    time_to_prep_aveperiod = datetime.utcnow() - time_start_of_aveperiod
                    rad_ctrl_print('Time to prep aveperiod: {}'.format(time_to_prep_aveperiod))

                #  Time to start averaging in the below loop
                
                num_sequences = 0
                time_remains = True
                pulse_transmit_data_tracker = {}
                debug_samples = []
                
                while time_remains:
                    for sequence_index, sequence in enumerate(aveperiod.sequences):

                        # Alternating sequences if there are multiple in the averaging_period.
                        start_time = datetime.utcnow()
                        if intt_break:
                            if start_time >= integration_period_done_time:
                                time_remains = False
                                integration_period_time = (start_time - integration_period_start_time)
                                break
                        else:  # break at a certain number of integrations
                            if num_sequences == ending_number_of_sequences:
                                time_remains = False
                                integration_period_time = start_time - integration_period_start_time
                                break

                        # on first sequence, we make the first set of samples.
                        if sequence_index not in pulse_transmit_data_tracker:
                            pulse_transmit_data_tracker[sequence_index] = {}
                            sqn, dbg = sequence.make_sequence(aveperiod.beam_iter, num_sequences)
                            if dbg:
                                debug_samples.append(dbg)
                            pulse_transmit_data_tracker[sequence_index][num_sequences] = sqn

                        def send_pulses():
                            for pulse_transmit_data in pulse_transmit_data_tracker[sequence_index][num_sequences]:
                                data_to_driver(driverpacket, radar_control_to_driver,
                                               options.driver_to_radctrl_identity,
                                               pulse_transmit_data['samples_array'],
                                               experiment.txctrfreq,
                                               experiment.rxctrfreq, experiment.txrate,
                                               experiment.rxrate,
                                               sequence.numberofreceivesamples,
                                               sequence.seqtime,
                                               pulse_transmit_data['startofburst'],
                                               pulse_transmit_data['endofburst'],
                                               pulse_transmit_data['timing'],
                                               seqnum_start + num_sequences,
                                               repeat=pulse_transmit_data['isarepeat'])

                            if TIME_PROFILE:
                                time_after_pulses = datetime.utcnow()
                                pulses_to_driver_time = time_after_pulses - start_time
                                output = 'Time for pulses to driver: {}'.format(pulses_to_driver_time)
                                rad_ctrl_print(output)

                        def send_dsp_meta():
                            rx_beam_phases = sequence.get_rx_phases(aveperiod.beam_iter)
                            send_dsp_metadata(sigprocpacket,
                                              radar_control_to_dsp,
                                              options.dsp_to_radctrl_identity,
                                              radar_control_to_brian,
                                              options.brian_to_radctrl_identity,
                                              experiment.rxrate,
                                              experiment.output_rx_rate,
                                              seqnum_start + num_sequences,
                                              sequence.slice_ids, experiment.slice_dict,
                                              rx_beam_phases, sequence.seqtime,
                                              sequence.first_rx_sample_start,
                                              options.main_antenna_count, experiment.rxctrfreq,
                                              decimation_scheme)

                            if TIME_PROFILE:
                                time_after_sequence_metadata = datetime.utcnow()
                                sequence_metadata_time = time_after_sequence_metadata - start_time
                                output = 'Time to send meta to DSP: {}'.format(sequence_metadata_time)
                                rad_ctrl_print(output)

                        def make_next_samples():
                            sqn, dbg = sequence.make_sequence(aveperiod.beam_iter, num_sequences + 1)
                            if dbg:
                                debug_samples.append(dbg)
                            pulse_transmit_data_tracker[sequence_index][num_sequences+1] = sqn

                            if TIME_PROFILE:
                                time_after_making_new_sqn = datetime.utcnow()
                                new_sequence_time = time_after_making_new_sqn - start_time
                                output = 'Time to make new sequence: {}'.format(new_sequence_time)
                                rad_ctrl_print(output)

                        # These three things can happen simultaneously. We can spawn them as
                        # threads.
                        threads = [threading.Thread(target=send_pulses),
                                   threading.Thread(target=send_dsp_meta),
                                   threading.Thread(target=make_next_samples)]

                        for thread in threads:
                            thread.daemon = True
                            thread.start()

                        for thread in threads:
                            thread.join()

                        num_sequences += 1

                        if first_integration:
                            decimation_scheme = None
                            first_integration = False

                        # Sequence is done
                        if __debug__:
                            time.sleep(1)

                if TIME_PROFILE:
                    time_at_end_aveperiod = datetime.utcnow()

                msg = "Number of sequences: {}"
                msg = msg.format(sm.COLOR("magenta", num_sequences))
                rad_ctrl_print(msg)

                if scan.aveperiod_iter == 0 and aveperiod.beam_iter == 0:
                    # This is the first integration time in the scan object.
                    # if scanbound is aligned to beamorder, the scan_iter will also = 0 at this point.
                    scan_flag = True
                else:
                    scan_flag = False

                last_sequence_num = seqnum_start + num_sequences - 1
                def send_dw():
                    send_datawrite_metadata(integration_time_packet, radar_control_to_dw,
                                        options.dw_to_radctrl_identity, last_sequence_num,
                                        num_sequences, scan_flag, integration_period_time,
                                        aveperiod.sequences, aveperiod.beam_iter,
                                        experiment.cpid, experiment.experiment_name,
                                        experiment.scheduling_mode,
                                        experiment.output_rx_rate, experiment.comment_string,
                                        experiment.decimation_scheme.filter_scaling_factors,
                                        experiment.rxctrfreq,
                                        debug_samples=debug_samples)

                thread = threading.Thread(target=send_dw)
                thread.daemon = True
                thread.start()
                # end of the averaging period loop - move onto the next averaging period.
                # Increment the sequence number by the number of sequences that were in this
                # averaging period.
                seqnum_start += num_sequences

                if TIME_PROFILE:
                    time_to_finish_aveperiod = datetime.utcnow() - time_at_end_aveperiod
                    rad_ctrl_print('Time to finish aveperiod: {}'.format(time_to_finish_aveperiod))

                aveperiod.beam_iter += 1
                if aveperiod.beam_iter == aveperiod.num_beams_in_scan:
                    aveperiod.beam_iter = 0
                scan_iter += 1
                scan.aveperiod_iter += 1
                if scan.aveperiod_iter == len(scan.aveperiods):
                    scan.aveperiod_iter = 0

if __name__ == "__main__":
    radar()
