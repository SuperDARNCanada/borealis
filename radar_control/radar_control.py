#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# radar_control.py
# 2017-03-13
# Get a radar control program made of objects (scans, averaging periods, and sequences).
# Communicate with the n200_driver to control the radar.
# Communicate with the rx_dsp_chain to process the data.

import cmath
import sys
import time
from datetime import datetime, timedelta
import os
import zmq
sys.path.append(os.environ["BOREALISPATH"])
from experiments.experiment_exception import ExperimentException
from utils.experiment_options.experimentoptions import ExperimentOptions

if __debug__:
	sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')  # TODO need to get this from scons environment, 'release' may be 'debug'
else:
	sys.path.append(os.environ["BOREALISPATH"]+ '/build/release/utils/protobuf')
import driverpacket_pb2
import sigprocpacket_pb2

from sample_building.sample_building import get_wavetables, azimuth_to_antenna_offset
from experiments.experiment_prototype import ExperimentPrototype

from radar_status.radar_status import RadarStatus


def setup_socket(context, address):
    """
    Setup a paired zmq socket and return it.
    :param context: zmq context
    :return: zmq socket
    """
    socket = context.socket(zmq.PAIR)
    socket.connect(address)
    return socket


def data_to_driver(driverpacket, txsocket, antennas, samples_array,
                   txctrfreq, rxctrfreq, txrate,
                   numberofreceivesamples, SOB, EOB, timing, seqnum,
                   repeat=False):
    """ Place data in the driver packet and send it via zeromq to the driver.
        :param driverpacket: 
        :param txsocket: 
        :param antennas: 
        :param samples_array: 
        :param txctrfreq: 
        :param rxctrfreq: 
        :param txrate: 
        :param numberofreceivesamples: 
        :param SOB: 
        :param EOB: 
        :param timing: 
        :param seqnum: 
        :param repeat: a boolean indicating whether the pulse is the exact same as the last pulse
        in the sequence, in which case we will save the time and not send the samples list and other
        params that will be the same.
    """
    driverpacket.Clear()
    driverpacket.timetosendsamples = timing  # us, past time zero which is start of the sequence.
    driverpacket.SOB = SOB
    driverpacket.EOB = EOB
    driverpacket.sequence_num = seqnum

    if repeat:
        # antennas empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        print("EMPTY {0} {1} {2} {3}".format(timing, SOB, EOB, antennas)) # TODO add debug to print statements
    else:  # TODO add check to make sure we don't have blanks when we shouldn't
        # SETUP data to send to driver for transmit.
        for ant in antennas:
            driverpacket.channels.append(ant)
        for sample_index in range(len(samples_array)):
            sample_add = driverpacket.samples.add()
            # Add one Samples message for each channel.
            # Protobuf expects types: int, long, or float, will reject numpy types and throw a
            # TypeError so we must convert the numpy arrays to lists
            sample_add.real.extend(samples_array[sample_index].real.tolist())
            sample_add.imag.extend(samples_array[sample_index].imag.tolist())
        driverpacket.txcenterfreq = txctrfreq * 1000  # convert to Hz
        driverpacket.rxcenterfreq = rxctrfreq * 1000  # convert to Hz
        driverpacket.txrate = txrate
        driverpacket.numberofreceivesamples = numberofreceivesamples
        print("New samples {0} {1} {2} {3}".format(timing, SOB, EOB, antennas))

    txsocket.send(driverpacket.SerializeToString())

    del driverpacket.samples[:]  # TODO find out - Is this needed in conjunction with .Clear()?


def data_to_rx_dsp(packet, socket, seqnum, slice_ids, slice_dict,
                   beam_dict):
    """ Place data in the receiver packet and send it via zeromq to the
        receiver unit.
    """

    packet.Clear()
    packet.sequence_num = seqnum
    for num, slice_id in enumerate(slice_ids):
        chan_add = packet.rxchannel.add()
        if slice_dict[slice_id]['rxonly']:
            chan_add.rxfreq = slice_dict[slice_id]['rxfreq']
        elif slice_dict[slice_id]['clrfrqflag']:
            pass  # TODO - get freq from clear frequency search.
        else:
            chan_add.rxfreq = slice_dict[slice_id]['txfreq']
        chan_add.nrang = slice_dict[slice_id]['nrang']
        chan_add.frang = slice_dict[slice_id]['frang']
        for bnum, beamdir in enumerate(beam_dict[slice_id]):
            beam_add = packet.rxchannel[num].beam_directions.add()
            # beamdir is a list (len = total antennas, main and interferometer) with phase for each
            # antenna for that beam direction
            for pnum, phi in enumerate(beamdir):
                # print(phi)
                phase = cmath.exp(phi * 1j)
                phase_add = beam_add.phase.add()
                phase_add.real_phase = phase.real
                phase_add.imag_phase = phase.imag

    # Don't need to send channel numbers, will always send with length = total antennas.
    # TODO : Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel. ** make this update phase = 0 on channels not included.

    socket.send(packet.SerializeToString())
    return


def get_ack(socket, procpacket):
    """
    
    """
    try:
        msg = socket.recv(flags=zmq.NOBLOCK)
        procpacket.ParseFromString(msg)  # REVIEW #37 error check this - it might not throw a zmq.Again exception. You can use multiple except blocks after 1 try block
        return procpacket.sequence_num
    except zmq.Again:
        errmsg = "No Message Available"
        raise ExperimentException(errmsg)  # TODO what to do in this case
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]


def search_for_experiment(socket, status):
    """
    Check for new experiments from the experiment handler
    :param socket: socket to experiment handler
    :param status: status of type RadarStatus.
    :return: boolean (True for new experiment received), and the experiment (or None if there is no new experiment)
    """

    try:
        socket.send_pyobj(status)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    while True:
        try:
            socks = dict(poller.poll(10))  # polls for 10 ms, before inttime timer starts.
        except zmq.NotDone:
            experiment = None
            new_experiment_received = False
            print("NO NEW EXPERIMENT. CONTINUING.")
            break  # TODO log
        except zmq.ZMQBaseError as e:
            errmsg = "ZMQ ERROR"
            raise [ExperimentException(errmsg), e]

        if socket in socks:  #
            if socks[socket] == zmq.POLLIN:
                try:
                    new_exp = socket.recv_pyobj()  # message should be waiting
                except zmq.ZMQBaseError as e:
                    errmsg = "ZMQ ERROR"
                    raise [ExperimentException(errmsg), e]
                if isinstance(new_exp, ExperimentPrototype):
                    experiment = new_exp
                    new_experiment_received = True
                    print("NEW EXPERIMENT FOUND")
                    break
                else:
                    experiment = None
                    new_experiment_received = False
                    print("NO NEW EXPERIMENT. CONTINUING.")
                    break
        else:
            print("No Experiment Provided - Continuing to Poll")

    return new_experiment_received, experiment


def verify_completed_sequence(tx_poller, tx_rx_poller, tx_socket, rx_ack_socket, rx_time_socket,
                              sigprocpacket, driverpacket, poll_timeout, seqnum, ):
    """
    
    :param tx_poller: 
    :param tx_rx_poller: 
    :param tx_socket: 
    :param rx_ack_socket: 
    :param rx_time_socket: 
    :param sigprocpacket: 
    :param driverpacket: 
    :param poll_timeout: currently equal to scope sync time.
    :param seqnum: 
    :return: 
    """
    if seqnum != 0:
        rx_seq_ack = False
        tx_seq_ack = False
        while not rx_seq_ack or not tx_seq_ack:
            try:
                socks = dict(tx_rx_poller.poll(poll_timeout))
                #socks, wlist, xlist = zmq.select([tx_socket, rx_ack_socket], [], [])
            except zmq.NotDone:
                pass  # TODO can use this
            except zmq.ZMQBaseError as e:
                errmsg = "ZMQ ERROR"
                raise [ExperimentException(errmsg), e]

            if rx_ack_socket in socks:  # need one message from both.
                if socks[rx_ack_socket] == zmq.POLLIN:
                    rxseqnum = get_ack(rx_ack_socket, sigprocpacket)
                    # rx processing block is working on the pulse sequence before the one we just
                    # transmitted, therefore should = seqnum - 1.
                    if rxseqnum != seqnum - 1:
                        errmsg = "WRONG RXSEQNUM received from rx_signal_processing {} ; "\
                              "Expected {}".format(rxseqnum, seqnum - 1)
                        raise ExperimentException(errmsg)
                    else:  # TODO add if debug
                        rx_seq_ack = True
                        print("RXSEQNUM {}".format(rxseqnum))
            elif tx_socket in socks:
                if socks[tx_socket] == zmq.POLLIN:
                    txseqnum = get_ack(tx_socket, driverpacket)
                    # driver should have received and sent the current seqnum.
                    if txseqnum != seqnum:
                        errmsg = "WRONG TXSEQNUM received from driver {} ; Expected {}".format(
                            txseqnum, seqnum)
                        raise ExperimentException(errmsg)
                    else:
                        tx_seq_ack = True
                        print("TXSEQNUM {}".format(txseqnum))
            else:
                errmsg = "Did not receive ack from either rx_ack_socket or tx_socket"
                raise ExperimentException(errmsg)
                # TODO what to do here - some lag or something is not running
    else:  # on the very first sequence since starting the radar.
        # extra poll time on first sequence as found required 
        first_sequence_poll_timeout = 2000
        print("Polling for first sequence for {} ms".format(first_sequence_poll_timeout))
        try:
            sock = tx_poller.poll(first_sequence_poll_timeout)
        except zmq.NotDone:
            # TODO start a timer and trying sending again on the first sequence. - in case you start experiment handler second ***
            pass
        except zmq.ZMQBaseError as e:
            errmsg = "ZMQ ERROR"
            raise [ExperimentException(errmsg), e]

        try:
            tx_socket.recv(flags=zmq.NOBLOCK)
            print("FIRST ACK RECEIVED")
        except zmq.Again:
            errmsg = "No first ack from driver - This shouldn't happen"
            raise ExperimentException(errmsg)
        except zmq.ZMQBaseError as e:
            errmsg = "ZMQ ERROR"
            raise [ExperimentException(errmsg), e]
    # Now, check how long the kernel is taking to process (non-blocking)
    try:
        kernel_time_ack = rx_time_socket.recv(flags=zmq.NOBLOCK)  # TODO units of the ack (ms?) TODO: use this ack value in error checking
    except zmq.Again:
        pass  # TODO: Should do something if don't receive kernel_time_ack for many sequences.
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]


def radar():
    """
    Receives an instance of an experiment. Iterates through
    the Scans, AveragingPeriods, Sequences, and pulses of the experiment.
    For every pulse, samples and other control information are sent to the n200_driver.
    For every pulse sequence, processing information is sent to the signal processing block.
    After every integration time (AveragingPeriod), the experiment block is given the opportunity
    to change the control program (if it sends a new one, runradar will halt the old one and begin 
    with the new experiment).
    """

    # Initialize driverpacket.
    driverpacket = driverpacket_pb2.DriverPacket()

    # Get config options.
    options = ExperimentOptions()

    context = zmq.Context()  # single context for all sockets involved in this process.
    # Setup sockets.
    # Socket to send pulse samples over.
    try:
        tx_socket = setup_socket(context, options.radar_control_to_driver_address)
        rx_socket = setup_socket(context, options.radar_control_to_rx_dsp_address)
        rx_ack_socket = setup_socket(context, options.rx_dsp_to_radar_control_ack_address)
        rx_time_socket = setup_socket(context, options.rx_dsp_to_radar_control_timing_address)
        experiment_socket = setup_socket(context, options.experiment_handler_to_radar_control_address)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR Setting up sockets"
        raise [ExperimentException(errmsg), e]

    poller = zmq.Poller()
    poller.register(tx_socket, zmq.POLLIN)

    poller2 = zmq.Poller()
    poller2.register(tx_socket, zmq.POLLIN)
    poller2.register(rx_ack_socket, zmq.POLLIN)

    sigprocpacket = sigprocpacket_pb2.SigProcPacket()
    # seqnum is used as a identifier in all packets while
    # radar is running so set it up here.
    # seqnum will get increased by nave (number of averages or sequences in the integration period)
    # at the end of every integration time.
    # REVIEW #1 The wording here makes it sound like seqnum is increased once at the end of every integration time, not once every pulse sequence # REPLY that's true here, increased by nave.
    seqnum_start = 0

    status = RadarStatus()

    new_experiment_flag = False

    while not new_experiment_flag:  # poll the socket at the start until we have an experiment to run.
        new_experiment_flag, experiment = search_for_experiment(experiment_socket, status)

    new_experiment_flag = False

    while True:
        # This loops through all scans in an experiment, or restarts this loop if a new experiment occurs.

        # Wavetables are currently None for sine waves, instead just
        #   use a sampling freq in rads/sample.
        # TODO move this to experiment handler before experiment is transferred ?
        for slice_id, expslice in experiment.slice_dict.items():
            # print("Slice ID {}, slice {}".format(slice_id, expslice))
            expslice['iwavetable'], expslice['qwavetable'] = \
                get_wavetables(expslice['wavetype'])

        # Iterate through Scans, AveragingPeriods, Sequences, Pulses.
        for scan in experiment.scan_objects:
            if new_experiment_flag:  # start anew on first scan if we have a new experiment. Question - should this be how we handle a new experiment that's received at the start of a new integration period - end scan after that inttime and start the new experiment?
                try:
                    experiment = new_experiment
                except NameError:
                    # new_experiment does not exist, should never happen as flag only gets set when
                    # there is a new experiment.
                    errmsg = 'Experiment could not be found'
                    raise ExperimentException(errmsg)
                new_experiment_flag = False
                break
            beam_remaining = True  # started a new scan, therefore this must be True.

            # Make iterator for cycling through beam numbers
            aveperiods_done_list = []
            beam_iter = 0
            while beam_remaining and not new_experiment_flag:
                for aveperiod in scan.aveperiods:
                    # If there are multiple aveperiods in a scan they are alternated
                    #   beam by beam. So we need to iterate through
                    # Go through each aveperiod once then increase the beam
                    #   iterator to the next beam in each scan.

                    # poll for new experiment here, before starting a new integration.
                    # if new_experiment_flag is set here, we will implement the new_experiment after this integration
                    # period.
                    new_experiment_flag, new_experiment = search_for_experiment(experiment_socket,
                                                                                status)

                    # Check if there are beams remaining in this aveperiod, or in any aveperiods.
                    if aveperiod in aveperiods_done_list:
                        continue  # beam_iter index is past the end of the beam_order list for this aveperiod, but other aveperiods must still contain a beam at index beam_iter in the beam_order list.
                    else:
                        if beam_iter == len(scan.scan_beams[aveperiod.slice_ids[0]]):  # REVIEW #3 We just don't understand this. REPLY: fixed up, tried to make more clear
                            # All slices in the aveperiod have the same length beam_order.
                            # Check if we are at the end of the beam_order list (scan) for this aveperiod instance.
                            # If we are, we still might not be done all of the beams in another aveperiod,
                            # so we should just record that we are done with this one for this scan and
                            # keep going to check the next aveperiod type.
                            aveperiods_done_list.append(aveperiod)
                            if len(aveperiods_done_list) == len(scan.aveperiods):
                                beam_remaining = False  # all aveperiods are at the end of their beam_order list - must restart scan of alternating aveperiod types.
                                break
                            continue
                    print("New AveragingPeriod")
                    integration_period_start_time = datetime.utcnow()  # ms

                    slice_to_beamdir_dict = aveperiod.set_beamdirdict(beam_iter)

                    # Build an ordered list of sequences
                    # A sequence is a list of pulses in order
                    # A pulse is a dictionary with all required information for that pulse.
                    sequence_dict_list = aveperiod.build_sequences(slice_to_beamdir_dict,
                                                                   experiment.txctrfreq,
                                                                   experiment.txrate, options)  # TODO pass in only options needed.

                    beam_phase_dict_list = []

                    for sequence_index, sequence in enumerate(aveperiod.sequences):
                        beam_phase_dict = {}
                        for slice_id in sequence.slice_ids:

                            beamdir = slice_to_beamdir_dict[slice_id]
                            beam_phase_dict[slice_id] = \
                                azimuth_to_antenna_offset(beamdir, options.main_antenna_count,
                                                          options.interferometer_antenna_count,
                                                          options.main_antenna_spacing,
                                                          options.interferometer_antenna_spacing,
                                                          experiment.slice_dict[slice_id]['txfreq'])

                        beam_phase_dict_list.append(beam_phase_dict)

                    nave = 0
                    time_remains = True
                    integration_period_done_time = integration_period_start_time + \
                        timedelta(milliseconds=(float(aveperiod.intt)))  # ms
                    while time_remains:
                        for sequence_index, sequence in enumerate(aveperiod.sequences):
                            if datetime.utcnow() >= integration_period_done_time:
                                time_remains = False
                                break
                            beam_phase_dict = beam_phase_dict_list[sequence_index]
                            data_to_rx_dsp(sigprocpacket, rx_socket, seqnum_start + nave,
                                           sequence.slice_ids, experiment.slice_dict,
                                           beam_phase_dict)
                            # beam_phase_dict is slice_id : list of beamdirs, where beamdir = list
                            # of antenna phase offsets for all antennas for that direction ordered
                            # [0 ... main_antenna_count, 0 ... interferometer_antenna_count]


                            # Just alternating sequences
                            # print(sequence_dict_list)

                            # SEND ALL PULSES IN SEQUENCE.
                            for pulse_index, pulse_dict in \
                                    enumerate(sequence_dict_list[sequence_index]):
                                data_to_driver(driverpacket, tx_socket,
                                               pulse_dict['pulse_antennas'],
                                               pulse_dict['samples_array'], experiment.txctrfreq,
                                               experiment.rxctrfreq, experiment.txrate,
                                               sequence.numberofreceivesamples,
                                               pulse_dict['startofburst'], pulse_dict['endofburst'],
                                               pulse_dict['timing'], seqnum_start + nave,
                                               repeat=pulse_dict['isarepeat'])
                                # Pulse is done.
                            poll_timeout = int(sequence.sstime / 1000 + 20)  # ms
                            # Get sequence acknowledgements and log synchronization and
                            # communication errors between the n200_driver, rx_sig_proc, and
                            # radar_control.

                            verify_completed_sequence(poller, poller2, tx_socket, rx_ack_socket,
                                                      rx_time_socket, sigprocpacket, driverpacket,
                                                      poll_timeout, seqnum_start + nave)

                            # TODO: Make sure you can have a CPO that doesn't transmit, only receives on a frequency. # REVIEW #1 what do you mean, what is this TODO for? REPLY : driver acks wouldn't be required etc need to make sure this is possible
                            #time.sleep(1)  # TODO add if debug
                            # Sequence is done
                            nave = nave + 1
                    print("Number of integrations: {}".format(nave))
                    seqnum_start += nave
                beam_iter = beam_iter + 1


if __name__ == "__main__":
    radar()
