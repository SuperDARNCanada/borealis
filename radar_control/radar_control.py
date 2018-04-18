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
import os
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

from sample_building.sample_building import azimuth_to_antenna_offset
from experiments.experiment_prototype import ExperimentPrototype
from radar_status.radar_status import RadarStatus
from utils.zmq_borealis_helpers import socket_operations

def data_to_driver(driverpacket, radctrl_to_driver, driver_to_radctrl_iden, antennas, samples_array,
                   txctrfreq, rxctrfreq, txrate,
                   numberofreceivesamples, SOB, EOB, timing, seqnum,
                   repeat=False):
    """ Place data in the driver packet and send it via zeromq to the driver.
        :param driverpacket: the protobuf packet to fill and pass over zmq
        :param radctrl_to_driver: the sender socket for sending the driverpacket
	:param driver_to_radctrl_iden: the reciever socket identity on the driver side
        :param antennas: the antennas to transmit on.
        :param samples_array: this is a list of length main_antenna_count from the config file. It contains one
            numpy array of complex values per antenna. If the antenna will not be transmitted on, it contains a
            numpy array of zeros of the same length as the rest. All arrays will have the same length according to
            the pulse length.
        :param txctrfreq: the transmit centre frequency to tune to.
        :param rxctrfreq: the receive centre frequency to tune to. With rx_sample_rate from config.ini file, this
            determines the received signal band.
        :param txrate: the tx sampling rate.
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

    if repeat:
        # antennas empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        if __debug__:
            print("REPEAT; TIMING: {0}; SOB: {1}; EOB: {2}; ANTENNAS: {3};".format(timing, SOB, EOB, antennas))
    else:
        # SETUP data to send to driver for transmit.
        for ant in antennas:
            driverpacket.channels.append(ant)
        for samples in samples_array:
            sample_add = driverpacket.channel_samples.add()
            # Add one Samples message for each channel.
            # Protobuf expects types: int, long, or float, will reject numpy types and throw a
            # TypeError so we must convert the numpy arrays to lists
            sample_add.real.extend(samples.real.tolist())
            sample_add.imag.extend(samples.imag.tolist())
        driverpacket.txcenterfreq = txctrfreq * 1000  # convert to Hz
        driverpacket.rxcenterfreq = rxctrfreq * 1000  # convert to Hz
        driverpacket.txrate = txrate
        driverpacket.numberofreceivesamples = numberofreceivesamples
        if __debug__:
            print("NOT A REPEAT; TIMING: {0}; SOB: {1}; EOB: {2}; ANTENNAS: {3};".format(timing, SOB, EOB, antennas))

   # txsocket.send(driverpacket.SerializeToString())
    socket_operations.send_pulse(radctrl_to_driver, driver_to_radctrl_iden, driverpacket.SerializeToString())

    del driverpacket.channel_samples[:]  # TODO find out - Is this necessary in conjunction with .Clear()?


def data_to_rx_dsp(packet, radctrl_to_dsp, dsp_radctrl_iden, seqnum, slice_ids, slice_dict, beam_dict):
    """ Place data in the receiver packet and send it via zeromq to the signal processing unit.
        :param packet: the signal processing packet of the protobuf sigprocpacket type.
        :param radctrl_to_dsp: The sender socket for sending data to dsp
	:param dsp_radctrl_iden: The reciever socket identity on the dsp side
        :param seqnum: the sequence number. This is a unique identifier for the sequence that is always increasing
            with increasing sequences while radar_control is running. It is only reset when program restarts.
        :param slice_ids: The identifiers of the slices that are combined in this sequence. These IDs tell us where to
            look in the beam dictionary and slice dictionary for frequency information and beam direction information
            about this sequence to give to the signal processing unit.
        :param slice_dict: The slice dictionary, which contains information about all slices and will be referenced for
            information about the slices in this sequence. Namely, we get the frequency we want to receive at, the
            number of ranges and the first range information.
        :param beam_dict: The dictionary containing beam directions for each slice.

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
        for beamdir in beam_dict[slice_id]:
            beam_add = chan_add.beam_directions.add()
            # beamdir is a list (len = total antennas, main and interferometer) with phase for each
            # antenna for that beam direction
            for phi in beamdir:
                phase = cmath.exp(phi * 1j)
                phase_add = beam_add.phase.add()
                phase_add.real_phase = phase.real
                phase_add.imag_phase = phase.imag

    # Don't need to send channel numbers, will always send with length = total antennas.
    # TODO : Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel. ** make this update phase = 0 on channels not included.

    socket_operations.send_reply(radctrl_to_dsp, dsp_radctrl_iden, packet.SerializeToString())
    # TODO : is it necessary to do a del packet.rxchannel[:] - test


def get_ack(socket_sender_identity, socket_receiver_identity, procpacket):
    """ Get the acknowledgement from the process. This works with both the driver and the signal processing sockets
    as both packets have the field for sequence number.
    :param sender_socket: The sender socket that the acknowledgement packet comes from. 
	Either the driver packet socket or the sigprocpacket socket.
    :param receiver_identity: The receiver identity of the socket that the acknowledgement packet comes from.
    :param procpacket: The packet type that we have. Either driverpacket or sigprocpacket.
    """
    
        #msg = socket.recv(flags=zmq.NOBLOCK)
#       TODO: LEFT OFF HERE, WHERE DO ACKS GET HANDLED?
	procpacket.ParseFromString(msg)
        return procpacket.sequence_num
    except zmq.Again:
        errmsg = "No Message Available"
        raise ExperimentException(errmsg)  # TODO what to do in this case
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]


def search_for_experiment(radctrl_to_exphan_iden, exphan_to_radctrl_iden, status):
    """
    Check for new experiments from the experiment handler
    :param radctrl_to_exphan_iden: The 
    :param status: status of type RadarStatus.
    :return: boolean (True for new experiment received), and the experiment (or None if there is no new experiment)
    """

    def printing(msg):
	RADAR_CONTROL = "\033[33m" + "RADAR_CONTROL: " + "\033[0m"
        sys.stdout.write(RADAR_CONTROL + msg + "\n")

    try:
        socket_operations.send_request(radctrl_to_exphan_iden, exphan_to_radctrl_iden, status)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]
    
    experiment = None
    new_experiment_received = False
    
    try:
	new_exp = socket_operations.recv_reply(radar_control_to_exp_handler, exp_handler_iden, printing)
    except zmq.ZMQBaseError as e:
	errmsg = "ZMQ ERROR"
	raise [ExperimentException(errmsg), e]

    if isinstance(new_exp, ExperimentPrototype):
    	experiment = new_exp
        new_experiment_received = True
        if __debug__:
            print("NEW EXPERIMENT FOUND")
    else:
        if __debug__:
            print("RECEIVED AN EXPERIMENT NOT OF TYPE EXPERIMENT_PROTOTYPE. CANNOT RUN.")
        # TODO decide what to do here. I think we need this case if someone doesn't build their experiment
        # properly

    return new_experiment_received, experiment


def verify_completed_sequence(tx_poller, tx_rx_poller, tx_socket, rx_ack_socket, rx_time_socket,
                              sigprocpacket, driverpacket, poll_timeout, seqnum, ):
    """ Check the sequence was successfully transmitted by the driver, and the previous sequence was successfully
        processed by the signal processing unit.
    :param tx_poller: The poller set up on the socket to the driver only, used on the first pulse sequence.
    :param tx_rx_poller: The poller set up on both the sockets to the driver and to the signal processing unit.
    :param tx_socket: The socket set up for communication both to and from the driver.
    :param rx_ack_socket: The socket set up for getting the acknowledgement from the signal processing unit.
    :param rx_time_socket: The socket set up for getting the kernel time for processing from the signal processing unit.
    :param sigprocpacket: The protobuf signal processing packet, used on the rx_ack_socket and the rx_time_socket.
    :param driverpacket: The protobuf driver packet, used on the tx_socket.
    :param poll_timeout: the timeout time for the pollers, currently passed in as scope sync time.
    :param seqnum: The unique identifier for the sequence just sent to the driver. The driver should return this same
        sequence number. The signal processing unit should return this number - 1 as it was working on data from the
        last pulse sequence concurrently.
    """
    if seqnum != 0:
        rx_seq_ack = False
        tx_seq_ack = False
        while not rx_seq_ack or not tx_seq_ack:
            poll_timeout += 100  # TODO remove when poll timeout updated taking into account time to pulse in the future
            try:
                socks = dict(tx_rx_poller.poll(poll_timeout))
                #socks, wlist, xlist = zmq.select([tx_socket, rx_ack_socket], [], [])
            except zmq.NotDone:
                pass  # TODO can use this
                # TODO test using this case for the first sequence, that is we have tx_socket in socks but not rx_ack_socket ***
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
        if __debug__:
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
            if __debug__:
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
    to change the control program. If a new program is sent, radar will halt the old one and begin 
    with the new experiment.
    """

    # Initialize driverpacket.
    driverpacket = driverpacket_pb2.DriverPacket()

    # Get config options.
    options = ExperimentOptions()

    # The socket identities for radar_control, retrieved from options
    ids = [options.radctl_to_exphan_identity, options.radctl_to_dsp_identity, 
           options.radctl_to_driver_identity, options.radctl_to_brian_identity]

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
    #tx_socket = setup_socket(context, options.radar_control_to_driver_address)
    #rx_socket = setup_socket(context, options.radar_control_to_rx_dsp_address)
    #rx_ack_socket = setup_socket(context, options.rx_dsp_to_radar_control_ack_address)
    #rx_time_socket = setup_socket(context, options.rx_dsp_to_radar_control_timing_address)
    #experiment_socket = setup_socket(context, options.experiment_handler_to_radar_control_address)

    #poller_for_first_completed_sequence = zmq.Poller()
    #poller_for_first_completed_sequence.register(tx_socket, zmq.POLLIN)

    #poller_for_completed_sequence = zmq.Poller()
    #poller_for_completed_sequence.register(tx_socket, zmq.POLLIN)
    #poller_for_completed_sequence.register(rx_ack_socket, zmq.POLLIN)

    sigprocpacket = sigprocpacket_pb2.SigProcPacket()
    # seqnum is used as a identifier in all packets while
    # radar is running so set it up here.
    # seqnum will get increased by nave (number of averages or sequences in the integration period)
    # at the end of every integration time.
    seqnum_start = 0

    status = RadarStatus()

    new_experiment_flag = False

    while not new_experiment_flag:  #  Wait for experiment handler at the start until we have an experiment to run.
        new_experiment_flag, experiment = search_for_experiment(options.radctrl_to_exphan_identity, status)

    new_experiment_flag = False

    while True:
        # This loops through all scans in an experiment, or restarts this loop if a new experiment occurs.
        # TODO : further documentation throughout in comments (high level) and in separate documentation.
        # Iterate through Scans, AveragingPeriods, Sequences, Pulses.
        for scan in experiment.scan_objects:
            # if a new experiment was received during the last scan, it finished the integration period it was on and
            # returned here with new_experiment_flag set to True. Now change experiments if necessary.
            if new_experiment_flag:  # start anew on first scan if we have a new experiment.
                try:
                    experiment = new_experiment
                except NameError:
                    # new_experiment does not exist, should never happen as flag only gets set when
                    # there is a new experiment.
                    errmsg = 'Experiment could not be found'
                    raise ExperimentException(errmsg)
                new_experiment_flag = False
                new_experiment = None
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

                    # get new experiment here, before starting a new integration.
                    # if new_experiment_flag is set here, we will implement the new_experiment after this integration
                    # period.
		    # TODO: This needs a timeout, or we'll just get stuck here... in brian maybe?
                    new_experiment_flag, new_experiment = search_for_experiment(options.radctl_to_exphan_identity,
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
                    if __debug__:
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

                    # all phases are set up for this averaging period for the beams required. Time to start averaging
                    # in the below loop.
                    nave = 0
                    time_remains = True
                    integration_period_done_time = integration_period_start_time + \
                        timedelta(milliseconds=(float(aveperiod.intt)))  # ms
                    while time_remains:
                        for sequence_index, sequence in enumerate(aveperiod.sequences):
                            # Alternating sequences if there are multiple in the averaging_period.
                            if datetime.utcnow() >= integration_period_done_time:
                                time_remains = False
                                break
                                # TODO add a break for nave == intn if going for number of averages instead of
                                # integration time
                            beam_phase_dict = beam_phase_dict_list[sequence_index]
                            data_to_rx_dsp(sigprocpacket, options.radctrl_to_dsp_identity, seqnum_start + nave,
                                           sequence.slice_ids, experiment.slice_dict,
                                           beam_phase_dict)
                            # beam_phase_dict is slice_id : list of beamdirs, where beamdir = list
                            # of antenna phase offsets for all antennas for that direction ordered
                            # [0 ... main_antenna_count, 0 ... interferometer_antenna_count]


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
                            poll_timeout = int(sequence.sstime / 1000)  # ms TODO change based on pulse time into the future once timing info exchange is set up
                            # Get sequence acknowledgements and log synchronization and
                            # communication errors between the n200_driver, rx_sig_proc, and
                            # radar_control.

                            verify_completed_sequence(poller_for_first_completed_sequence,
                                                      poller_for_completed_sequence, tx_socket, rx_ack_socket,
                                                      rx_time_socket, sigprocpacket, driverpacket,
                                                      poll_timeout, seqnum_start + nave)

                            # TODO: Make sure you can have a slice that doesn't transmit, only receives on a frequency. # REVIEW #1 what do you mean, what is this TODO for? REPLY : driver acks wouldn't be required etc need to make sure this is possible
                            if __debug__:
                                time.sleep(1)
                            # Sequence is done
                            nave = nave + 1
                    if __debug__:
                        print("Number of integrations: {}".format(nave))
                    seqnum_start += nave
                    # end of the averaging period loop - move onto the next averaging period. Increment the sequence
                    # number by the number of sequences that were in this averaging period.
                beam_iter = beam_iter + 1


if __name__ == "__main__":
    radar()
