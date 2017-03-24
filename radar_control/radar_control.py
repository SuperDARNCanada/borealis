#!/usr/bin/python

# radar_control.py
# 2017-03-13
# Marci Detwiller
# Get a radar control program made of objects (scans, averaging periods, and sequences).
# Communicate with the n200_driver to control the radar.
# Communicate with the rx_dsp_chain to process the data.

import sys
import os
import math
import cmath
from datetime import datetime, timedelta
import time

import zmq
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import kaiserord, lfilter, firwin, freqz
#from multiprocessing import Process, Value

sys.path.append('../build/release/utils/protobuf')
import driverpacket_pb2
import sigprocpacket_pb2

sys.path.append('./include')
from sample_building import get_phshift, get_wavetables, make_pulse_samples

sys.path.append('../experiments')
import normalscan # TODO - have it setup by scheduler

import radar_status

def setup_driver_socket(): # to send pulses to driver.
    """
    Setup a zmq socket to communicate with the driver. 
    """

    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/0")
    return cpsocket


def setup_sigproc_params_socket(): #to send data to receive code.
    """
    Setup a zmq socket to communicate with rx_signal_processing.
    """

    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/2")
    return cpsocket


def setup_sigproc_cpack_socket(): #to send data to receive code.
    """
    Setup a zmq socket to get acks from rx_signal_processing.
    """

    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/3")
    return cpsocket


def setup_sigproc_timing_ack_socket():
    """
    Setup a zmq socket to get timing information from rx_signal_processing.
    """

    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/4")
    return cpsocket


def setup_cp_socket(): 
    """
    Setup a zmq socket to get updated experiment parameters from the experiment.
    """

    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/5")
    return cpsocket


#def setup_ack_poller(socket1,socket2):
#    """
#     
#    """
#
#    poller=zmq.Poller()
#    poller.register(socket1, zmq.POLLIN)
#    poller.register(socket2, zmq.POLLIN)
#    return poller


#def pollzmq(socket1, socket2, rxseqnum, txseqnum):
#    """
#    
#    """
#
#    poller=zmq.Poller()
#    poller.register(socket1, zmq.POLLIN)
#    poller.register(socket2, zmq.POLLIN)
#    socks = dict(poller.poll(100)) # get two messages
#    if procsocket in socks and txsocket in socks: # need one message from both.
#        if socks[procsocket] == zmq.POLLIN:
#            rxseqnum = get_ack(procsocket)
#        if socks[txsocket] == zmq.POLLIN:
#            txseqnum = get_ack(txsocket)


def get_prog(socket):
    """
    Receive pickled python object of the experiment class. 
    """

    prog = socket.recv_pyobj()

    return prog

def data_to_driver(driverpacket, txsocket, channels, isamples_list,
                   qsamples_list, txctrfreq, rxctrfreq, txrate,
                   numberofreceivesamples, SOB, EOB, timing, seqnum, repeat=False):
    """ Place data in the driver packet and send it via zeromq to the driver.
        Then receive the acknowledgement.
    """

    if repeat:
        driverpacket.Clear()
        # channels empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        driverpacket.timetosendsamples=timing
        driverpacket.SOB=SOB
        driverpacket.EOB=EOB
        driverpacket.sequence_num=seqnum
        print "EMPTY {0} {1} {2} {3}".format(timing,SOB,EOB,channels)
        # timetoio empty
    else:
        # SETUP data to send to driver for transmit.
        driverpacket.Clear()
        for chan in channels:
            chan_add=driverpacket.channels.append(chan)
        for chi in range(len(isamples_list)):
            sample_add=driverpacket.samples.add()
            # Add one Samples message for each channel.
            real_samp=driverpacket.samples[chi].real.extend(isamples_list[chi])
            imag_samp=driverpacket.samples[chi].imag.extend(qsamples_list[chi])
        driverpacket.txcenterfreq=txctrfreq * 1000
        driverpacket.rxcenterfreq=rxctrfreq * 1000
        driverpacket.txrate=txrate
        driverpacket.numberofreceivesamples=numberofreceivesamples
        driverpacket.timetosendsamples=timing
        driverpacket.sequence_num=seqnum
        # Past time zero which is start of the sequence.
        print "New samples {0} {1} {2} {3}".format(timing,SOB,EOB,channels)
        driverpacket.SOB=SOB
        driverpacket.EOB=EOB

    txsocket.send(driverpacket.SerializeToString())
    # get response:
    #tx_ack = socket.recv()
    tx_ack=1

    del driverpacket.samples[:]

    return tx_ack


def data_to_processing(packet,procsocket, seqnum, cpos, cpo_list, beam_dict):
    """ Place data in the receiver packet and send it via zeromq to the
        receiver unit.
    """

    packet.Clear()
    packet.sequence_num=seqnum
    for num, cpo in enumerate(cpos):
        channel_add = packet.rxchannel.add()
        packet.rxchannel[num].rxfreq = cpo_list[cpo]['rxfreq']
        packet.rxchannel[num].nrang = cpo_list[cpo]['nrang']
        packet.rxchannel[num].frang = cpo_list[cpo]['frang']
        for bnum, beamdir in enumerate(beam_dict[cpo]):
            beam_add = packet.rxchannel[num].beam_directions.add()
            # beamdir is a list 20-long with phase for each antenna for that beam direction.
            for pnum, phi in enumerate(beamdir):
                #print phi
                phase = cmath.exp(phi*1j)
                phase_add = packet.rxchannel[num].beam_directions[bnum].phase.add()
                packet.rxchannel[num].beam_directions[bnum].phase[pnum].real_phase = phase.real
                packet.rxchannel[num].beam_directions[bnum].phase[pnum].imag_phase = phase.imag


    # Don't need channel numbers, always send 20 beam directions
    # Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel.

    procsocket.send(packet.SerializeToString());
    return


def get_ack(xsocket,procpacket): 
    """
    
    """
    try:
        x=xsocket.recv(flags=zmq.NOBLOCK)
        procpacket.ParseFromString(x)
        return procpacket.sequence_num
        print procpacket.sequence_num
    except zmq.Again:
        print "ACK ERROR"
        return


def radar():
    """
    Receives an instance of an experiment. Iterates through
    the Scans, AveragingPeriods, Sequences, and pulses of the experiment. 
    For every pulse, samples and other control information are sent to the n200_driver.
    For every pulse sequence, processing information is sent to the signal processing block.
    After every integration time (AveragingPeriod), the experiment block is given the opportunity
    to change the control program (if it sends a new one, runradar will halt the old one and begin with
    the new ControlProg.
    """

    # Setup socket to send pulse samples over.
    txsocket=setup_driver_socket()
    # Initialize driverpacket.
    driverpacket=driverpacket_pb2.DriverPacket()

    procsocket=setup_sigproc_params_socket()
    sigprocpacket=sigprocpacket_pb2.SigProcPacket()
    proctimesocket=setup_sigproc_timing_ack_socket()
    proccpsocket=setup_sigproc_cpack_socket()

    cpsocket=setup_cp_socket()
    # seqnum is used as a identifier in all packets while
    # radar is running so set it up here.
    # seqnum will get increased by nave at the end of
    # every integration time.
    seqnum_start = 0

    status = radar_status.RadarStatus()

    cpsocket.send_pyobj(status)
    should_poll = True
    while should_poll:
        poller=zmq.Poller()
        poller.register(cpsocket, zmq.POLLIN)
        cpsocks = dict(poller.poll(10)) #polls for 3 ms, NOTE this is before inttime timer starts.
        if cpsocket in cpsocks: #
            if cpsocks[cpsocket] == zmq.POLLIN:
                new_cp = get_prog(cpsocket) # TODO: write this function
                if new_cp == None:
                    print "NO NEW CP"
                elif isinstance(new_cp, normalscan.Normalscan): # is this how to check if it's a correct class type?
                    # TODO: scheduler
                    prog = new_cp
                    beam_remaining = False
                    updated_cp_received = True
                    print "NEW CP!!"
                    status.status = 1
                    should_poll = False
        else:
            print "No CP - keep polling"


    while True:

        cpos=prog.cpo_list
        updated_cp_received = False

        # Wavetables are currently None for sine waves, instead just
        #   use a sampling freq in rads/sample
        wavetable_dict={}
        for cpo in range(prog.cponum):
            wavetable_dict[cpo]=get_wavetables(prog.cpo_list[cpo]['wavetype'])

        # Iterate through Scans, AveragingPeriods, Sequences, Pulses.
        for scan in prog.scan_objects:
            if updated_cp_received == True:
                break
            beam_remaining=True
            # Make iterator for cycling through beam numbers
            scan_iter=0
            scans_done=0
            while (beam_remaining and not updated_cp_received):
                for aveperiod in scan.aveperiods:
                # If there are multiple aveperiods in a scan they are alternated
                #   beam by beam. So we need to iterate through
                # Go through each aveperiod once then increase the scan
                #   iterator to the next beam in each scan.

                    # poll for new cp here, before starting a new integration.
                    cpsocket.send_pyobj(status)
                    poller=zmq.Poller()
                    poller.register(cpsocket, zmq.POLLIN)
                    cpsocks = dict(poller.poll(3)) #polls for 3 ms, NOTE this is before inttime timer starts.
                    if cpsocket in cpsocks: #
                        if cpsocks[cpsocket] == zmq.POLLIN:
                            new_cp = get_prog(cpsocket) # TODO: write this function
                            if new_cp == None:
                                print "NO NEW CP"
                            elif isinstance(new_cp,controlprog.ControlProg): # is this how to check if it's a correct class type?
                                prog = new_cp
                                updated_cp_received = True
                                print "NEW CP!!"
                                break

                    if scan_iter>=len(scan.scan_beams[aveperiod.keys[0]]):
                    # All keys will have the same length scan_beams
                    #   inside the aveperiod, but not necessarily all aveperiods
                    #   in the scan will have same length scan_beams so we have to
                    #   record how many scans are done.
                        # TODO: Fix this to record in a list which aveperiods are done
                        # so we do not increase scans_done for same aveperiod
                        scans_done=scans_done+1
                        if scans_done==len(scan.aveperiods):
                            beam_remaining=False
                            break
                        continue
                    print "New AveragingPeriod"
                    int_time=datetime.utcnow()
                    time_remains=True
                    done_time=int_time+timedelta(0,float(aveperiod.intt)/1000)
                    nave=0
                    beamdir={}
                    # Create a dictionary of beam directions with the
                    #   keys being the cpos in this averaging period.
                    for cpo in aveperiod.keys:
                        bmnums=scan.scan_beams[cpo][scan_iter]
                        beamdir[cpo]=[]
                        if type(bmnums) == int:
                            beamdir[cpo]=scan.beamdir[cpo][bmnums]
                        else: # is a list
                            for bmnum in bmnums:
                                beamdir[cpo].append(scan.beamdir[cpo][bmnum])
                        # Get the beamdir from the beamnumber for this
                        #    CP-object at this iteration.
                    
                    # Create a pulse dictionary before running through the
                    #   averaging period.
                    sequence_dict_list=[]
                    for sequence in aveperiod.integrations:
                        # create pulse dictionary.
                        # use pulse_list as dictionary keys.
                        sequence_dict_list.append({})
                        # Just alternating sequences
                        #print sequence.pulse_time
                        for pulse_index in range(0, len(sequence.combined_pulse_list)):
                            # Pulses are in order
                            pulse_list=sequence.combined_pulse_list[pulse_index][:]
                            if pulse_index==0:
                                startofburst=True
                            else:
                                startofburst=False
                            if pulse_index==len(sequence.combined_pulse_list)-1:
                                endofburst=True
                            else:
                                endofburst=False
                            repeat=sequence.combined_pulse_list[pulse_index][0]
                            isamples_list=[]
                            qsamples_list=[]
                            if repeat:
                                pulse_channels=[]
                            else:
                                # Initialize a list of lists for
                                #   samples on all channels.
                                pulse_list.pop(0) # remove boolean repeat value
                                timing=pulse_list[0]
                                pulse_list.pop(0)
                                # TODO:need to determine what power
                                #   to use - should determine using
                                #   number of frequencies in
                                #   sequence, but for now use # of
                                #   pulses combined here.
                                power_divider=len(pulse_list)
                                print "POWER DIVIDER: {}".format(power_divider)
                                pulse_samples, pulse_channels = (
                                    make_pulse_samples(pulse_list, cpos,
                                        beamdir, prog.txctrfreq,
                                        prog.txrate, power_divider))
                                # Can plot for testing here
                                #plot_samples('channel0.png',
                                #    pulse_samples[0])
                                #plot_fft('fftplot.png', pulse_samples[0],
                                #    prog.txrate)
                                for channel in pulse_channels:
                                    isamples_list.append((pulse_samples
                                        [channel].real).tolist())
                                    qsamples_list.append((pulse_samples
                                        [channel].imag).tolist())

                            # Add to dictionary at last place in list (which is
                            #   the current sequence location in the list)
                            # This the the pulse_data.
                            sequence_dict_list[-1][pulse_index]=[startofburst,
                                endofburst, pulse_channels,
                                isamples_list, qsamples_list]
                    while (time_remains):
                        for sequence in aveperiod.integrations:
                            poll_timeout = int(sequence.seqtime/1000) + 1 # seqtime is in us, need ms
                            if datetime.utcnow()>=done_time:
                                time_remains=False
                                break
                            beam_phase_dict = {}
                            for cpo in sequence.cpos:
                                beam_phase_dict[cpo]=[]
                                if type(beamdir[cpo]) != list:
                                    phase_array = []
                                    for channel in range(0,16):
                                        # Get phase shifts for all channels
                                        phase_array.append(get_phshift(
                                                                beamdir[cpo],
                                                                prog.cpo_list[cpo]['txfreq'],channel,
                                                                0))
                                    for channel in range(6,9): # interferometer
                                        # Get phase shifts for all channels
                                        phase_array.append(get_phshift(
                                                                beamdir[cpo],
                                                                prog.cpo_list[cpo]['txfreq'],channel,
                                                                0)) # zero pulse shift b/w pulses when beamforming.
                                    beam_phase_dict[cpo].append(phase_array)
                                else:
                                    for beam in beamdir[cpo]:
                                        phase_array = []
                                        for channel in range(0,16):
                                            # Get phase shifts for all channels
                                            phase_array.append(get_phshift(
                                                                    beam,
                                                                    prog.cpo_list[cpo]['txfreq'],channel,
                                                                    0))
                                        for channel in range(6,9): # interferometer
                                            # Get phase shifts for all channels
                                            phase_array.append(get_phshift(
                                                                    beam,
                                                                    prog.cpo_list[cpo]['txfreq'],channel,
                                                                    0)) # zero pulse shift b/w pulses when beamforming.
                                        beam_phase_dict[cpo].append(phase_array)
                            data_to_processing(sigprocpacket, procsocket, seqnum_start + nave, sequence.cpos, prog.cpo_list, beam_phase_dict) # beamdir is dictionary
                            # Just alternating sequences
                            #print sequence.pulse_time
                            print sequence.combined_pulse_list

                            #
                            #
                            # SEND ALL PULSES IN SEQUENCE.
                            #
                            for pulse_index in range(0, len(sequence.combined_pulse_list)):
                                pulse_list=sequence.combined_pulse_list[pulse_index]
                                repeat=pulse_list[0]
                                pulse_data = sequence_dict_list[
                                    aveperiod.integrations.index(sequence)][
                                    pulse_index]
                                if repeat:
                                    ack = data_to_driver(
                                        driverpacket, txsocket, [], [], [], 0,
                                        0, 0, 0, pulse_data[0],
                                        pulse_data[1],
                                        pulse_list[1], seqnum_start + nave, repeat=True)
                                else:
                                    ack = data_to_driver(
                                        driverpacket, txsocket,
                                        pulse_data[2], #pulse_channels
                                        pulse_data[3], #isamples_list
                                        pulse_data[4], #qsamples_list
                                        prog.txctrfreq, prog.rxctrfreq,
                                        prog.txrate, sequence.numberofreceivesamples,
                                        pulse_data[0], #startofburst
                                        pulse_data[1], #endofburst,
                                        pulse_list[1], seqnum_start + nave, repeat=False)
                                # Pulse is done.

                            # Get sequence acknowledgements and log
                            # synchronization and communication errors between
                            # the n200_driver, rx_sig_proc, and radar_control.
                            if seqnum_start + nave != 0:
                                poller2=zmq.Poller()
                                poller2.register(txsocket, zmq.POLLIN)
                                poller2.register(proccpsocket, zmq.POLLIN)
                                should_poll = True
                                while should_poll:
                                    #print "Polling for {} - why is it not polling this long?".format(poll_timeout)
                                    socks = dict(poller2.poll(poll_timeout + 23000)) # get two messages with timeout of 100 ms
                                    if proccpsocket in socks and txsocket in socks: # need one message from both.
                                        if socks[proccpsocket] == zmq.POLLIN:
                                            rxseqnum = get_ack(proccpsocket,sigprocpacket)
                                            if rxseqnum != seqnum_start + nave - 1:
                                                print "**********************8ERROR: Wrong rxseqnum {} != {}".format(rxseqnum, seqnum_start+nave-1)
                                            else:
                                                print "RXSEQNUM {}".format(rxseqnum)
                                        if socks[txsocket] == zmq.POLLIN:
                                            txseqnum = get_ack(txsocket,driverpacket)
                                            if txseqnum != seqnum_start + nave:
                                                print "*********************ERROR: wrong txseqnum {} != {}".format(txseqnum, seqnum_start+nave)
                                                # TODO: LOG ERRORS,break as required
                                            else:
                                                print "TXSEQNUM {}".format(txseqnum)
                                        should_poll = False
                                    else:
                                        pass
                                        #print "******************ERROR: Have not received both ACKS"
                            else: # on the very first sequence since starting radar.
                                poller=zmq.Poller()
                                poller.register(txsocket, zmq.POLLIN)
                                print "Polling for {}".format(poll_timeout)
                                sock = poller.poll(poll_timeout + 23000)
                                try:
                                    txsocket.recv(flags=zmq.NOBLOCK)
                                    print "FIRST ACK RECEIVED"
                                except zmq.Again:
                                    print "No first ack from driver - This shouldn't happen"
                                    # TODO: Log error because no ack returned from driver on first send.
                            # Now, check how long the kernel is taking to process (non-blocking)
                            try:
                                kernel_time_ack=proctimesocket.recv(flags=zmq.NOBLOCK)
                            except zmq.Again:
                                pass # TODO: Should do something if don't receive kernel_time_ack for many sequences.


                            # TODO: Make sure you can have a CPO that doesn't transmit, only receives on a frequency.
                            time.sleep(1)
                            # Sequence is done
                            nave = nave + 1
                        #print "updating time"
                        #int_time=datetime.utcnow()
                    print "Number of integrations: {}".format(nave)
                    seqnum_start += nave
                scan_iter=scan_iter+1


radar()

