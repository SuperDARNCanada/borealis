#!/usr/bin/python

# experiment.py
# 2016-11-17
# Marci Detwiller
# Get a radar control program and build it.
# Create pulse samples in sequences, in averaging periods, in scans.
# Communicate with the driver to control the radar.

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
sys.path.append('../experiment')
import myexperiment # this brings in myprog.

sys.path.append('./include')
from sample_building import get_phshift, get_wavetables, make_pulse_samples

import controlprog
import radar_status

def setup_driver_socket(): # to send pulses to driver.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/0")
    return cpsocket


def setup_sigproc_params_socket(): #to send data to receive code.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/2")
    return cpsocket


def setup_sigproc_cpack_socket(): #to send data to receive code.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/3")
    return cpsocket


def setup_sigproc_timing_ack_socket():
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/4")
    return cpsocket


def setup_cp_socket(): #to get refreshed control program updates.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/5")
    return cpsocket


def setup_ack_poller(socket1,socket2):
    poller=zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)
    return poller


def pollzmq(socket1, socket2, rxseqnum, txseqnum):
    poller=zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)
    socks = dict(poller.poll(100)) # get two messages
    if procsocket in socks and txsocket in socks: # need one message from both.
        if socks[procsocket] == zmq.POLLIN:
            rxseqnum = get_ack(procsocket)
        if socks[txsocket] == zmq.POLLIN:
            txseqnum = get_ack(txsocket)


def get_prog(socket):
#    update=json.dumps("UPDATE")
#    socket.send(update)
#    ack=socket.recv(zmq.NOBLOCK)
#    reply=json.loads(ack)
#    if reply=="YES":
#        socket.send(json.dumps("READY"))
#        new_prog=socket.recv(zmq.NOBLOCK)
#        prog=json.loads(new_prog)
#        return prog
#    #TODO: serialize a control program (class not JSON serializable)
#    elif reply=="NO":
#        print "no update"
#        return None
#
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
        driverpacket.sqnnum=seqnum
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
        driverpacket.sqnnum=seqnum
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
        packet.rxchannel[num].rxfreq = cpo_list[cpo].rxfreq
        packet.rxchannel[num].nrang = cpo_list[cpo].nrang
        packet.rxchannel[num].frang = cpo_list[cpo].frang
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
    #for chan in channels:
    #    receiverpacket.channels.append(chan)
    # Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel.

#    for i in range(0,len(rxfreqs)):
#        beam_array_add=receiverpacket.BeamDirections.add()
#        for phi in beamdirs[i,:]:
#            phase = math.exp(phi*1j)
#            receiverpacket.BeamDirections[i].phase.append(phase)
#
    # get response TODO
    procsocket.send(packet.SerializeToString());
    return


def get_ack(xsocket,procpacket):
    try:
        x=xsocket.recv(flags=zmq.NOBLOCK)
        procpacket.ParseFromString(x)
        return procpacket.sequence_num
        print procpacket.sequence_num
    except zmq.Again:
        print "ACK ERROR"
        return


def main():

    # Setup socket to send pulse samples over.
    txsocket=setup_driver_socket()
    # Initialize driverpacket.
    driverpacket=driverpacket_pb2.DriverPacket()

    procsocket=setup_sigproc_params_socket()
    sigprocpacket=sigprocpacket_pb2.SigProcPacket()
    proctimesocket=setup_sigproc_timing_ack_socket()
    proccpsocket=setup_sigproc_cpack_socket()

    cpsocket=setup_cp_socket()

#    seqpoller = setup_ack_poller(txsocket,procsocket)

    # seqnum is used as a identifier in all packets while
    # runradar is running so set it up here.
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
                elif isinstance(new_cp, controlprog.ControlProg): # is this how to check if it's a correct class type?
                    prog = new_cp
                    beam_remaining = False
                    updated_cp_received = True
                    print "NEW CP!!"
                    status.status = 1
                    should_poll = False
        else:
            print "No CP - keep polling"


    while True:
        # Receive pulse data from run_RCP
        #cpsocket=setup_cp_socket()
        #scan=None
        #while scan is None:
        #    prog=get_prog(cp_socket)
        # Build_RCP will reload after every scan.
        #print "got a prog"

        # Make myprog, defined in currentctrlprog
        # For now just import the experiment.
        #import experiment
        #prog = experiment.main()
        #prog=myexperiment.build_RCP()

        cpos=prog.cpo_list
        updated_cp_received = False

        # Wavetables are currently None for sine waves, instead just
        #   use a sampling freq in rads/sample
        wavetable_dict={}
        for cpo in range(prog.cpo_num):
            wavetable_dict[cpo]=get_wavetables(prog.cpo_list[cpo].wavetype)

        # TODO: dictionary of pulses so we aren't wasting time making
        #   them in actual scan. To do this we need to move out the
        #   phase and beam shifting though.
        #   This is not possible after combining multiple freqs

#        samples_dictionary={}
#        for scan in prog.scan_objects:
#            for aveperiod in scan.aveperiods:
#                for sequence in aveperiod.integrations:
#                    print sequence.combined_pulse_list
#                    for pulse_index in range(0, len(sequence.combined_pulse_list)):
#                        repeat=sequence.combined_pulse_list[pulse_index][0]
#                        if not repeat:
#                            # Initialize a list of lists for
#                            #   samples on all channels.
#                            pulse_list=sequence.combined_pulse_list[pulse_index][:]
#                            pulse_list.pop(0) # remove boolean repeat value
#                            pulse_list.pop(1)
#                            isamples_list=[]
#                            qsamples_list=[]
#                            # TODO:need to determine what power
#                            #   to use - should determine using
#                            #   number of frequencies in
#                            #   sequence, but for now use # of
#                            #   pulses combined here.
#                            power_divider=len(pulse_list)
#                            pulse_samples, pulse_channels = (
#                                make_pulse_samples(pulse_list, cpos,
#                                    beamdir, prog.txctrfreq,
#                                    prog.txrate, power_divider))
#                            # Plot for testing
#                            plot_samples('channel0.png',
#                                pulse_samples[0])
#                            plot_fft('fftplot.png', pulse_samples[0],
#                                prog.txrate)
#                            samples_dictionary[pulse_list]=[pulse_samples,pulse_channels]
#                            for channel in pulse_channels:
#                                isamples_list.append((pulse_samples
#                                    [channel].real).tolist())
#                                qsamples_list.append((pulse_samples
#                                    [channel].imag).tolist())

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
                    # TODO:send RX data for this averaging period (each
                    #   cpo will have own data file? (how stereo is
                    #   implemented currently))

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
                                # Plot for testing
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
                           # TODO: Consider where the best place to break should be for communication w/ driver and sigproc
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
                                                                prog.cpo_list[cpo].freq,channel,
                                                                0))
                                    for channel in range(6,9): # interferometer
                                        # Get phase shifts for all channels
                                        phase_array.append(get_phshift(
                                                                beamdir[cpo],
                                                                prog.cpo_list[cpo].freq,channel,
                                                                0)) # zero pulse shift b/w pulses when beamforming.
                                    beam_phase_dict[cpo].append(phase_array)
                                else:
                                    for beam in beamdir[cpo]:
                                        phase_array = []
                                        for channel in range(0,16):
                                            # Get phase shifts for all channels
                                            phase_array.append(get_phshift(
                                                                    beam,
                                                                    prog.cpo_list[cpo].freq,channel,
                                                                    0))
                                        for channel in range(6,9): # interferometer
                                            # Get phase shifts for all channels
                                            phase_array.append(get_phshift(
                                                                    beam,
                                                                    prog.cpo_list[cpo].freq,channel,
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
                            # the n200_driver, sig_proc, and runradar.
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
                                            txseqnum = get_ack(txsocket,sigprocpacket)
                                            if txseqnum != seqnum_start + nave:
                                                print "*********************ERROR: wrong txseqnum {} != {}".format(txseqnum, seqnum_start+nave)
                                                # TODO: LOG ERRORS,break as required
                                            else:
                                                print "TXSEQNUM {}".format(txseqnum)
                                        should_poll = False
                                    else:
                                        pass
                                        #print "******************ERROR: Have not received both ACKS"
                            else: # on the very first sequence since starting runradar.
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


main()

