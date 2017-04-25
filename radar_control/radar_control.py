#!/usr/bin/python

# radar_control.py
# 2017-03-13
# Marci Detwiller
# Get a radar control program made of objects (scans, averaging periods, and sequences).
# Communicate with the n200_driver to control the radar.
# Communicate with the rx_dsp_chain to process the data.

import sys
import cmath
from datetime import datetime, timedelta
import time

import zmq
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import kaiserord, lfilter, firwin, freqz

# from multiprocessing import Process, Value

sys.path.append('../build/release/utils/protobuf')  # REVIEW #6 TODO need to make this a dynamic import since
# 'release' may be 'debug'
# REPLY agreed will do
import driverpacket_pb2
import sigprocpacket_pb2

from sample_building import get_phshift, get_wavetables, make_pulse_samples

sys.path.append('../experiments')
import normalscan  # TODO - have it setup by scheduler

import radar_status


# REVIEW #1 Add description of params with units in docstring. Maybe find a docstring generator for vim.
def setup_driver_socket():  # to send pulses to driver. # REVIEW #38 could move this into docstring
    """
    Setup a zmq socket to communicate with the driver.
    """

    context = zmq.Context()  # REVIEW #33 Apparently it's best to just use one zmq context in the entire application - http://stackoverflow.com/questions/32280271/zeromq-same-context-for-multiple-sockets. So maybe have a global context or set it up in the main function and pass it to these functions.
    cpsocket = context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/0")
    return cpsocket


def setup_sigproc_params_socket():  # to send data to receive code. # REVIEW #38 could move this into docstring
    """
    Setup a zmq socket to communicate with rx_signal_processing.
    """

    context = zmq.Context()  # REVIEW #33
    cpsocket = context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/2")
    return cpsocket


def setup_sigproc_cpack_socket():  # to send data to receive code. # REVIEW #38 could move this into docstring
    """
    Setup a zmq socket to get acks from rx_signal_processing.
    """

    context = zmq.Context()  # REVIEW #33
    cpsocket = context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/3")
    return cpsocket


def setup_sigproc_timing_ack_socket():
    """
    Setup a zmq socket to get timing information from rx_signal_processing.
    """

    context = zmq.Context()  # REVIEW #33
    cpsocket = context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/4")
    return cpsocket


def setup_cp_socket():
    """
    Setup a zmq socket to get updated experiment parameters from the experiment.
    """

    context = zmq.Context()  # REVIEW #33
    cpsocket = context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/5")
    return cpsocket


# REVIEW #33 Can this be removed?
# def setup_ack_poller(socket1,socket2):
#    """
#
#    """
#
#    poller=zmq.Poller()
#    poller.register(socket1, zmq.POLLIN)
#    poller.register(socket2, zmq.POLLIN)
#    return poller


# def pollzmq(socket1, socket2, rxseqnum, txseqnum):
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


# REVIEW #39 Could make default arguments for repeat?
def data_to_driver(driverpacket, txsocket, channels, isamples_list,  # REVIEW #26 change channels to antennas
                   qsamples_list, txctrfreq, rxctrfreq, txrate,
                   numberofreceivesamples, SOB, EOB, timing, seqnum,
                   repeat=False):  # REVIEW #5 Add description of params with units in docstring. Maybe find a docstring generator for vim.
    """ Place data in the driver packet and send it via zeromq to the driver.
        Then receive the acknowledgement.
    """

    if repeat:  # REVIEW #1 Add detailed description to docstring on how the repeat functionality works.
        driverpacket.Clear()  # REVIEW #22 duplicated code can be moved above and below if/else
        # channels empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        driverpacket.timetosendsamples = timing  # REVIEW #22
        driverpacket.SOB = SOB  # REVIEW #22
        driverpacket.EOB = EOB  # REVIEW #22
        driverpacket.sequence_num = seqnum  # REVIEW #22
        print "EMPTY {0} {1} {2} {3}".format(timing, SOB, EOB, channels)
        # timetoio empty
    else:
        # SETUP data to send to driver for transmit.
        driverpacket.Clear()  # REVIEW #22
        for chan in channels:
            chan_add = driverpacket.channels.append(chan)  # REVIEW #33 this assignment is not needed?
        for chi in range(len(isamples_list)):  # REVIEW #26 Chi? Perhaps better name?
            sample_add = driverpacket.samples.add()  # REVIEW #0 sample_add not used. Can driverpacket.samples[chi] be replaced with sample_add? Do they point the the same object, because C++ works a like that.
            # Add one Samples message for each channel.
            real_samp = driverpacket.samples[chi].real.extend(
                isamples_list[chi])  # REVIEW #33 this assignment is not needed?
            imag_samp = driverpacket.samples[chi].imag.extend(
                qsamples_list[chi])  # REVIEW #33 this assignment is not needed?
        driverpacket.txcenterfreq = txctrfreq * 1000  # REVIEW #5 comment on unit conversion?
        driverpacket.rxcenterfreq = rxctrfreq * 1000  # REVIEW #5 comment on unit conversion?
        driverpacket.txrate = txrate
        driverpacket.numberofreceivesamples = numberofreceivesamples
        driverpacket.timetosendsamples = timing  # REVIEW #22
        driverpacket.sequence_num = seqnum  # REVIEW #22
        # Past time zero which is start of the sequence.
        print "New samples {0} {1} {2} {3}".format(timing, SOB, EOB, channels)
        driverpacket.SOB = SOB  # REVIEW #22
        driverpacket.EOB = EOB  # REVIEW #22

    txsocket.send(driverpacket.SerializeToString())
    # get response:
    # tx_ack = socket.recv() #REVIEW #6 Add todo for the response
    tx_ack = 1

    del driverpacket.samples[:]  # REVIEW #33 Is this needed in conjunction with .Clear()?

    return tx_ack


def data_to_processing(packet, procsocket, seqnum, cpos, cpo_list,
                       beam_dict):  # REVIEW #26 Perhaps better name now that dsp naming convention is figured out - send_data_to_rx_dsp?
    """ Place data in the receiver packet and send it via zeromq to the
        receiver unit.
    """

    packet.Clear()
    packet.sequence_num = seqnum
    for num, cpo in enumerate(cpos):
        channel_add = packet.rxchannel.add()  # REVIEW #0 channel_add not used. Same as above.
        packet.rxchannel[num].rxfreq = cpo_list[cpo]['rxfreq']
        packet.rxchannel[num].nrang = cpo_list[cpo]['nrang']
        packet.rxchannel[num].frang = cpo_list[cpo]['frang']
        for bnum, beamdir in enumerate(beam_dict[cpo]):
            beam_add = packet.rxchannel[num].beam_directions.add()  # REVIEW #0 beam_add not used. Same as above.
            # beamdir is a list 20-long with phase for each antenna for that beam direction. #REVIEW 1 This doesn't have to 20. Its just total antennas.
            for pnum, phi in enumerate(beamdir):
                # print phi
                phase = cmath.exp(phi * 1j)
                phase_add = packet.rxchannel[num].beam_directions[
                    bnum].phase.add()  # REVIEW #0 phase_add not used. Same as above.
                packet.rxchannel[num].beam_directions[bnum].phase[pnum].real_phase = phase.real
                packet.rxchannel[num].beam_directions[bnum].phase[pnum].imag_phase = phase.imag

    # Don't need channel numbers, always send 20 beam directions #REVIEW 1 This doesn't have to 20. Its just total antennas.
    # Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel.

    procsocket.send(packet.SerializeToString())
    return


def get_ack(xsocket, procpacket):  # REVIEW #26 what does the 'x' mean in xsocket? Could be more clear
    """
# REVIEW #1 Add docstring
    """
    try:
        x = xsocket.recv(flags=zmq.NOBLOCK)
        procpacket.ParseFromString(
            x)  # REVIEW #37 error check this - it might not throw a zmq.Again exception. You can use multiple except blocks after 1 try block
        return procpacket.sequence_num
        print procpacket.sequence_num  # REVIEW #33 this won't run, can remove
    except zmq.Again:
        print "ACK ERROR"
        return  # REVIEW #6 Here we return None implicity, can type "return None" to be more explicit, or do something else with the error, TODO?


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
    txsocket = setup_driver_socket()
    # Initialize driverpacket.
    driverpacket = driverpacket_pb2.DriverPacket()

    procsocket = setup_sigproc_params_socket()
    sigprocpacket = sigprocpacket_pb2.SigProcPacket()
    proctimesocket = setup_sigproc_timing_ack_socket()
    proccpsocket = setup_sigproc_cpack_socket()

    cpsocket = setup_cp_socket()
    # seqnum is used as a identifier in all packets while
    # radar is running so set it up here.
    # seqnum will get increased by nave at the end of # REVIEW #1 The wording here makes it sound like seqnum is increased once at the end of every integration time, not once every pulse sequence
    # every integration time.
    seqnum_start = 0

    status = radar_status.RadarStatus()

    cpsocket.send_pyobj(status)
    should_poll = True
    while should_poll:
        poller = zmq.Poller()  # REVIEW #0 Should these two lines really be in the while loop?
        poller.register(cpsocket, zmq.POLLIN)
        cpsocks = dict(poller.poll(
            10))  # polls for 3 ms, NOTE this is before inttime timer starts. # REVIEW #5 where do you find the units for poll? we assume that the comment is just outdated and should read "10 ms"
        if cpsocket in cpsocks:  #
            if cpsocks[cpsocket] == zmq.POLLIN:  # REVIEW #3 Why do you need to check if this is a zmq.POLLIN?
                new_cp = get_prog(cpsocket)  # TODO: write this function
                if new_cp == None:  # REVIEW #39 This if/elif can be slimmed down to one if isinstance(new_cp, normalscan.Normalscan)?, else statement could just be "No New CP" unless you're distinguising between new_cp == None and new_cp == something else
                    print "NO NEW CP"
                elif isinstance(new_cp,
                                normalscan.Normalscan):  # is this how to check if it's a correct class type? # REVIEW #5 TODO need to make this dynamic or check some other way (parent?)
                    # TODO: scheduler
                    prog = new_cp  # REVIEW #26 'prog' should now be 'experiment' or similar. same with 'cp'
                    beam_remaining = False  # REVIEW #0 Why is this set to False here, when it's being set to True at the top of the while True loop below?
                    updated_cp_received = True  # REVIEW #0 Why is this set to True here, when it's being set to False at the top of the while True loop below?
                    print "NEW CP!!"
                    status.status = 1  # REVIEW #26 What does this mean, better names for status'
                    should_poll = False  # REVIEW #33 This could just be a 'break' with the while loop using 'while True' since you're not doing anything else after this elif block is entered
        else:
            print "No CP - keep polling"

    while True:  # REVIEW #35 This is > 250 lines, should be refactored into smaller chunks

        cpos = prog.cpo_list
        updated_cp_received = False

        # Wavetables are currently None for sine waves, instead just
        #   use a sampling freq in rads/sample
        wavetable_dict = {}
        for cpo in range(prog.cponum):
            wavetable_dict[cpo] = get_wavetables(prog.cpo_list[cpo][
                                                     'wavetype'])  # REVIEW #6 #33 Is this not needed anymore or is there a TODO to implement type of wave somewhere? We noticed it wasn't used in this file and it is a local dictionary variable

        # Iterate through Scans, AveragingPeriods, Sequences, Pulses.
        for scan in prog.scan_objects:
            if updated_cp_received == True:
                break
            beam_remaining = True
            # Make iterator for cycling through beam numbers
            scan_iter = 0  # REVIEW #26 Seems more apt to name this beam_iter.
            scans_done = 0
            while (beam_remaining and not updated_cp_received):
                for aveperiod in scan.aveperiods:
                    # If there are multiple aveperiods in a scan they are alternated
                    #   beam by beam. So we need to iterate through
                    # Go through each aveperiod once then increase the scan
                    #   iterator to the next beam in each scan.

                    # poll for new cp here, before starting a new integration.
                    cpsocket.send_pyobj(
                        status)  # REVIEW #22 Duplicated code (above lines ~256-275), can be refactored into a function like 'search_for_experiment'
                    poller = zmq.Poller()
                    poller.register(cpsocket, zmq.POLLIN)
                    cpsocks = dict(poller.poll(3))  # polls for 3 ms, NOTE this is before inttime timer starts.
                    if cpsocket in cpsocks:  #
                        if cpsocks[cpsocket] == zmq.POLLIN:
                            new_cp = get_prog(cpsocket)  # TODO: write this function
                            if new_cp == None:
                                print "NO NEW CP"
                            elif isinstance(new_cp,
                                            controlprog.ControlProg):  # is this how to check if it's a correct class type?
                                prog = new_cp
                                updated_cp_received = True
                                print "NEW CP!!"
                                break

                    if scan_iter >= len(scan.scan_beams[aveperiod.keys[0]]):  # REVIEW #3 We just don't understand this.
                        # All keys will have the same length scan_beams
                        #   inside the aveperiod, but not necessarily all aveperiods
                        #   in the scan will have same length scan_beams so we have to
                        #   record how many scans are done.
                        # TODO: Fix this to record in a list which aveperiods are done
                        # so we do not increase scans_done for same aveperiod
                        scans_done = scans_done + 1
                        if scans_done == len(scan.aveperiods):
                            beam_remaining = False
                            break
                        continue
                    print "New AveragingPeriod"
                    int_time = datetime.utcnow()  # REVIEW #26 Name is confusing with respect to what int time is usually known as
                    time_remains = True
                    done_time = int_time + timedelta(0, float(
                        aveperiod.intt) / 1000)  # REVIEW #1 comment on unit conversion
                    nave = 0  # REVIEW #32 Put the initialization of nave right before it's used, i.e. right above the below while time_remains loop
                    beamdir = {}
                    # Create a dictionary of beam directions with the
                    #   keys being the cpos in this averaging period.
                    for cpo in aveperiod.keys:
                        bmnums = scan.scan_beams[cpo][
                            scan_iter]  # REVIEW #3 Do not understand how this could be an iterable???
                        beamdir[cpo] = []
                        if type(
                                bmnums) == int:  # REVIEW #39 should use isinstance instead http://stackoverflow.com/questions/707674/how-to-compare-type-of-an-object-in-python
                            beamdir[cpo] = scan.beamdir[cpo][
                                bmnums]  # REVIEW #33 why not just always have bmnums a list and then you can get rid of half this code, just keeping the for loop under the else clause.
                        else:  # is a list
                            for bmnum in bmnums:
                                beamdir[cpo].append(scan.beamdir[cpo][bmnum])
                                # Get the beamdir from the beamnumber for this
                                #    CP-object at this iteration.

                    # Create a pulse dictionary before running through the
                    #   averaging period.
                    sequence_dict_list = []
                    for sequence in aveperiod.integrations:
                        # create pulse dictionary.
                        # use pulse_list as dictionary keys.
                        sequence_dict_list.append({})
                        # Just alternating sequences
                        # print sequence.pulse_time
                        for pulse_index in range(0, len(sequence.combined_pulse_list)):
                            # Pulses are in order
                            pulse_list = sequence.combined_pulse_list[pulse_index][:]
                            if pulse_index == 0:
                                startofburst = True
                            else:
                                startofburst = False
                            if pulse_index == len(sequence.combined_pulse_list) - 1:
                                endofburst = True
                            else:
                                endofburst = False
                            repeat = sequence.combined_pulse_list[pulse_index][0]
                            isamples_list = []
                            qsamples_list = []
                            if repeat:
                                pulse_channels = []  # REVIEW #26 channel vs antenna name
                            else:
                                # Initialize a list of lists for
                                #   samples on all channels.
                                pulse_list.pop(0)  # remove boolean repeat value
                                timing = pulse_list[0]  # REVIEW #33 You can just do: 'timing = pulse_list.pop(0)'
                                pulse_list.pop(0)
                                # TODO:need to determine what power
                                #   to use - should determine using
                                #   number of frequencies in
                                #   sequence, but for now use # of
                                #   pulses combined here.
                                power_divider = len(
                                    pulse_list)  # REVIEW #6 We should make a test of the hardware (transmitter input level limits)
                                print "POWER DIVIDER: {}".format(power_divider)
                                pulse_samples, pulse_channels = (
                                    make_pulse_samples(pulse_list, cpos,
                                                       beamdir, prog.txctrfreq,
                                                       prog.txrate, power_divider))
                                # Can plot for testing here
                                # plot_samples('channel0.png',
                                #    pulse_samples[0])
                                # plot_fft('fftplot.png', pulse_samples[0],
                                #    prog.txrate)
                                for channel in pulse_channels:  # REVIEW #1 explain why you need to make these into python lists (protobuf reasons?) Can you just leave it as .real and .imag?
                                    isamples_list.append((pulse_samples
                                                          [channel].real).tolist())
                                    qsamples_list.append((pulse_samples
                                                          [channel].imag).tolist())

                            # Add to dictionary at last place in list (which is
                            #   the current sequence location in the list)
                            # This the the pulse_data.
                            sequence_dict_list[-1][pulse_index] = [startofburst,
                                                                   # REVIEW #39 to make this more understandable: remove the sequence_dict_list.append({}) at the top of the outer for loop, create a blank dictionary 'sequence_dict' there instead, then here do: 'sequence_dict[pulse_index] = [...]' then after the for loop, do 'sequence_dict_list.append(sequence_dict)'
                                                                   endofburst, pulse_channels,
                                                                   isamples_list, qsamples_list]
                    while (
                    time_remains):  # REVIEW #32 Put the initialization of 'time_remains = True' right before it's used, i.e. right above this while loop
                        for sequence in aveperiod.integrations:  # REVIEW #39 is it possible to refactor this to use zip or enumerate because we noticed you're using aveperiod.integrations.index(sequence) down below. Then you would have 'for sequence_index, sequence in enumerate(aveperiod.integrations)'
                            poll_timeout = int(
                                sequence.seqtime / 1000) + 1  # seqtime is in us, need ms # REVIEW #1 explain why you need to add 1 to this (int rounds down?)
                            if datetime.utcnow() >= done_time:  # REVIEW #32 Put the initialization of done_time right before it's used, i.e. right above this while loop
                                time_remains = False
                                break
                            beam_phase_dict = {}
                            for cpo in sequence.cpos:
                                beam_phase_dict[cpo] = []
                                if type(beamdir[
                                            cpo]) != list:  # REVIEW #33 why check if this is a list? why not just have a list of one element and build up from there for more beam directions? then you can get rid of half this code and just keep the code under the else clause.
                                    phase_array = []
                                    for channel in range(0, 16):  # REVIEW #26 #29 channel /antenna magic 16 number
                                        # Get phase shifts for all channels
                                        phase_array.append(get_phshift(
                                            beamdir[cpo],
                                            prog.cpo_list[cpo]['txfreq'], channel,
                                            0))
                                    for channel in range(6,
                                                         9):  # interferometer # REVIEW #0 #1, #29 #35 should be 6,10 to get 6,7,8,9, also explain why you are going for channels 6 through 9, also magic numbers - also make a function to phase main array as well a function for phasing int array, decouple them. Potentially means you need a second set of variables for the interferometer such as beamdir - alternatively you could have the beamdir and other variables 20 long for both int and main antennas..
                                        # Get phase shifts for all channels
                                        phase_array.append(get_phshift(
                                            beamdir[cpo],
                                            prog.cpo_list[cpo]['txfreq'], channel,
                                            0))  # zero pulse shift b/w pulses when beamforming.
                                    beam_phase_dict[cpo].append(phase_array)
                                else:
                                    for beam in beamdir[cpo]:
                                        phase_array = []
                                        for channel in range(0, 16):
                                            # Get phase shifts for all channels
                                            phase_array.append(get_phshift(
                                                beam,
                                                prog.cpo_list[cpo]['txfreq'], channel,
                                                0))
                                        for channel in range(6, 9):  # interferometer
                                            # Get phase shifts for all channels
                                            phase_array.append(get_phshift(
                                                beam,
                                                prog.cpo_list[cpo]['txfreq'], channel,
                                                0))  # zero pulse shift b/w pulses when beamforming.
                                        beam_phase_dict[cpo].append(phase_array)
                            data_to_processing(sigprocpacket, procsocket, seqnum_start + nave, sequence.cpos,
                                               prog.cpo_list, beam_phase_dict)  # beamdir is dictionary
                            # Just alternating sequences
                            # print sequence.pulse_time
                            print sequence.combined_pulse_list

                            #
                            #
                            # SEND ALL PULSES IN SEQUENCE.
                            #
                            for pulse_index in range(0, len(
                                    sequence.combined_pulse_list)):  # REVIEW #35 maybe this for loop can be refactored into a send_data_to_driver function along with the data_to_driver code
                                pulse_list = sequence.combined_pulse_list[pulse_index]
                                repeat = pulse_list[0]
                                pulse_data = sequence_dict_list[
                                    aveperiod.integrations.index(sequence)][
                                    # REVIEW #39 Here is where you could use sequence_index as discussed above
                                    pulse_index]
                                if repeat:  # REVIEW #33 don't really need the if/else here since the data_to_driver function will ignore the empty and none data. this goes along with the comment above about refactoring both this loop and data_to_driver into one function
                                    ack = data_to_driver(
                                        driverpacket, txsocket, [], [], [], 0,
                                        0, 0, 0, pulse_data[0],
                                        pulse_data[1],
                                        pulse_list[1], seqnum_start + nave,
                                        repeat=True)  # REVIEW #33 can just put 'repeat' instead of 'repeat=...'
                                else:
                                    ack = data_to_driver(
                                        driverpacket, txsocket,
                                        pulse_data[2],  # pulse_channels
                                        pulse_data[3],  # isamples_list
                                        pulse_data[4],  # qsamples_list
                                        prog.txctrfreq, prog.rxctrfreq,
                                        prog.txrate, sequence.numberofreceivesamples,
                                        pulse_data[0],  # startofburst
                                        pulse_data[1],  # endofburst,
                                        pulse_list[1], seqnum_start + nave, repeat=False)
                                    # Pulse is done.

                            # Get sequence acknowledgements and log
                            # synchronization and communication errors between
                            # the n200_driver, rx_sig_proc, and radar_control.
                            if seqnum_start + nave != 0:  # REVIEW #35 make this if/else a function - call it something like acknowledge_completed_sequence/ verify_completed_sequence/ verify_something
                                poller2 = zmq.Poller()  # REVIEW #33 can these three lines be done only once or do they need to be repeated every sequence (and every time through the while(time_remains) loop)?
                                poller2.register(txsocket, zmq.POLLIN)
                                poller2.register(proccpsocket, zmq.POLLIN)
                                should_poll = True
                                while should_poll:  # REVIEW #33 could be while True with a break when should_poll is set to False
                                    # print "Polling for {} - why is it not polling this long?".format(poll_timeout)  # REVIEW #32 Put the initialization of poll_timeout right above where it is used (right above the if seqnum_start + nave != 0: statement)
                                    socks = dict(poller2.poll(
                                        poll_timeout + 23000))  # get two messages with timeout of 100 ms # REVIEW #29 23000, is this to get your correct timeout of 100ms or what is happening?
                                    if proccpsocket in socks and txsocket in socks:  # need one message from both.
                                        if socks[proccpsocket] == zmq.POLLIN:
                                            rxseqnum = get_ack(proccpsocket, sigprocpacket)
                                            if rxseqnum != seqnum_start + nave - 1:  # REVIEW #1 a comment should be added to explain the proper rxseqnum and txseqnum values (one has -1 one doesn't)
                                                print "**********************ERROR: Wrong rxseqnum {} != {}".format(
                                                    rxseqnum,
                                                    seqnum_start + nave - 1)  # REVIEW #6 TODO for handling error and breaking as required
                                            else:
                                                print "RXSEQNUM {}".format(rxseqnum)
                                        if socks[txsocket] == zmq.POLLIN:
                                            txseqnum = get_ack(txsocket, driverpacket)
                                            if txseqnum != seqnum_start + nave:
                                                print "*********************ERROR: wrong txseqnum {} != {}".format(
                                                    txseqnum, seqnum_start + nave)
                                                # TODO: LOG ERRORS,break as required
                                            else:
                                                print "TXSEQNUM {}".format(txseqnum)
                                        should_poll = False
                                    else:
                                        pass
                                        # print "******************ERROR: Have not received both ACKS"
                            else:  # on the very first sequence since starting radar. # REVIEW #1 what does this comment imply?
                                poller = zmq.Poller()
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
                                kernel_time_ack = proctimesocket.recv(
                                    flags=zmq.NOBLOCK)  # REVIEW #5 units of the ack (ms?)
                            except zmq.Again:
                                pass  # TODO: Should do something if don't receive kernel_time_ack for many sequences.

                            # TODO: Make sure you can have a CPO that doesn't transmit, only receives on a frequency. # REVIEW #1 what do you mean, what is this TODO for?
                            time.sleep(1)  # REVIEW #33 this can be removed for release version
                            # Sequence is done
                            nave = nave + 1
                            # print "updating time"
                            # int_time=datetime.utcnow()
                    print "Number of integrations: {}".format(nave)
                    seqnum_start += nave
                scan_iter = scan_iter + 1


radar()  # REVIEW #39 this should be in a 'if __name__ == "__main__":' block otherwise it will run if you import this file
