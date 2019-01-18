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

sys.path.append(os.environ["BOREALISPATH"])
from experiment_prototype.experiment_exception import ExperimentException
from utils.experiment_options.experimentoptions import ExperimentOptions

if __debug__:
    debug_path = os.environ["BOREALISPATH"] + '/testing/tmp/'
    if not os.path.isdir(debug_path):
        os.mkdir(debug_path)
    sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')
else:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')

from driverpacket_pb2 import DriverPacket
from sigprocpacket_pb2 import SigProcPacket
from datawritemetadata_pb2 import IntegrationTimeMetadata

from sample_building.sample_building import azimuth_to_antenna_offset, create_debug_sequence_samples
from experiment_prototype.experiment_prototype import ExperimentPrototype

from radar_status.radar_status import RadarStatus
from utils.zmq_borealis_helpers import socket_operations


def printing(msg):
    """
    :param msg:
    :return:
    """
    RADAR_CONTROL = "\033[33m" + "RADAR_CONTROL: " + "\033[0m"
    sys.stdout.write(RADAR_CONTROL + msg + "\n")


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
    driverpacket.numberofreceivesamples = numberofreceivesamples


    if repeat:
        # antennas empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        if __debug__:
            print("REPEAT; TIMING: {0}; SOB: {1}; EOB: {2}; ANTENNAS: {3}"
                  ";".format(timing, SOB, EOB, antennas))
    else:
        # SETUP data to send to driver for transmit.
        for ant in antennas:
            driverpacket.channels.append(ant)
        for samples in samples_array:
            sample_add = driverpacket.channel_samples.add()
            # Add one Samples message for each channel.
            # Protobuf expects types: int, long, or float, will reject numpy types and
            # throw a TypeError so we must convert the numpy arrays to lists
            sample_add.real.extend(samples.real.tolist())
            sample_add.imag.extend(samples.imag.tolist())
        driverpacket.txcenterfreq = txctrfreq * 1000  # convert to Hz
        driverpacket.rxcenterfreq = rxctrfreq * 1000  # convert to Hz
        driverpacket.txrate = txrate
        driverpacket.numberofreceivesamples = numberofreceivesamples
        if __debug__:
            print("NOT A REPEAT; TIMING: {0}; SOB: {1}; EOB: {2}; ANTENNAS: {3}"
                  ";".format(timing, SOB, EOB, antennas))

   # txsocket.send(driverpacket.SerializeToString())
    socket_operations.send_pulse(radctrl_to_driver, driver_to_radctrl_iden, driverpacket.SerializeToString())

    del driverpacket.channel_samples[:]  # TODO find out - Is this necessary in conjunction with .Clear()?



def send_metadata(packet, radctrl_to_dsp, dsp_radctrl_iden, radctrl_to_brian,
                   brian_radctrl_iden, seqnum, slice_ids,
                   slice_dict, beam_dict, sequence_time, main_antenna_count):
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
        :param sequence_time: entire duration of sequence, including receive time after all
        transmissions.
        :param main_antenna_count: number of main array antennas, from the config file.

    """

    # TODO: does the for loop below need to happen every time. Could be only updated
    # as necessary to make it more efficient.
    packet.Clear()
    packet.sequence_time = sequence_time
    packet.sequence_num = seqnum
    for num, slice_id in enumerate(slice_ids):
        chan_add = packet.rxchannel.add()
        chan_add.slice_id = slice_id
        if slice_dict[slice_id]['rxonly']:
            chan_add.rxfreq = slice_dict[slice_id]['rxfreq'] * 1.0e3
        elif slice_dict[slice_id]['clrfrqflag']:
            pass  # TODO - get freq from clear frequency search.
        else:
            chan_add.rxfreq = slice_dict[slice_id]['txfreq'] * 1.0e3
        chan_add.nrang = slice_dict[slice_id]['nrang']
        chan_add.frang = slice_dict[slice_id]['frang']
        for beamdir in beam_dict[slice_id]:
            beam_add = chan_add.beam_directions.add()
            # beamdir is a list (len = total antennas, main and interferometer) with phase for each
            # antenna for that beam direction
            for antenna_num, phi in enumerate(beamdir):
                phase_add = beam_add.phase.add()
                if antenna_num in slice_dict[slice_id]['rx_main_antennas'] or antenna_num - main_antenna_count in slice_dict[slice_id]['rx_int_antennas']:
                    phase = cmath.exp(-1 * phi * 1j)
                else:
                    phase = 0.0 + 0.0j
                phase_add.real_phase = phase.real
                phase_add.imag_phase = phase.imag

    # Don't need to send channel numbers, will always send with length = total antennas.
    # TODO : Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel. ** make this update phase = 0 on channels not included.

    # Brian requests sequence metadata for timeouts

    request = socket_operations.recv_request(radctrl_to_brian, brian_radctrl_iden,
                                             printing)
    if __debug__:
        request_output = "Brian requested -> {}".format(request)
        printing(request_output)

    bytes_packet = packet.SerializeToString()

    socket_operations.send_obj(radctrl_to_brian, brian_radctrl_iden, bytes_packet)

    #Radar control receives request for metadata from DSP
    # request = socket_operations.recv_request(radctrl_to_dsp, dsp_radctrl_iden, printing)
    if __debug__:
        request_output = "DSP requested -> {}".format(request)
        printing(request_output)

    socket_operations.send_obj(radctrl_to_dsp, dsp_radctrl_iden, packet.SerializeToString())
    # TODO : is it necessary to do a del packet.rxchannel[:] - test


def search_for_experiment(radar_control_to_exp_handler,
                          exphan_to_radctrl_iden,
                          status):
    """
    Check for new experiments from the experiment handler
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
                                               printing)
    except zmq.ZMQBaseError as e:
        errmsg = "ZMQ ERROR"
        raise [ExperimentException(errmsg), e]

    new_exp = pickle.loads(serialized_exp)  # protocol detected automatically

    if isinstance(new_exp, ExperimentPrototype):
        experiment = new_exp
        new_experiment_received = True
        if __debug__:
            print("NEW EXPERIMENT FOUND")
    elif new_exp != None:
        if __debug__:
            print("RECEIVED AN EXPERIMENT NOT OF TYPE EXPERIMENT_PROTOTYPE. CANNOT RUN.")
    else:
        if __debug__:
            print("The experiment was not updated - continuing.")
        # TODO decide what to do here. I think we need this case if someone doesn't build their experiment
        # properly

    return new_experiment_received, experiment



def send_datawrite_metadata(packet, radctrl_to_datawrite, datawrite_radctrl_iden,
                            seqnum, nave, scan_flag, inttime, sequences, beamdir_dict,
                            txctrfreq, rxctrfreq, experiment_id, experiment_string,
                            debug_samples=None):
    """
    Send the metadata about this integration time to datawrite so that it can be recorded.
    :param packet: The IntegrationTimeMetadata protobuf packet.
    :param radctrl_to_datawrite: The socket to send the packet on.
    :param datawrite_radctrl_iden: Identity of datawrite on the socket.
    :param seqnum: The last sequence number (identifier) that is valid for this integration
    period. Used to verify and synchronize driver, dsp, datawrite.
    :param nave: The number of sequences that were sent in this integration period. (number of
    averages).
    :param scan_flag: True if this integration period is the first in a scan.
    :param inttime: The time that expired during this integration period.
    :param sequences: The sequences of class Sequence for this integration period (
    AveragingPeriod).
    :param beamdir_dict: Dictionary where each slice_id key corresponds to a list of beam
    directions for that slice for this integration period.
    :param experiment_id: the ID of the experiment that is running
    :param experiment_string: the experiment string to be placed in the data files.
    :param debug_samples: the debug samples for this integration period, to be written to the
    file if debug is set. This is a list of dictionaries for each Sequence in the
    AveragingPeriod. The dictionary is set up in the sample_building module function
    create_debug_sequence_samples. The keys are 'txrate', 'txctrfreq', 'pulse_sequence_timing',
    'pulse_offset_error', 'sequence_samples', 'decimated_sequence', 'dmrate_error', and 'dmrate'.
    The 'sequence_samples' and 'decimated_sequence' values are themselves dictionaries, where the
    keys are the antenna numbers (there is a sample set for each transmit antenna).
    """

    packet.Clear()
    packet.experiment_id = experiment_id
    packet.experiment_string = experiment_string
    packet.nave = nave
    packet.last_seqn_num = seqnum
    packet.scan_flag = scan_flag
    packet.rx_center_freq = rxctrfreq
    packet.tx_center_freq = txctrfreq

    packet.integration_time = inttime.total_seconds()

    for sequence_index, sequence in enumerate(sequences):
        sequence_add = packet.sequences.add()
        sequence_add.blanks[:] = sequence.blanks
        if debug_samples:
            sequence_add.tx_data.txrate = debug_samples[sequence_index]['txrate']
            sequence_add.tx_data.txctrfreq = debug_samples[sequence_index]['txctrfreq']
            sequence_add.tx_data.pulse_sequence_timing_us[:] = debug_samples[sequence_index][
                'pulse_sequence_timing']
            sequence_add.tx_data.pulse_offset_error_us[:] = debug_samples[sequence_index][
                'pulse_offset_error']
            for ant, ant_samples in debug_samples[sequence_index]['sequence_samples'].items():
                tx_samples_add = sequence_add.tx_data.tx_samples.add()
                tx_samples_add.real[:] = ant_samples['real']
                tx_samples_add.imag[:] = ant_samples['imag']
                tx_samples_add.tx_antenna_number = ant
            sequence_add.tx_data.dmrate = debug_samples[sequence_index]['dmrate']
            sequence_add.tx_data.dmrate_error = debug_samples[sequence_index]['dmrate_error']
            for ant, ant_samples in debug_samples[sequence_index]['decimated_sequence'].items():
                tx_samples_add = sequence_add.tx_data.decimated_tx_samples.add()
                tx_samples_add.real[:] = ant_samples['real']
                tx_samples_add.imag[:] = ant_samples['imag']
                tx_samples_add.tx_antenna_number = ant
        for slice_id in sequence.slice_ids:
            rxchan_add = sequence_add.rxchannel.add()
            rxchan_add.slice_id = slice_id
            rxchan_add.frang = sequence.slice_dict[slice_id]['frang']
            rxchan_add.nrang = sequence.slice_dict[slice_id]['nrang']

            beamdirs = beamdir_dict[slice_id]
            for index, beamdir in enumerate(beamdirs):
                beam = rxchan_add.beams.add()
                beam.beamnum = sequence.slice_dict[slice_id]["beam_angle"].index(beamdir)
                beam.beamazimuth = beamdir

            rxchan_add.rx_only = sequence.slice_dict[slice_id]['rxonly']
            rxchan_add.pulse_len = sequence.slice_dict[slice_id]['pulse_len']
            rxchan_add.tau_spacing = sequence.slice_dict[slice_id]['mpinc']
            if sequence.slice_dict[slice_id]['rxonly']:
                rxchan_add.rxfreq = sequence.slice_dict[slice_id]['rxfreq']
            else:
                rxchan_add.rxfreq = sequence.slice_dict[slice_id]['txfreq']

            rxchan_add.ptab.pulse_position[:] = sequence.slice_dict[slice_id]['pulse_sequence']

            if sequence.slice_dict[slice_id]['acf']:
                rxchan_add.acf = sequence.slice_dict[slice_id]['acf']
                rxchan_add.xcf = sequence.slice_dict[slice_id]['xcf']
                rxchan_add.acfint = sequence.slice_dict[slice_id]['acfint']
                rxchan_add.rsep = sequence.slice_dict[slice_id]['rsep']
                for lag in sequence.slice_dict[slice_id]['lag_table']:
                    lag_add = rxchan_add.ltab.lag.add()
                    lag_add.pulse_position = lag
                    lag_add.lag_num = int(lag[1] - lag[0])

            rxchan_add.comment = sequence.slice_dict[slice_id]['comment']
            rxchan_add.interfacing = '{}'.format(sequence.slice_dict[slice_id]['slice_interfacing'])

            rxchan_add.rx_main_antennas[:] = sequence.slice_dict[slice_id]['rx_main_antennas']
            rxchan_add.rx_intf_antennas[:] = sequence.slice_dict[slice_id]['rx_int_antennas']

    if __debug__:
        printing('Sending metadata to datawrite.')

    socket_operations.send_bytes(radctrl_to_datawrite, datawrite_radctrl_iden,
                                 packet.SerializeToString())



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
    # seqnum will get increased by nave (number of averages or sequences in the integration period)
    # at the end of every integration time.
    seqnum_start = 0

    new_experiment_waiting = False

    while not new_experiment_waiting:  #  Wait for experiment handler at the start until we have an experiment to run.
        new_experiment_waiting, experiment = search_for_experiment(
            radar_control_to_exp_handler, options.exphan_to_radctrl_identity,
            'EXPNEEDED')

    new_experiment_waiting = False
    new_experiment_loaded = True

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

        for scan in experiment.scan_objects:

            # if a new experiment was received during the last scan, it finished the integration period it was on and
            # returned here with new_experiment_waiting set to True. Break to load new experiment.
            if new_experiment_waiting:  # start anew on first scan if we have a new experiment.
                break
            beam_remaining = True  # started a new scan, therefore this must be True.

            # Make iterator for cycling through beam numbers
            aveperiods_done_list = []
            beam_iter = 0
            while beam_remaining and not new_experiment_waiting:
                for aveperiod in scan.aveperiods:

                    # If there are multiple aveperiods in a scan they are alternated
                    #   beam by beam. So we need to iterate through
                    # Go through each aveperiod once then increase the beam
                    #   iterator to the next beam in each scan.

                    # get new experiment here, before starting a new integration.
                    # if new_experiment_waiting is set here, we will implement the new_experiment after this integration
                    # period.

                    # Check if there are beams remaining in this aveperiod, or in any aveperiods.
                    if aveperiod in aveperiods_done_list:
                        continue  # beam_iter index is past the end of the beam_order list for this aveperiod, but other aveperiods must still contain a beam at index beam_iter in the beam_order list.
                    else:
                        if beam_iter == len(scan.scan_beams[aveperiod.slice_ids[0]]):
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

                    if not new_experiment_waiting and not new_experiment_loaded: # there could already be a new experiment waiting, or we could have just loaded a new experiment.
                        new_experiment_waiting, new_experiment = search_for_experiment(
                            radar_control_to_exp_handler,
                            options.exphan_to_radctrl_identity, 'NOERROR')
                    elif new_experiment_loaded:
                        new_experiment_loaded = False

                    if __debug__:
                        print("New AveragingPeriod")

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


                    # Setup debug samples if in debug mode.
                    debug_samples = []
                    if __debug__:
                        for sequence_index, sequence in enumerate(aveperiod.sequences):
                            sequence_samples_dict = create_debug_sequence_samples(experiment.txrate,
                                                  experiment.txctrfreq,
                                                  sequence_dict_list[sequence_index],
                                                  debug_path,
                                                  options.main_antenna_count,
                                                  options.output_sample_rate,
                                                  sequence.ssdelay)
                            debug_samples.append(sequence_samples_dict)

                    integration_period_start_time = datetime.utcnow()  # ms
                    # all phases are set up for this averaging period for the beams required. Time to start averaging
                    # in the below loop.

                    nave = 0
                    time_remains = True
                    integration_period_done_time = integration_period_start_time + \
                        timedelta(milliseconds=(float(aveperiod.intt)))  # ms
                    first_sequence_out = False

                    while time_remains:
                        for sequence_index, sequence in enumerate(aveperiod.sequences):

                            # Alternating sequences if there are multiple in the averaging_period.
                            time_now = datetime.utcnow()
                            if time_now >= integration_period_done_time:
                                time_remains = False
                                integration_period_time = time_now - integration_period_start_time
                                break
                                # TODO add a break for nave == intn if going for number of averages instead of
                                # integration time
                            beam_phase_dict = beam_phase_dict_list[sequence_index]
                            send_metadata(sigprocpacket,
                                           radar_control_to_dsp,
                                           options.dsp_to_radctrl_identity,
                                           radar_control_to_brian,
                                           options.brian_to_radctrl_identity,
                                           seqnum_start + nave,
                                           sequence.slice_ids, experiment.slice_dict,
                                           beam_phase_dict, sequence.seqtime, options.main_antenna_count)

                            # beam_phase_dict is slice_id : list of beamdirs, where beamdir = list
                            # of antenna phase offsets for all antennas for that direction ordered
                            # [0 ... main_antenna_count, 0 ... interferometer_antenna_count]

                            # SEND ALL PULSES IN SEQUENCE.
                            # If we have sent first sequence already and there is only one unique
                            #  pulse in the averaging period, send all repeats until end of the
                            #  averaging period.
                            if first_sequence_out and aveperiod.one_pulse_only:
                                for pulse_index, pulse_dict in \
                                        enumerate(sequence_dict_list[sequence_index]):
                                    data_to_driver(driverpacket, radar_control_to_driver,
                                                   options.driver_to_radctrl_identity,
                                                   pulse_dict['pulse_antennas'],
                                                   pulse_dict['samples_array'], experiment.txctrfreq,
                                                   experiment.rxctrfreq, experiment.txrate,
                                                   sequence.numberofreceivesamples,
                                                   pulse_dict['startofburst'], pulse_dict['endofburst'],
                                                   pulse_dict['timing'], seqnum_start + nave,
                                                   repeat=True)
                            else:
                                for pulse_index, pulse_dict in \
                                        enumerate(sequence_dict_list[sequence_index]):
                                    data_to_driver(driverpacket, radar_control_to_driver,
                                                   options.driver_to_radctrl_identity,
                                                   pulse_dict['pulse_antennas'],
                                                   pulse_dict['samples_array'], experiment.txctrfreq,
                                                   experiment.rxctrfreq, experiment.txrate,
                                                   sequence.numberofreceivesamples,
                                                   pulse_dict['startofburst'], pulse_dict['endofburst'],
                                                   pulse_dict['timing'], seqnum_start + nave,
                                                   repeat=pulse_dict['isarepeat'])
                                first_sequence_out = True


                            # TODO: Make sure you can have a slice that doesn't transmit, only receives on a frequency. # REVIEW #1 what do you mean, what is this TODO for? REPLY : driver acks wouldn't be required etc need to make sure this is possible
                            # Sequence is done
                            nave = nave + 1

                    #if __debug__:
                    print("Number of integrations: {}".format(nave))

                    if beam_iter == 0:  # The first integration time in the scan.
                        scan_flag = True
                    else:
                        scan_flag = False

                    last_sequence_num = seqnum_start + nave - 1
                    send_datawrite_metadata(integration_time_packet, radar_control_to_dw,
                                            options.dw_to_radctrl_identity, last_sequence_num, nave,
                                            scan_flag, integration_period_time,
                                            aveperiod.sequences, slice_to_beamdir_dict,
                                            experiment.txctrfreq, experiment.rxctrfreq,
                                            experiment.cpid, experiment.comment_string,
                                            debug_samples)

                    # end of the averaging period loop - move onto the next averaging period.
                    # Increment the sequence number by the number of sequences that were in this
                    # averaging period.
                    seqnum_start += nave

                beam_iter = beam_iter + 1


if __name__ == "__main__":
    radar()

