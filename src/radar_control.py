#!/usr/bin/env python3

"""
    radar_control process
    ~~~~~~~~~~~~~~~~~~~~~
    Radar_control is the process that runs the radar (sends pulses to the driver with
    timing information and sends processing information to the signal processing process).
    Experiment_handler provides the experiment for radar_control to run. It iterates
    through the interface_class_base objects to control the radar.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
import os
import sys
import time
from datetime import datetime, timedelta
import zmq
import pickle
import threading
import numpy as np
from functools import reduce
from dataclasses import dataclass, field

from experiment_prototype.experiment_prototype import ExperimentPrototype
from utils.options import Options
import utils.message_formats as messages
from utils import socket_operations

sys.path.append(os.environ["BOREALISPATH"])
if __debug__:
    from build.debug.src.utils.protobuf.driverpacket_pb2 import DriverPacket
else:
    from build.release.src.utils.protobuf.driverpacket_pb2 import DriverPacket

TIME_PROFILE = False


@dataclass
class RadctrlParameters:
    """
    Class holding parameters that are passed between processes during the radar
    operation.
    """

    experiment: ...
    aveperiod: ...
    sequence: ...
    options: ...
    decimation_scheme: ... = field(init=False)
    seqnum_start: int
    startup_flag: bool
    num_sequences: int = 0
    last_sequence_num: int = 0
    sequence_index: int = 0
    start_time: datetime = datetime.utcnow()
    averaging_period_start_time: datetime = datetime.utcnow()
    averaging_period_time: timedelta = timedelta(seconds=0)
    debug_samples: list = field(default_factory=list)
    pulse_transmit_data_tracker: dict = field(default_factory=dict)
    slice_dict: dict = field(default_factory=dict, init=False)
    cfs_scan_flag: bool = False
    scan_flag: bool = False

    def __post_init__(self):
        self.slice_dict = self.experiment.slice_dict
        # Set slice_dict after an experiment has been assigned
        if self.sequence:
            self.decimation_scheme = self.sequence.decimation_scheme
        else:
            self.decimation_scheme = None
        # define decimation scheme only is a sequence was defined


def driver_comms_thread(radctrl_driver_iden, driver_socket_iden, router_addr):
    """
    Thread for handling communication between radar_control and rx_signal_processing.
    """

    driver_comms_socket = zmq.Context().instance().socket(zmq.PAIR)
    driver_comms_socket.connect("inproc://radctrl_driver")

    radctrl_driver_socket = socket_operations.create_sockets(
        [radctrl_driver_iden], router_addr
    )[0]

    first_time = True
    while True:
        message = driver_comms_socket.recv_pyobj()
        # Wait until the main thread sends a message intended for the driver

        socket_operations.send_bytes(radctrl_driver_socket, driver_socket_iden, message)
        # Send the message to the driver

        if first_time:
            socket_operations.recv_data(radctrl_driver_socket, driver_socket_iden, log)
            # Driver sends back an ACK on the first communication

            driver_comms_socket.send_string("Driver ready to go")
            # Let the main thread know that the driver is ready to go

            first_time = False


def create_driver_message(driver_params, pulse_transmit_data):
    """
    Place data in the driver packet and send it via zeromq to the driver.

    :param driver_params: The current averaging period parameters dataclass. The message extracts,
        * ``sequence.txctrfreq`` - the transmit center frequency to tune to
        * ``sequence.rxctrfreq`` - the receive center frequency to tune to. With rx_sample_rate from config.ini
          file, this determines the received signal band
        * ``experiment.txrate`` - the tx sampling rate (Hz)
        * ``experiment.rxrate`` - the rx sampling rate (Hz)
        * ``sequence.numberofreceivesamples`` - number of samples to receive at the rx_sample_rate from config.ini
          file. This determines length of Scope Sync GPIO being high for this sequence
        * ``sequence.seqtime`` - relative timing offset
        * ``seqnum`` - the sequence number. This is a unique identifier for the sequence that is always increasing
          with increasing sequences while radar_control is running. It is only reset when program restarts. It is
          determined as ``seqnum_start`` + ``num_sequences``
        * ``sequence.align_sequences`` - a boolean indicating whether to align the start of the sequence to a
          clean tenth of a second.
    :param pulse_transmit_data: dictionary of current transmit pulse data. The message extracts,
        * ``[samples_array]`` - this is a list of length main_antenna_count from the config file. It contains one
          numpy array of complex values per antenna. If the antenna will not be transmitted on, it contains a
          numpy array of zeros of the same length as the rest. All arrays will have the same length according to
          the pulse length
        * ``[startofburst]`` - start of burst boolean, true for first pulse in sequence
        * ``[endofburst]`` - end of burst boolean, true for last pulse in sequence
        * ``[timing]`` - in us, the time past timezero to send this pulse. Timezero is the start of the sequence
        * ``[isarepeat]`` - a boolean indicating whether the pulse is the exact same as the last pulse
          in the sequence, in which case we will save the time and not send the samples list and other
          params that will be the same
    """

    message = DriverPacket()

    # If this is the first time the driver is being set-up, only send tx and rx rates and center frequencies
    if driver_params.startup_flag:
        message.txcenterfreq = (
            driver_params.slice_dict[0].txctrfreq * 1000
        )  # convert to Hz
        message.rxcenterfreq = (
            driver_params.slice_dict[0].rxctrfreq * 1000
        )  # convert to Hz
        message.txrate = driver_params.experiment.txrate
        message.rxrate = driver_params.experiment.rxrate
    else:
        timing = pulse_transmit_data["timing"]
        SOB = pulse_transmit_data["startofburst"]
        EOB = pulse_transmit_data["endofburst"]

        message.timetosendsamples = timing
        message.SOB = SOB
        message.EOB = EOB
        message.sequence_num = driver_params.seqnum_start + driver_params.num_sequences
        message.numberofreceivesamples = driver_params.sequence.numberofreceivesamples
        message.seqtime = driver_params.sequence.seqtime
        message.align_sequences = driver_params.sequence.align_sequences

        if pulse_transmit_data["isarepeat"]:
            # antennas empty
            # samples empty
            # ctrfreq empty
            # rxrate and txrate empty
            log.debug("repeat timing", timing=timing, sob=SOB, eob=EOB)
        else:
            # Setup data to send to driver for transmit.
            log.debug("non-repeat timing", timing=timing, sob=SOB, eob=EOB)
            samples_array = pulse_transmit_data["samples_array"]
            for ant_idx in range(samples_array.shape[0]):
                sample_add = message.channel_samples.add()
                # Add one Samples message for each channel possible in config.
                # Any unused channels will be sent zeros.
                # Protobuf expects types: int, long, or float, will reject numpy types and
                # throw a TypeError so we must convert the numpy arrays to lists
                sample_add.real.extend(samples_array[ant_idx, :].real.tolist())
                sample_add.imag.extend(samples_array[ant_idx, :].imag.tolist())

            message.txcenterfreq = (
                driver_params.sequence.txctrfreq * 1000
            )  # convert to Hz
            message.rxcenterfreq = (
                driver_params.sequence.rxctrfreq * 1000
            )  # convert to Hz
            message.txrate = driver_params.experiment.txrate
            message.rxrate = driver_params.experiment.rxrate
            message.numberofreceivesamples = (
                driver_params.sequence.numberofreceivesamples
            )

    return message.SerializeToString()


def dsp_comms_thread(radctrl_dsp_iden, dsp_socket_iden, router_addr):
    """
    Thread for handling communication between radar_control and rx_signal_processing.
    """

    inproc_socket = zmq.Context().instance().socket(zmq.PAIR)
    inproc_socket.connect("inproc://radctrl")

    radctrl_dsp_socket = socket_operations.create_sockets(
        [radctrl_dsp_iden], router_addr
    )[0]

    while True:
        message = inproc_socket.recv_pyobj()
        # Wait for metadata from the main thread

        bytes_packet = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        socket_operations.send_obj(radctrl_dsp_socket, dsp_socket_iden, bytes_packet)
        # Send the message to rx_signal_processing

        socket_operations.recv_data(radctrl_dsp_socket, dsp_socket_iden, log)
        # Wait for rx_signal_processing to acknowledge that it received the metadata

        inproc_socket.send_string("DSP acknowledged metadata")
        # Let the main thread know that DSP has received the metadata


def create_dsp_message(dsp_params):
    """
    Place data in the receiver packet and send it via zeromq to the signal processing unit and brian.
    Happens every sequence.

    :param dsp_params: The radar control parameter dataclass updated during the averaging period. The message grabs,
        * ``experiment.rxrate`` - The receiver sampling rate (Hz)
        * ``experiment.output_rx_rate`` - The output sample rate desired for the output data (Hz)
        * ``seqnum`` - the sequence number. This is a unique identifier for the sequence that is always increasing
          with increasing sequences while radar_control is running. It is only reset when program restarts.
          It is calculated from ``seqnum_start`` + ``num_sequences``
        * ``sequence.slice_ids`` - The identifiers of the slices that are combined in this sequence.
          These IDs tell us where to look in the beam dictionary and slice dictionary for frequency information
          and beam direction information about this sequence to give to the signal processing unit
        * ``sequence.slice_dict`` - The slice dictionary, which contains information about all slices and will be
          referenced for information about the slices in this sequence. Namely, we get the frequency we want to
          receive at, the number of ranges and the first range information
        * ``beam_dict`` - The dictionary containing beam directions for each slice, generated using
          ``sequence.get_rx_phases(aveperiod.beam_iter)``
        * ``sequence.seqtime`` - entire duration of sequence, including receive time after all transmissions
        * ``sequence.first_rx_sample_start`` - The sample where the first rx sample will start relative to the
          tx data
        * ``sequence.rxctrfreq`` - the center frequency of receiving
        * ``sequence.output_encodings`` - Phase offsets (degrees) applied to each pulse in the sequence
        * ``decimation_scheme`` - object of type DecimationScheme that has all decimation and filtering data
        * ``cfs_scan_flag`` - flag indicating of sequence is a clear frequency search rx only sequence
    """
    # TODO: does the for loop below need to happen every time. Could be only updated
    #       as necessary to make it more efficient.

    message = messages.SequenceMetadataMessage()
    message.sequence_time = dsp_params.sequence.seqtime
    message.sequence_num = dsp_params.seqnum_start + dsp_params.num_sequences
    message.offset_to_first_rx_sample = dsp_params.sequence.first_rx_sample_start
    message.rx_rate = dsp_params.experiment.rxrate
    message.output_sample_rate = dsp_params.experiment.output_rx_rate
    message.rx_ctr_freq = dsp_params.sequence.rxctrfreq * 1.0e3
    message.cfs_scan_flag = dsp_params.cfs_scan_flag

    if dsp_params.decimation_scheme is not None:
        for stage in dsp_params.decimation_scheme.stages:
            dm_stage_add = messages.DecimationStageMessage(
                stage.stage_num, stage.input_rate, stage.dm_rate, stage.filter_taps
            )
            message.add_decimation_stage(dm_stage_add)

    slice_dict = dsp_params.sequence.slice_dict
    if dsp_params.aveperiod.cfs_flag:
        beam_dict = dsp_params.sequence.get_rx_phases(0)
    else:
        beam_dict = dsp_params.sequence.get_rx_phases(dsp_params.aveperiod.beam_iter)
    for slice_id in dsp_params.sequence.slice_ids:
        chan_add = messages.RxChannel(slice_id)
        chan_add.tau_spacing = slice_dict[slice_id].tau_spacing

        # Send the translational frequencies to dsp in order to bandpass filter correctly.
        chan_add.rx_freq = slice_dict[slice_id].freq * 1.0e3
        chan_add.num_ranges = slice_dict[slice_id].num_ranges
        chan_add.first_range = slice_dict[slice_id].first_range
        chan_add.range_sep = slice_dict[slice_id].range_sep
        chan_add.rx_intf_antennas = slice_dict[slice_id].rx_intf_antennas
        chan_add.pulses = slice_dict[slice_id].pulse_sequence

        main_bms = beam_dict[slice_id]["main"]
        intf_bms = beam_dict[slice_id]["intf"]

        # Combine main and intf such that for a given beam all main phases come first.
        beams = np.hstack((main_bms, intf_bms))
        chan_add.beam_phases = np.array(beams)

        for lag in slice_dict[slice_id].lag_table:
            lag_add = messages.Lag(lag[0], lag[1], int(lag[1] - lag[0]))

            # Get the phase offset for this pulse combination
            pulse_phase_offsets = dsp_params.sequence.output_encodings
            if len(pulse_phase_offsets[slice_id]) != 0:
                pulse_phase_offset = pulse_phase_offsets[slice_id][-1]
                lag0_idx = slice_dict[slice_id].pulse_sequence.index(lag[0])
                lag1_idx = slice_dict[slice_id].pulse_sequence.index(lag[1])
                phase_in_rad = np.radians(
                    pulse_phase_offset[lag0_idx] - pulse_phase_offset[lag1_idx]
                )
                phase_offset = np.exp(1j * np.array(phase_in_rad, np.float32))
            # Catch case where no pulse phase offsets are specified
            else:
                phase_offset = 1.0 + 0.0j

            lag_add.phase_offset_real = np.real(phase_offset)
            lag_add.phase_offset_imag = np.imag(phase_offset)
            chan_add.add_lag(lag_add)
        message.add_rx_channel(chan_add)

    return message


def search_for_experiment(radar_control_to_exp_handler, exphan_to_radctrl_iden, status):
    """
    Check for new experiments from the experiment handler

    :param radar_control_to_exp_handler: TODO
    :param exphan_to_radctrl_iden: The TODO
    :param status: status string (EXP_NEEDED or NO_ERROR).
    :returns new_experiment_received: boolean (True for new experiment received)
    :returns experiment: experiment instance (or None if there is no new experiment)
    """

    try:
        socket_operations.send_request(
            radar_control_to_exp_handler, exphan_to_radctrl_iden, status
        )
    except zmq.ZMQBaseError as e:
        log.error("zmq failed request", error=e)
        log.exception("zmq failed request", exception=e)
        sys.exit(1)

    experiment = None
    new_experiment_received = False

    try:
        serialized_exp = socket_operations.recv_exp(
            radar_control_to_exp_handler, exphan_to_radctrl_iden, log
        )
    except zmq.ZMQBaseError as e:
        log.error("zmq failed receive", error=e)
        log.exception("zmq failed receive", exception=e)
        sys.exit(1)

    new_exp = pickle.loads(serialized_exp)  # Protocol detected automatically

    if isinstance(new_exp, ExperimentPrototype):
        experiment = new_exp
        new_experiment_received = True
        log.info("new experiment found")
    elif new_exp is not None:
        log.debug("received non experiment_prototype type")
    else:
        log.debug("experiment continuing without update")
        # TODO decide what to do here. I think we need this case if someone doesn't build their experiment properly

    return new_experiment_received, experiment


def make_next_samples(sample_params):
    sqn, dbg = sample_params.sequence.make_sequence(
        sample_params.aveperiod.beam_iter,
        sample_params.num_sequences + len(sample_params.aveperiod.sequences),
    )
    if dbg:
        sample_params.debug_samples.append(dbg)
    sample_params.pulse_transmit_data_tracker[sample_params.sequence_index][
        sample_params.num_sequences + len(sample_params.aveperiod.sequences)
    ] = sqn

    if TIME_PROFILE:
        new_sequence_time = datetime.utcnow() - sample_params.start_time
        log.verbose(
            "make new sequence time",
            new_sequence_time=new_sequence_time,
            new_sequence_time_units="s",
        )


def dw_comms_thread(radctrl_dw_iden, dw_socket_iden, router_addr):
    """
    Thread for handling communication between radar_control and data write.
    """

    inproc_socket = zmq.Context().instance().socket(zmq.PAIR)
    inproc_socket.connect("inproc://radctrl_dw")

    radctrl_dw_socket = socket_operations.create_sockets(
        [radctrl_dw_iden], router_addr
    )[0]

    while True:
        message = inproc_socket.recv_pyobj()
        # Wait for metadata from the main thread

        bytes_packet = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        socket_operations.send_obj(radctrl_dw_socket, dw_socket_iden, bytes_packet)
        # Send the message to data write


def create_dw_message(dw_params):
    """
    Send the metadata about this averaging period to datawrite so that it can be recorded.

    :param dw_params: The radar control parameter dataclass updated during the aveperiod. The message grabs,
        * ``seqnum`` - The last sequence number (identifier) that is valid for this averaging period. Used
          to verify and synchronize driver, dsp, datawrite. Calculated with ``seqnum_start`` + ``num_sequences``
        * ``num_sequences``- The number of sequences that were sent in this averaging period. (number of
          sequences to average together)
        * ``scan_flag`` - True if this averaging period is the first in a scan
        * ``averaging_period_time`` - The time that expired during this averaging period
        * ``aveperiod.sequences`` - The sequences of class Sequence for this averaging period (AveragingPeriod)
        * ``aveperiod.beam_iter`` - The beam iterator of this averaging period
        * ``experiment.cpid`` - the ID of the experiment that is running
        * ``experiment.experiment_name`` - the experiment name to be placed in the data files
        * ``experiment.scheduling_mode`` - the type of scheduling mode running at this time, to write to file
        * ``experiment.output_rx_rate`` - The output sample rate of the output data, defined by the
          experiment, in Hz
        * ``experiment.comment_string`` - The comment string for the experiment, user-defined
        * ``decimation_scheme.filter_scaling_factors`` - The decimation scheme scaling factors used for the
          experiment, to get the scaling for the data for accurate power measurements between experiments
        * ``experiment.slice_dict[0].rxctrfreq`` - The receive center frequency (kHz)
        * ``debug_samples`` - the debug samples for this averaging period, to be written to the
          file if debug is set. This is a list of dictionaries for each Sequence in the
          AveragingPeriod. The dictionary is set up in the sample_building module function
          create_debug_sequence_samples. The keys are ``txrate``, ``txctrfreq``, ``pulse_timing`,
          ``pulse_sample_start``, ``sequence_samples``, ``decimated_sequence``, and ``dmrate``.
          The ``sequence_samples`` and ``decimated_samples`` values are themselves dictionaries, where the
          keys are the antenna numbers (there is a sample set for each transmit antenna)
    """

    message = messages.AveperiodMetadataMessage()
    message.experiment_id = dw_params.experiment.cpid
    message.experiment_name = dw_params.experiment.experiment_name
    message.experiment_comment = dw_params.experiment.comment_string
    message.rx_ctr_freq = dw_params.experiment.slice_dict[0].rxctrfreq
    message.num_sequences = dw_params.num_sequences
    message.last_sqn_num = dw_params.last_sequence_num
    message.scan_flag = dw_params.scan_flag
    message.aveperiod_time = dw_params.averaging_period_time.total_seconds()
    message.output_sample_rate = dw_params.experiment.output_rx_rate
    message.data_normalization_factor = reduce(
        lambda x, y: x * y, dw_params.decimation_scheme.filter_scaling_factors
    )  # multiply all
    message.scheduling_mode = dw_params.experiment.scheduling_mode

    for sequence_index, sequence in enumerate(dw_params.aveperiod.sequences):
        sequence_add = messages.Sequence()
        sequence_add.blanks = sequence.blanks

        if len(dw_params.debug_samples) > 0:
            tx_data = messages.TxData()
            tx_data.tx_rate = dw_params.debug_samples[sequence_index]["txrate"]
            tx_data.tx_ctr_freq = dw_params.debug_samples[sequence_index]["txctrfreq"]
            tx_data.pulse_timing_us = dw_params.debug_samples[sequence_index][
                "pulse_timing"
            ]
            tx_data.pulse_sample_start = dw_params.debug_samples[sequence_index][
                "pulse_sample_start"
            ]
            tx_data.tx_samples = dw_params.debug_samples[sequence_index][
                "sequence_samples"
            ]
            tx_data.dm_rate = dw_params.debug_samples[sequence_index]["dmrate"]
            tx_data.decimated_tx_samples = dw_params.debug_samples[sequence_index][
                "decimated_samples"
            ]
            sequence_add.tx_data = tx_data

        for slice_id in sequence.slice_ids:
            sqn_slice = sequence.slice_dict[slice_id]
            rxchannel = messages.RxChannelMetadata()
            rxchannel.slice_id = slice_id
            rxchannel.slice_comment = sqn_slice.comment
            rxchannel.interfacing = "{}".format(sqn_slice.slice_interfacing)
            rxchannel.rx_only = sqn_slice.rxonly
            rxchannel.pulse_len = sqn_slice.pulse_len
            rxchannel.tau_spacing = sqn_slice.tau_spacing
            rxchannel.rx_freq = sqn_slice.freq
            rxchannel.ptab = sqn_slice.pulse_sequence

            # We always build one sequence in advance, so we trim the last one from when radar
            # control stops processing the averaging period.
            for encoding in sequence.output_encodings[slice_id][
                : dw_params.num_sequences
            ]:
                rxchannel.add_sqn_encodings(encoding.flatten().tolist())
            sequence.output_encodings[slice_id] = []

            rxchannel.rx_main_antennas = sqn_slice.rx_main_antennas
            rxchannel.rx_intf_antennas = sqn_slice.rx_intf_antennas
            rxchannel.tx_antenna_phases = sequence.tx_main_phase_shifts[slice_id][
                dw_params.aveperiod.beam_iter
            ]

            beams = sqn_slice.rx_beam_order[dw_params.aveperiod.beam_iter]
            if isinstance(beams, int):
                beams = [beams]

            rx_main_phases = []
            rx_intf_phases = []
            for beam in beams:
                beam_add = messages.Beam(sqn_slice.beam_angle[beam], beam)
                rxchannel.add_beam(beam_add)
                rx_main_phases.append(
                    sequence.rx_beam_phases[slice_id]["main"][
                        beam, sequence.rx_main_antenna_indices[slice_id]
                    ]
                )
                rx_intf_phases.append(
                    sequence.rx_beam_phases[slice_id]["intf"][
                        beam, sequence.rx_intf_antenna_indices[slice_id]
                    ]
                )
            rxchannel.rx_main_phases = np.array(rx_main_phases, dtype=np.complex64)
            rxchannel.rx_intf_phases = np.array(rx_intf_phases, dtype=np.complex64)

            rxchannel.first_range = float(sqn_slice.first_range)
            rxchannel.num_ranges = sqn_slice.num_ranges
            rxchannel.range_sep = sqn_slice.range_sep

            if sqn_slice.acf:
                rxchannel.acf = sqn_slice.acf
                rxchannel.xcf = sqn_slice.xcf
                rxchannel.acfint = sqn_slice.acfint

                for lag in sqn_slice.lag_table:
                    lag_add = messages.LagTable(lag, int(lag[1] - lag[0]))
                    rxchannel.add_ltab(lag_add)
                rxchannel.averaging_method = sqn_slice.averaging_method
            sequence_add.add_rx_channel(rxchannel)
        message.sequences.append(sequence_add)

    return message


def round_up_time(dt=None, round_to=60):
    """
    Round a datetime object to any time lapse in seconds

    :param dt: datetime.datetime object, default now.
    :param round_to: Closest number of seconds to round to, default 1 minute.
    :author: Thierry Husson 2012 - Use it as you want but don't blame me.
    :modified: K.Kotyk 2019

    Will round to the nearest minute mark. Adds one minute if rounded down.
    """

    if dt is None:
        dt = datetime.utcnow()
    midnight = dt.replace(hour=0, minute=0, second=0)
    seconds = (dt.replace(tzinfo=None) - midnight).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    result = dt + timedelta(0, rounding - seconds, -dt.microsecond)

    if result < dt:
        result += timedelta(minutes=1)
    return result


def run_cfs_scan(
    radctrl_process_socket, radctrl_brian_socket, cfs_params, to_driver_inproc_sock
):
    # 2. Send DSP meta data
    # 3. Retrieve DSP results

    dsp_socket_iden = cfs_params.options.dsp_cfs_identity
    router_addr = cfs_params.options.router_address
    cfs_socket = socket_operations.create_sockets(
        [cfs_params.options.radctrl_cfs_identity], router_addr
    )[0]

    aveperiod = cfs_params.aveperiod
    sequence = aveperiod.cfs_sequence
    sqn, dbg = sequence.make_sequence(aveperiod.beam_iter, 0)

    socket_operations.send_bytes(
        cfs_socket,
        cfs_params.options.dw_cfs_identity,
        pickle.dumps(cfs_params.seqnum_start + cfs_params.num_sequences),
    )

    brian_socket_iden = cfs_params.options.brian_to_radctrl_identity
    socket_operations.recv_data(radctrl_brian_socket, brian_socket_iden, log)
    # The above call is blocking; will wait until something is received

    for pulse_transmit_data in sqn:
        driver_message = create_driver_message(cfs_params, pulse_transmit_data)
        to_driver_inproc_sock.send_pyobj(driver_message)

    cfs_params.cfs_scan_flag = True

    dsp_message = create_dsp_message(cfs_params)
    radctrl_process_socket.send_pyobj(dsp_message)
    # Send metadata to dsp_comms_thread, so it can package into a message for rx_signal_processing

    socket_operations.send_bytes(
        radctrl_brian_socket, brian_socket_iden, pickle.dumps(dsp_message)
    )
    # Send the metadata along to brian

    processed_cfs = socket_operations.recv_bytes(cfs_socket, dsp_socket_iden, log)
    # Receive the processed data back from CFS
    freq_data = pickle.loads(processed_cfs)

    radctrl_process_socket.recv_string()
    # Consume the DSP ACK that dsp_comms_thread will send

    return freq_data


def main():
    """
    Run the radar with the experiment supplied by experiment_handler.

    Receives an instance of an experiment. Iterates through the Scans,
    AveragingPeriods, Sequences, and pulses of the experiment.

    For every pulse, samples and other control information are sent to the n200_driver.

    For every pulse sequence, processing information is sent to the signal processing
    block.

    After every averaging period, the experiment block is given the
    opportunity to change the experiment (not currently implemented). If a new
    experiment is sent, radar will halt the old one and begin with the new experiment.
    """

    # Get config options
    options = Options()

    # The socket identities for radar_control, retrieved from options
    ids = [
        options.radctrl_to_exphan_identity,
        options.radctrl_to_brian_identity,
    ]

    # Setup sockets
    # Socket to send pulse samples over
    # TODO test: need to make sure that we know that all sockets are set up after this try...except block.
    # TODO test: starting the programs in different orders.
    try:
        sockets_list = socket_operations.create_sockets(ids, options.router_address)
    except zmq.ZMQBaseError as e:
        log.error("zmq failed setting up sockets", error=e)
        log.exception("zmq failed setting up sockets", exception=e)
        sys.exit(1)

    radar_control_to_exp_handler = sockets_list[0]
    radctrl_brian_socket = sockets_list[1]

    # Sockets for thread communication
    radctrl_inproc_socket = zmq.Context().instance().socket(zmq.PAIR)
    radctrl_inproc_socket.bind("inproc://radctrl")

    driver_comms_socket = zmq.Context().instance().socket(zmq.PAIR)
    driver_comms_socket.bind("inproc://radctrl_driver")

    dw_comms_socket = zmq.Context().instance().socket(zmq.PAIR)
    dw_comms_socket.bind("inproc://radctrl_dw")

    # seqnum is used as an identifier in all packets while radar is running so set it up here.
    # seqnum will get increased by num_sequences (number of averages or sequences in the averaging period)
    # at the end of every averaging period.
    seqnum_start = 0

    # Wait for experiment handler at the start until we have an experiment to run.
    new_experiment_waiting = False

    while not new_experiment_waiting:
        new_experiment_waiting, experiment = search_for_experiment(
            radar_control_to_exp_handler,
            options.exphan_to_radctrl_identity,
            "EXPNEEDED",
        )

    new_experiment_waiting = False
    new_experiment_loaded = True

    # Flag for starting the radar on the minute boundary
    wait_for_first_scanbound = experiment.slice_dict.get("wait_for_first_scanbound")

    # Send driver initial setup data - rates and center frequency from experiment.
    # Wait for acknowledgment that USRP object is set up.

    start_up_params = RadctrlParameters(
        experiment, None, None, options, seqnum_start, True
    )
    driver_start_message = create_driver_message(start_up_params, None)

    metadata_thread = threading.Thread(
        target=dsp_comms_thread,
        args=(
            options.radctrl_to_dsp_identity,
            options.dsp_to_radctrl_identity,
            options.router_address,
        ),
    )

    driver_comms = threading.Thread(
        target=driver_comms_thread,
        args=(
            options.radctrl_to_driver_identity,
            options.driver_to_radctrl_identity,
            options.router_address,
        ),
    )

    dw_comms = threading.Thread(
        target=dw_comms_thread,
        args=(
            options.radctrl_to_dw_identity,
            options.dw_to_radctrl_identity,
            options.router_address,
        ),
    )

    metadata_thread.start()
    driver_comms.start()
    dw_comms.start()

    driver_comms_socket.send_pyobj(driver_start_message)
    driver_comms_socket.recv_string()
    # Wait until driver acknowledges that it is good to start

    first_aveperiod = True
    next_scan_start = None

    while True:
        # This loops through all scans in an experiment, or restarts this loop if a new experiment occurs.
        # TODO : further documentation throughout in comments (high level) and in separate documentation.
        # Iterate through Scans, AveragingPeriods, Sequences, Pulses.
        # Start anew on first scan if we have a new experiment.
        if new_experiment_waiting:
            try:
                experiment = new_experiment
            except NameError as e:
                # new_experiment does not exist, should never happen as flag only gets set when
                # there is a new experiment.
                log.error("experiment could not be found", error=e)
                log.exception("experiment could not be found", exception=e)
                sys.exit(1)

            new_experiment_waiting = False
            new_experiment = None
            new_experiment_loaded = True

        for scan_num, scan in enumerate(experiment.scan_objects):
            log.debug("scan number", scan_num=scan_num)

            # Scan iter is the iterator through the scanbound or through the number of averaging periods in the scan
            if (
                first_aveperiod
                and scan.scanbound is not None
                and not wait_for_first_scanbound
            ):
                # On first integration, determine current averaging period and set scan_iter to it
                now = datetime.utcnow()
                current_minute = now.replace(second=0, microsecond=0)
                scan_iter = next(
                    (
                        i
                        for i, v in enumerate(scan.scanbound)
                        if current_minute + timedelta(seconds=v) > now
                    ),
                    0,
                )
            else:
                # Otherwise start at first averaging period
                scan_iter = 0

            # If a new experiment was received during the last scan, it finished the integration period
            # it was on and returned here with new_experiment_waiting set to True. Break to load new experiment.
            # Start anew on first scan if we have a new experiment
            if new_experiment_waiting:
                break

            if scan.scanbound:
                if scan.align_scan_to_beamorder:
                    for aveperiod in scan.aveperiods:
                        # Always align first beam at start of scan
                        aveperiod.beam_iter = 0

                # Find the start of the next scan with a scanbound so we can determine time remaining for end of scan
                next_scanbound = None
                next_scan_num = scan_num
                while next_scanbound is None:
                    next_scan_num += 1
                    if next_scan_num == len(experiment.scan_objects):
                        next_scan_num = 0
                    next_scanbound = experiment.scan_objects[next_scan_num].scanbound

                if first_aveperiod:
                    # On the very first averaging period of Borealis starting, calculate the start minute
                    # align scanbound reference time to find when to start
                    now = datetime.utcnow()
                    dt = now.replace(second=0, microsecond=0)
                    if dt + timedelta(seconds=scan.scanbound[scan_iter]) >= now:
                        start_minute = dt
                    else:
                        start_minute = round_up_time(now)
                else:
                    # At the start of a scan object that has scanbound, recalculate the start minute to the
                    # previously calculated next_scan_start
                    start_minute = next_scan_start.replace(second=0, microsecond=0)

                # Find the modulus of the number of aveperiod times to run in the scan and the number
                # of AvePeriod classes. The classes will be alternated so we can determine which class
                # will be running at the end of the scan.
                index_of_last_aveperiod_in_scan = (
                    scan.num_aveperiods_in_scan + scan.aveperiod_iter
                ) % len(scan.aveperiods)
                last_aveperiod_intt = scan.aveperiods[
                    index_of_last_aveperiod_in_scan
                ].intt
                # A scanbound necessitates intt
                end_of_scan = (
                    start_minute
                    + timedelta(seconds=scan.scanbound[-1])
                    + timedelta(seconds=last_aveperiod_intt * 1e-3)
                )
                end_minute = end_of_scan.replace(second=0, microsecond=0)

                if end_minute + timedelta(seconds=next_scanbound[0]) >= end_of_scan:
                    next_scan_start = end_minute + timedelta(seconds=next_scanbound[0])
                else:
                    next_scan_start = round_up_time(end_of_scan) + timedelta(
                        seconds=next_scanbound[0]
                    )

            while (
                scan_iter < scan.num_aveperiods_in_scan and not new_experiment_waiting
            ):
                # If there are multiple aveperiods in a scan they are alternated (AVEPERIOD interfaced)
                aveperiod = scan.aveperiods[scan.aveperiod_iter]
                if TIME_PROFILE:
                    time_start_of_aveperiod = datetime.utcnow()

                # Get new experiment here, before starting a new averaging period.
                # If new_experiment_waiting is set here, implement new_experiment after this
                # averaging period. There may be a new experiment waiting, or a new experiment.
                if not new_experiment_waiting and not new_experiment_loaded:
                    new_experiment_waiting, new_experiment = search_for_experiment(
                        radar_control_to_exp_handler,
                        options.exphan_to_radctrl_identity,
                        "NOERROR",
                    )
                elif new_experiment_loaded:
                    new_experiment_loaded = False

                log.debug("new averaging period")

                # All phases are set up for this averaging period for the beams required.
                # Time to start averaging in the below loop.
                if not scan.scanbound:
                    averaging_period_start_time = datetime.utcnow()  # ms
                    log.verbose(
                        "averaging period start time",
                        averaging_period_start_time=averaging_period_start_time,
                        averaging_period_start_time_units="",
                    )
                if aveperiod.intt is not None:
                    intt_break = True

                    if scan.scanbound:
                        # Calculate scan start time. First beam in the sequence will likely
                        # be ready to go if the first scan aligns directly to the minute. The
                        # rest will need to wait until their boundary time is up.
                        beam_scanbound = start_minute + timedelta(
                            seconds=scan.scanbound[scan_iter]
                        )
                        time_diff = beam_scanbound - datetime.utcnow()
                        if time_diff.total_seconds() > 0:
                            if first_aveperiod:
                                log.verbose(
                                    "seconds to next avg period",
                                    time_until_avg_period=time_diff.total_seconds(),
                                    time_until_avg_period_units="s",
                                    scan_iter=scan_iter,
                                    beam_scanbound=beam_scanbound,
                                )
                            else:
                                log.debug(
                                    "seconds to next avg period",
                                    time_until_avg_period=time_diff.total_seconds(),
                                    time_until_avg_period_units="s",
                                    scan_iter=scan_iter,
                                    beam_scanbound=beam_scanbound,
                                )
                            # TODO: reduce sleep if we want to use GPS timestamped transmissions
                            time.sleep(time_diff.total_seconds())
                        else:
                            # TODO: This will be wrong if the start time is in the past.
                            # TODO: maybe use datetime.utcnow() like below instead of beam_scanbound
                            #       when the avg period should have started?
                            log.debug(
                                "expected avg period start time",
                                scan_iter=scan_iter,
                                beam_scanbound=beam_scanbound,
                            )

                        averaging_period_start_time = datetime.utcnow()
                        log.verbose(
                            "avg period start time",
                            avg_period_start_time=averaging_period_start_time,
                            avg_period_start_time_units="s",
                            scan_iter=scan_iter,
                            beam_scanbound=beam_scanbound,
                        )

                        # Here we find how much system time has elapsed to find the true amount
                        # of time we can integrate for this scan boundary. We can then see if
                        # we have enough time left to run the averaging period.
                        time_elapsed = averaging_period_start_time - start_minute
                        if scan_iter < len(scan.scanbound) - 1:
                            scanbound_time = scan.scanbound[scan_iter + 1]
                            # TODO: scanbound_time could be in the past if system has taken
                            #       too long, perhaps calculate which 'beam' (scan_iter) instead by
                            #       rewriting this code for an experiment-wide scanbound attribute instead
                            #       of individual scanbounds inside the scan objects
                            # TODO: if scan_iter skips ahead, aveperiod.beam_iter may also need to
                            #       if scan.align_to_beamorder is True
                            bound_time_remaining = (
                                scanbound_time - time_elapsed.total_seconds()
                            )
                        else:
                            bound_time_remaining = (
                                next_scan_start - averaging_period_start_time
                            )
                            bound_time_remaining = bound_time_remaining.total_seconds()

                        log.verbose(
                            "bound time remaining",
                            bound_time_remaining=bound_time_remaining,
                            bound_time_remaining_units="s",
                            scan_num=scan_num,
                            scan_iter=scan_iter,  # scan_iter is averaging period number for some reason
                            beam_scanbound=beam_scanbound,
                        )

                        if bound_time_remaining < aveperiod.intt * 1e-3:
                            # Reduce the averaging period to only the time remaining until the next scan boundary
                            # TODO: Check for bound_time_remaining > 0
                            #       to be sure there is actually time to run this intt
                            #       (if bound_time_remaining < 0, we need a solution to reset)
                            averaging_period_done_time = (
                                averaging_period_start_time
                                + timedelta(milliseconds=bound_time_remaining * 1e3)
                            )
                        else:
                            averaging_period_done_time = (
                                averaging_period_start_time
                                + timedelta(milliseconds=aveperiod.intt)
                            )
                    else:  # No scanbound for this scan
                        averaging_period_done_time = (
                            averaging_period_start_time
                            + timedelta(milliseconds=aveperiod.intt)
                        )
                else:  # intt does not exist, therefore using intn
                    intt_break = False
                    ending_number_of_sequences = aveperiod.intn  # this will exist

                msg = {
                    x: y[aveperiod.beam_iter]
                    for x, y in aveperiod.slice_to_beamorder.items()
                }
                log.verbose("avg period slice and beam number", slice_and_beam=msg)

                if TIME_PROFILE:
                    aveperiod_prep_time = datetime.utcnow() - time_start_of_aveperiod
                    log.verbose(
                        "time to prep aveperiod",
                        aveperiod_prep_time=aveperiod_prep_time,
                        aveperiod_prep_time_units="",
                    )

                # Time to start averaging in the below loop
                ave_params = RadctrlParameters(
                    experiment,
                    aveperiod,
                    aveperiod.sequences[0],
                    options,
                    seqnum_start,
                    False,
                )
                time_remains = True

                while time_remains:

                    # CFS block
                    if ave_params.num_sequences == 0 and aveperiod.cfs_flag:
                        ave_params.sequence = aveperiod.cfs_sequence
                        ave_params.decimation_scheme = (
                            aveperiod.cfs_sequence.decimation_scheme
                        )
                        freq_spectrum = run_cfs_scan(
                            radctrl_inproc_socket,
                            radctrl_brian_socket,
                            ave_params,
                            driver_comms_socket,
                        )

                        aveperiod.update_cfs_freqs(freq_spectrum)
                        ave_params.aveperiod = aveperiod
                        ave_params.num_sequences += 1
                        ave_params.cfs_scan_flag = False
                    # End CFS block

                    for sequence_index, sequence in enumerate(aveperiod.sequences):
                        # Alternating sequences if there are multiple in the averaging_period
                        ave_params.start_time = datetime.utcnow()
                        ave_params.sequence = sequence
                        ave_params.sequence_index = sequence_index

                        if intt_break:
                            if ave_params.start_time >= averaging_period_done_time:
                                time_remains = False
                                ave_params.averaging_period_time = (
                                    ave_params.start_time - averaging_period_start_time
                                )
                                break
                        else:  # Break at a certain number of sequences
                            if ave_params.num_sequences == ending_number_of_sequences:
                                time_remains = False
                                ave_params.averaging_period_time = (
                                    ave_params.start_time - averaging_period_start_time
                                )
                                break

                        # On first sequence, we make the first set of samples
                        if sequence_index not in ave_params.pulse_transmit_data_tracker:
                            ave_params.pulse_transmit_data_tracker[sequence_index] = {}
                            sqn, dbg = sequence.make_sequence(
                                aveperiod.beam_iter, ave_params.num_sequences
                            )
                            if dbg:
                                ave_params.debug_samples.append(dbg)
                            ave_params.pulse_transmit_data_tracker[sequence_index][
                                ave_params.num_sequences
                            ] = sqn

                        ave_params.decimation_scheme = sequence.decimation_scheme

                        # These three things can happen simultaneously. We can spawn them as threads.
                        threads = [
                            threading.Thread(target=make_next_samples(ave_params))
                        ]

                        for thread in threads:
                            thread.daemon = True
                            thread.start()

                        for (
                            pulse_transmit_data
                        ) in ave_params.pulse_transmit_data_tracker[sequence_index][
                            ave_params.num_sequences
                        ]:
                            driver_message = create_driver_message(
                                ave_params, pulse_transmit_data
                            )
                            driver_comms_socket.send_pyobj(driver_message)

                        dsp_message = create_dsp_message(ave_params)
                        radctrl_inproc_socket.send_pyobj(dsp_message)
                        # Send metadata to dsp_comms_thread, so it can pass it to rx_signal_processing

                        socket_operations.send_bytes(
                            radctrl_brian_socket,
                            options.brian_to_radctrl_identity,
                            pickle.dumps(dsp_message),
                        )
                        # Send the metadata along to brian

                        for thread in threads:
                            thread.join()

                        radctrl_inproc_socket.recv_string()
                        # wait for dsp to signal that it received the metadata

                        ave_params.num_sequences += 1

                        if first_aveperiod:
                            first_aveperiod = False

                        # Sequence is done
                        if __debug__:
                            time.sleep(1)

                if TIME_PROFILE:
                    avg_period_end_time = datetime.utcnow()
                    log.verbose(
                        "avg period end time",
                        avg_period_end_time=avg_period_end_time,
                        avg_period_end_time_units="s",
                    )

                log.info(
                    "aveperiod done",
                    num_sequences=ave_params.num_sequences,
                    slice_ids=aveperiod.slice_ids,
                )

                if scan.aveperiod_iter == 0 and aveperiod.beam_iter == 0:
                    # This is the first averaging period in the scan object.
                    # if scanbound is aligned to beamorder, the scan_iter will also = 0 at this point.
                    ave_params.scan_flag = True
                else:
                    ave_params.scan_flag = False

                ave_params.last_sequence_num = (
                    ave_params.seqnum_start + ave_params.num_sequences - 1
                )

                dw_message = create_dw_message(ave_params)
                dw_comms_socket.send_pyobj(dw_message)
                # Send metadata to dw_comms_thread, so it can package into a message for data write

                # end of the averaging period loop - move onto the next averaging period.
                # Increment the sequence number by the number of sequences that were in this
                # averaging period.
                seqnum_start += ave_params.num_sequences

                if TIME_PROFILE:
                    time_to_finish_aveperiod = datetime.utcnow() - avg_period_end_time
                    log.verbose(
                        "time to finish avg period",
                        avg_period_elapsed_time=time_to_finish_aveperiod,
                        avg_period_elapsed_time_units="s",
                    )

                aveperiod.beam_iter += 1
                if aveperiod.beam_iter == aveperiod.num_beams_in_scan:
                    aveperiod.beam_iter = 0
                scan_iter += 1
                scan.aveperiod_iter += 1
                if scan.aveperiod_iter == len(scan.aveperiods):
                    scan.aveperiod_iter = 0


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log()
    log.info(f"RADAR_CONTROL BOOTED")
    try:
        main()
        log.info(f"RADAR_CONTROL EXITED")
    except Exception as main_exception:
        log.critical("RADAR_CONTROL CRASHED", error=main_exception)
        log.exception("RADAR_CONTROL CRASHED", exception=main_exception)
