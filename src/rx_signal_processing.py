"""
    rx_signal_processing
    ~~~~~~~~~~~~~~~~~~~~~
    This process handles the digital signal processing side of Borealis

    :copyright: 2020 SuperDARN Canada
    :author: Keith Kotyk
"""

import copy
import math
import mmap
from multiprocessing import shared_memory
import os
import pickle
import sys
import threading
import time

import numpy as np
import posix_ipc as ipc

try:
    import cupy as xp
except ImportError:
    cupy_available = False
    import numpy as xp
else:
    cupy_available = True


sys.path.append(os.environ["BOREALISPATH"])

if __debug__:
    from build.debug.src.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata
else:
    from build.release.src.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata

from utils.message_formats import (
    ProcessedSequenceMessage,
    DebugDataStage,
    OutputDataset,
)
from utils.signals import DSP


def fill_datawrite_message(processed_data, slice_details, data_outputs):
    """
    Fills the datawrite message with processed data.

    :param      processed_data:  The processed data message
    :type       processed_data:  ProcessedSequenceMessage
    :param      slice_details:   The details for each slice that was processed.
    :type       slice_details:   list
    :param      data_outputs:    The processed data outputs.
    :type       data_outputs:    dict
    """

    for sd in slice_details:
        output_dataset = OutputDataset(
            sd["slice_id"], sd["num_beams"], sd["num_range_gates"], sd["num_lags"]
        )

        def add_array(ndarray):
            """
            Creates shared memory and stores ndarray in it.

            :param      ndarray: array to be created
            :type       ndarray: numpy.ndarray
            :returns:   The shared memory name.
            :rtype:     str
            """
            if ndarray.size != 0:
                shm = shared_memory.SharedMemory(create=True, size=ndarray.nbytes)
                shared_array = np.ndarray(
                    ndarray.shape, dtype=ndarray.dtype, buffer=shm.buf
                )
                shared_array[...] = ndarray[...]
                name = shm.name
                # This closes the current SharedMemory instance, but the memory isn't free until data_write unlinks it.
                shm.close()
                return name

        main_corrs = data_outputs["main_corrs"][sd["slice_num"]]
        output_dataset.main_acf_shm = add_array(main_corrs)

        intf_available = True
        try:
            intf_corrs = data_outputs["intf_corrs"][sd["slice_num"]]
            cross_corrs = data_outputs["cross_corrs"][sd["slice_num"]]
        except KeyError as e:
            # No interferometer data
            intf_available = False

        if intf_available:
            output_dataset.intf_acf_shm = add_array(intf_corrs)
            output_dataset.xcf_shm = add_array(cross_corrs)

        processed_data.add_output_dataset(output_dataset)


def main():
    options = Options()

    sockets = so.create_sockets(
        [options.dsp_to_radctrl_identity, options.dsp_to_driver_identity],
        options.router_address,
    )

    dsp_to_radar_control = sockets[0]
    dsp_to_driver = sockets[1]

    ringbuffer = None

    total_antennas = len(options.rx_main_antennas) + len(options.rx_intf_antennas)

    threads = []
    first_time = True
    while True:

        reply = so.recv_bytes(
            dsp_to_radar_control, options.radctrl_to_dsp_identity, log
        )

        sqn_meta_message = pickle.loads(reply)

        rx_rate = np.float64(sqn_meta_message.rx_rate)
        output_sample_rate = np.float64(sqn_meta_message.output_sample_rate)
        first_rx_sample_off = sqn_meta_message.offset_to_first_rx_sample
        rx_center_freq = sqn_meta_message.rx_ctr_freq

        processed_data = ProcessedSequenceMessage()

        processed_data.sequence_num = sqn_meta_message.sequence_num
        processed_data.rx_sample_rate = rx_rate
        processed_data.output_sample_rate = output_sample_rate

        mixing_freqs = []
        main_beam_angles = []
        intf_beam_angles = []
        intf_antennas = set()

        # Parse out details and force the data type so that Cupy can optimize with standardized
        # data types.
        slice_details = []
        for i, chan in enumerate(sqn_meta_message.rx_channels):
            detail = {}

            mixing_freqs.append(chan.rx_freq - rx_center_freq)

            detail["slice_id"] = chan.slice_id
            detail["slice_num"] = i
            detail["first_range"] = np.float32(chan.first_range)
            detail["range_sep"] = np.float32(chan.range_sep)
            detail["tau_spacing"] = np.uint32(chan.tau_spacing)
            detail["num_range_gates"] = np.uint32(chan.num_ranges)
            detail["first_range_off"] = np.uint32(chan.first_range / chan.range_sep)

            lag_phase_offsets = []

            lags = []
            for lag in chan.lags:
                lags.append([lag.pulse_1, lag.pulse_2])
                lag_phase_offsets.append(
                    lag.phase_offset_real + 1j * lag.phase_offset_imag
                )

            detail["lag_phase_offsets"] = np.array(
                lag_phase_offsets, dtype=np.complex64
            )

            detail["lags"] = np.array(lags, dtype=np.uint32)
            detail["num_lags"] = len(lags)

            main_beams = chan.beam_phases[:, : len(options.rx_main_antennas)]
            intf_beams = chan.beam_phases[:, len(options.rx_main_antennas) :]

            detail["num_beams"] = main_beams.shape[0]

            slice_details.append(detail)
            main_beam_angles.append(main_beams)
            intf_beam_angles.append(intf_beams)

            intf_antennas.update(
                set(chan.rx_intf_antennas)
            )  # Keep track of all intf antennas for the sequence

        # Different slices can have a different amount of beams used. Slices that use fewer beams
        # than the max number of beams are padded with zeros so that matrix calculations can be
        # used. The extra beams that are processed will be not be parsed for data writing.
        max_num_beams = max([x.shape[0] for x in main_beam_angles])
        padded_main_phases = np.zeros(
            (
                len(sqn_meta_message.rx_channels),
                max_num_beams,
                len(options.rx_main_antennas),
            ),
            dtype=np.complex64,
        )
        padded_intf_phases = np.zeros(
            (
                len(sqn_meta_message.rx_channels),
                max_num_beams,
                len(options.rx_intf_antennas),
            ),
            dtype=np.complex64,
        )

        for i, x in enumerate(main_beam_angles):
            padded_main_phases[i, : len(x)] = x
        for i, x in enumerate(intf_beam_angles):
            padded_intf_phases[i, : len(x)] = x

        mixing_freqs = np.array(mixing_freqs, dtype=np.float64)

        # Get meta from driver
        message = "Need data to process"
        so.send_data(dsp_to_driver, options.driver_to_dsp_identity, message)
        reply = so.recv_bytes(dsp_to_driver, options.driver_to_dsp_identity, log)

        rx_metadata = RxSamplesMetadata()
        rx_metadata.ParseFromString(reply)

        if sqn_meta_message.sequence_num != rx_metadata.sequence_num:
            log.error(
                "driver packets != radctrl packets",
                sqn_meta_sqn_num=sqn_meta_message.sequence_num,
                rx_meta_sqn_num=rx_metadata.sequence_num,
            )
            sys.exit(-1)

        # First time configuration
        if first_time:
            shm = ipc.SharedMemory(options.ringbuffer_name)
            mapped_mem = mmap.mmap(shm.fd, shm.size)
            ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(
                total_antennas, -1
            )

            if cupy_available:
                xp.cuda.runtime.hostRegister(ringbuffer.ctypes.data, ringbuffer.size, 0)

            first_time = False

        # Set up the filtering/downsampling strategy
        taps_per_stage = []
        dm_rates = []
        dm_scheme_taps = []
        extra_samples = 0
        for stage in sqn_meta_message.decimation_stages:
            dm_rates.append(stage.dm_rate)
            dm_scheme_taps.append(np.array(stage.filter_taps, dtype=np.complex64))
            taps_per_stage.append(len(stage.filter_taps))
        log.verbose(
            "stage decimation and filter taps",
            decimation_rates=dm_rates,
            filter_taps_per_stage=taps_per_stage,
        )
        dm_rates = np.array(dm_rates, dtype=np.uint32)
        for dm, taps in zip(reversed(dm_rates), reversed(dm_scheme_taps)):
            extra_samples = (extra_samples * dm) + len(taps) // 2
        total_dm_rate = np.prod(dm_rates)

        # Calculate where in the ringbuffer the samples are located.
        samples_needed = rx_metadata.numberofreceivesamples + 2 * extra_samples
        samples_needed = int(
            math.ceil(float(samples_needed) / float(total_dm_rate)) * total_dm_rate
        )

        sample_time_diff = (
            rx_metadata.sequence_start_time - rx_metadata.initialization_time
        )
        sample_in_time = (
            (sample_time_diff * rx_rate) + first_rx_sample_off - extra_samples
        )
        start_sample = int(math.fmod(sample_in_time, ringbuffer.shape[1]))
        end_sample = start_sample + samples_needed

        processed_data.initialization_time = rx_metadata.initialization_time
        processed_data.sequence_start_time = rx_metadata.sequence_start_time
        processed_data.gps_to_system_time_diff = rx_metadata.gps_to_system_time_diff
        processed_data.agc_status_bank_h = rx_metadata.agc_status_bank_h
        processed_data.lp_status_bank_h = rx_metadata.lp_status_bank_h
        processed_data.agc_status_bank_l = rx_metadata.agc_status_bank_l
        processed_data.lp_status_bank_l = rx_metadata.lp_status_bank_l
        processed_data.gps_locked = rx_metadata.gps_locked

        # This work is done in a thread
        def sequence_worker(**kwargs):
            sequence_num = kwargs["sequence_num"]
            main_beam_angles = kwargs["main_beam_angles"]
            intf_beam_angles = kwargs["intf_beam_angles"]
            mixing_freqs = kwargs["mixing_freqs"]
            slice_details = kwargs["slice_details"]
            start_sample = kwargs["start_sample"]
            end_sample = kwargs["end_sample"]
            processed_data = kwargs["processed_data"]
            intf_antennas = kwargs["intf_antennas"]
            filter_taps = kwargs["filter_taps"]
            downsample_rates = kwargs["downsample_rates"]

            if cupy_available:
                xp.cuda.runtime.setDevice(0)

            seq_begin_iden = options.dspbegin_to_brian_identity + str(sequence_num)
            seq_end_iden = options.dspend_to_brian_identity + str(sequence_num)
            dw_iden = options.dsp_to_dw_identity + str(sequence_num)
            gpu_socks = so.create_sockets(
                [seq_begin_iden, seq_end_iden, dw_iden], options.router_address
            )

            dspbegin_to_brian = gpu_socks[0]
            dspend_to_brian = gpu_socks[1]
            dsp_to_dw = gpu_socks[2]

            # Generate a timer dict for a uniform log
            log_dict = {"time_units": "ms"}
            start_timer = time.perf_counter()

            # Copy samples from ring buffer
            indices = np.arange(start_sample, start_sample + samples_needed)
            # x.take makes a copy of the array. We want to avoid making a copy using Cupy so that
            # data is moved directly from the ring buffer to the GPU. Simple indexing creates a view
            # of existing data without making a copy.
            if cupy_available:
                if end_sample > ringbuffer.shape[1]:
                    piece1 = ringbuffer[:, start_sample:]
                    piece2 = ringbuffer[:, : end_sample - ringbuffer.shape[1]]
                    tmp1 = xp.array(piece1)
                    tmp2 = xp.array(piece2)
                    sequence_samples = xp.concatenate((tmp1, tmp2), axis=1)
                else:
                    sequence_samples = xp.array(ringbuffer[:, start_sample:end_sample])
            else:
                sequence_samples = ringbuffer.take(indices, axis=1, mode="wrap")
            log_dict["copy_samples_from_ringbuffer_time"] = (
                time.perf_counter() - start_timer
            ) * 1e3

            # Tell brian DSP is about to begin
            mark_timer = time.perf_counter()
            reply_packet = {"sequence_num": sequence_num}
            msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)
            so.recv_bytes(dspbegin_to_brian, options.brian_to_dspbegin_identity, log)
            so.send_bytes(dspbegin_to_brian, options.brian_to_dspbegin_identity, msg)
            log_dict["dsp_begin_msg_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Process main samples
            mark_timer = time.perf_counter()
            main_sequence_samples = sequence_samples[: len(options.rx_main_antennas), :]
            main_sequence_samples_shape = main_sequence_samples.shape
            main_processor = DSP(rx_rate, filter_taps, mixing_freqs, downsample_rates)
            main_processor.apply_filters(main_sequence_samples)
            main_processor.move_filter_results()
            main_processor.beamform(main_beam_angles)
            main_corrs = DSP.correlations_from_samples(
                main_processor.beamformed_samples,
                main_processor.beamformed_samples,
                output_sample_rate,
                slice_details,
            )
            log_dict["main_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Process intf samples if intf exists
            mark_timer = time.perf_counter()
            intf_sequence_samples_shape = None
            log_dict["intf antennas"] = intf_antennas
            if len(intf_antennas) > 0:
                intf_sequence_samples = sequence_samples[
                    len(options.rx_main_antennas) :, :
                ]
                intf_sequence_samples_shape = intf_sequence_samples.shape
                intf_processor = DSP(
                    rx_rate, filter_taps, mixing_freqs, downsample_rates
                )
                intf_processor.apply_filters(intf_sequence_samples)
                intf_processor.move_filter_results()
                intf_processor.beamform(intf_beam_angles)
                intf_corrs = DSP.correlations_from_samples(
                    intf_processor.beamformed_samples,
                    intf_processor.beamformed_samples,
                    output_sample_rate,
                    slice_details,
                )
                cross_corrs = DSP.correlations_from_samples(
                    intf_processor.beamformed_samples,
                    main_processor.beamformed_samples,
                    output_sample_rate,
                    slice_details,
                )
            log_dict["intf_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Tell brian DSP how long it took
            mark_timer = time.perf_counter()
            reply_packet["kerneltime"] = (
                log_dict["main_dsp_time"] + log_dict["intf_dsp_time"]
            )
            msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)
            so.recv_bytes(dspend_to_brian, options.brian_to_dspend_identity, log)
            so.send_bytes(dspend_to_brian, options.brian_to_dspend_identity, msg)
            log_dict["dsp_end_msg_time"] = (time.perf_counter() - mark_timer) * 1e3

            total_processing_time = (time.perf_counter() - start_timer) * 1e3
            log_dict["total_sequence_process_time"] = total_processing_time
            log.verbose(
                "processing sequence",
                sequence_num=sequence_num,
                mixing_freqs=mixing_freqs,
                mixing_freqs_units="Hz",
                main_beam_angles=main_beam_angles.shape,
                intf_beam_angles=main_beam_angles.shape,
                main_buffer_shape=main_sequence_samples_shape,
                intf_buffer_shape=intf_sequence_samples_shape,
                **log_dict,
            )

            # Generate a new timer dict for a uniform log
            log_dict = {"time_units": "ms"}
            start_timer = time.perf_counter()

            # Extract outputs from processing into groups that will be put into message fields.
            data_outputs = {}

            def debug_data_in_shm(holder, data_array, array_name):
                """
                Adds an array of antennas data (filter outputs or antennas_iq) into a dictionary
                for later entry in a processed data message.

                :param  holder:     Dictionary to store the shared memory parameters.
                :type   holder:     dict
                :param  data_array: array to hold the data
                :type   data_array: cp.ndarray or np.ndarray
                :param  array_name: 'main' or 'intf'
                :type   array_name: str
                """

                shm = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
                data = np.ndarray(data_array.shape, dtype=np.complex64, buffer=shm.buf)
                if cupy_available:
                    data[...] = xp.asnumpy(data_array)
                else:
                    data[...] = data_array

                if array_name == "main":
                    holder.main_shm = shm.name
                elif array_name == "intf":
                    holder.intf_shm = shm.name
                else:
                    log.error(f"unknown array name {array_name} not in [main, intf]")
                    log.exception(f"unknown array name {array_name} [main, intf]")
                    sys.exit(1)

                holder.num_samps = data_array.shape[-1]
                shm.close()

            # Add the filter stage data if in debug mode
            if __debug__:
                for i, main_data in enumerate(main_processor.filter_outputs[:-1]):
                    stage = DebugDataStage(f"stage_{i}")
                    debug_data_in_shm(stage, main_data, "main")

                    if options.intf_antenna_count > 0:
                        intf_data = intf_processor.filter_outputs[i]
                        debug_data_in_shm(stage, intf_data, "intf")

                    processed_data.add_debug_data(stage)

            # Add antennas_iq data
            stage = DebugDataStage()
            stage.stage_name = "antennas"
            main_shm = main_processor.shared_mem["antennas_iq"]
            stage.main_shm = main_shm.name
            stage.num_samps = main_processor.antennas_iq_samples.shape[-1]
            main_shm.close()
            if len(intf_antennas) > 0:
                intf_shm = intf_processor.shared_mem["antennas_iq"]
                stage.intf_shm = intf_shm.name
                intf_shm.close()
            processed_data.add_debug_data(stage)
            log_dict["add_antiq_to_stage_time"] = (
                time.perf_counter() - start_timer
            ) * 1e3
            mark_timer = time.perf_counter()

            # Add rawrf data
            if __debug__:
                # np.complex64 in bytes * num_antennas * num_samps
                rawrf_size = (
                    np.dtype(np.complex64).itemsize
                    * ringbuffer.shape[0]
                    * indices.shape[-1]
                )
                rawrf_shm = shared_memory.SharedMemory(create=True, size=rawrf_size)
                rawrf_array = np.ndarray(
                    (ringbuffer.shape[0], indices.shape[-1]),
                    dtype=np.complex64,
                    buffer=rawrf_shm.buf,
                )
                rawrf_array[...] = ringbuffer.take(indices, axis=1, mode="wrap")
                processed_data.rawrf_shm = rawrf_shm.name
                processed_data.rawrf_num_samps = indices.shape[-1]
                rawrf_shm.close()
                log_dict["put_rawrf_in_shm_time"] = (
                    time.perf_counter() - mark_timer
                ) * 1e3

            # Add bfiq and correlations data
            mark_timer = time.perf_counter()
            # beamformed_m: [num_slices, num_beams, num_samps]
            beamformed_m = main_processor.beamformed_samples
            processed_data.bfiq_main_shm = main_processor.shared_mem["bfiq"].name
            processed_data.max_num_beams = beamformed_m.shape[1]
            processed_data.num_samps = beamformed_m.shape[-1]
            main_processor.shared_mem["bfiq"].close()

            data_outputs["main_corrs"] = main_corrs

            if len(intf_antennas) > 0:
                data_outputs["cross_corrs"] = cross_corrs
                data_outputs["intf_corrs"] = intf_corrs
                processed_data.bfiq_intf_shm = intf_processor.shared_mem["bfiq"].name
                intf_processor.shared_mem["bfiq"].close()

            # Fill message with the slice-specific fields
            fill_datawrite_message(processed_data, slice_details, data_outputs)
            sqn_message = pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL)
            log_dict["add_bfiq_and_acfs_to_stage_time"] = (
                time.perf_counter() - mark_timer
            ) * 1e3

            so.send_bytes(dsp_to_dw, options.dw_to_dsp_identity, sqn_message)

            log_dict["total_serialize_send_time"] = (
                time.perf_counter() - start_timer
            ) * 1e3
            log.info(
                "done with sequence",
                sequence_num=sequence_num,
                processing_time=total_processing_time,
                time_units="ms",
                slice_ids=[d["slice_id"] for d in slice_details],
            )
            log.verbose("sequence timing", sequence_num=sequence_num, **log_dict)

        args = {
            "sequence_num": copy.deepcopy(sqn_meta_message.sequence_num),
            "main_beam_angles": copy.deepcopy(padded_main_phases),
            "intf_beam_angles": copy.deepcopy(padded_intf_phases),
            "mixing_freqs": copy.deepcopy(mixing_freqs),
            "slice_details": copy.deepcopy(slice_details),
            "start_sample": copy.deepcopy(start_sample),
            "end_sample": copy.deepcopy(end_sample),
            "processed_data": copy.deepcopy(processed_data),
            "intf_antennas": copy.deepcopy(intf_antennas),
            "filter_taps": copy.deepcopy(dm_scheme_taps),
            "downsample_rates": copy.deepcopy(dm_rates),
        }

        seq_thread = threading.Thread(target=sequence_worker, kwargs=args)
        seq_thread.daemon = True
        seq_thread.start()

        threads.append(seq_thread)

        if len(threads) > 1:
            thread = threads.pop(0)
            thread.join()


if __name__ == "__main__":
    from utils.options import Options
    from utils import socket_operations as so
    from utils import log_config

    log = log_config.log()
    log.info(f"RX_SIGNAL_PROCESSING BOOTED")
    if not cupy_available:
        log.warning("cupy not installed")
    try:
        main()
        log.info(f"RX_SIGNAL_PROCESSING EXITED")
    except Exception as main_exception:
        log.critical("RX_SIGNAL_PROCESSING CRASHED", error=main_exception)
        log.exception("RX_SIGNAL_PROCESSING CRASHED", exception=main_exception)
