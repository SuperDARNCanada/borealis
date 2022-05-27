"""
Copyright SuperDARN Canada 2022
Authors: Keith Kotyk (rx_signal_processing.py)
         Remington Rohel (frankensteined rx_signal_processing.py into this file)
"""
import sys
import os
import time
import threading
import numpy as np
import posix_ipc as ipc
import math
import copy
from scipy.constants import speed_of_light
import h5py

try:
    import cupy as cp
except:
    cupy_available = False
else:
    cupy_available = True

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

sys.path.append(borealis_path + '/utils/')
import signal_processing_options.signal_processing_options as spo
import shared_macros.shared_macros as sm

from rx_signal_processing import dsp
from experiment_prototype.decimation_scheme.decimation_scheme import DecimationScheme

pprint = sm.MODULE_PRINT("rx signal processing", "magenta")


def save_to_file(slice_details, data_outputs):
    """
    Write the data outputs to file.

    :param      slice_details:   The details for each slice that was processed.
    :type       slice_details:   list
    :param      data_outputs:    The processed data outputs.
    :type       data_outputs:    dict
    """

    output_filename = "/home/radar/develop_dsp_testing.hdf5"
    with h5py.File(output_filename, 'w') as f:
        for sd in slice_details:
            slice_group = f.create_group('slice_{}'.format(sd['slice_id']))

            slice_group.attrs['num_beams'] = sd['num_beams']

            beam_samps = data_outputs['beamformed_m'][sd['slice_num']][:sd['num_beams']]
            try:
                intf_samps = data_outputs['beamformed_i'][sd['slice_num']][:sd['num_beams']]
                beam_samps = np.concatenate((beam_samps, intf_samps), axis=1)   # along antennas axis
            except:
                # No interferometer data
                pass
            slice_group.create_dataset('beamformed_samples', data=beam_samps)
            slice_group.create_dataset('beamformed_dims', data=['beams', 'antennas', 'samples'])

            def add_debug_data(stage, name):
                all_ant_samps = stage[sd['slice_num']]
                slice_group.create_dataset('{}_samples'.format(name), data=all_ant_samps)

            for i, stage in enumerate(data_outputs['debug_outputs'][:-1]):
                add_debug_data(stage, "stage_" + str(i))

            stage = data_outputs['debug_outputs'][-1]
            add_debug_data(stage, "antennas")


def main():
    sig_options = spo.SignalProcessingOptions()

    ringbuffer = None

    total_antennas = sig_options.main_antenna_count + sig_options.intf_antenna_count

    # TODO(Remington): Use default DecimationScheme here
    dm_rates = [10 ,5, 6, 5]
    dm_scheme_taps = []

    extra_samples = 0
    total_dm_rate = np.prod(dm_rates)

    threads = []

    rx_rate = np.float64(5e6)
    output_sample_rate = np.float64(rx_rate / total_dm_rate)
    first_rx_sample_off = 0

    decimation_stages = DecimationScheme(rx_rate, output_sample_rate).stages()

    mixing_freqs = [-1.25e6, 1.25e6]
    main_beam_angles = []
    intf_beam_angles = []

    # Parse out details and force the data type so that Cupy can optimize with standardized
    # data types.
    slice_details = []
    for i in range(len(mixing_freqs)):
        detail = {}

        detail['slice_id'] = i
        detail['slice_num'] = i
        detail['first_range'] = np.float32(180)     # km
        detail['range_sep'] = np.float32(300 * 1.0e-9 * speed_of_light / 2.0)   # km
        detail['tau_spacing'] = np.uint32(2400)     # us
        detail['num_range_gates'] = np.uint32(75)
        detail['first_range_off'] = np.uint32(detail['first_range'] / detail['range_sep'])

        main_beams = []
        intf_beams = []
        for _ in [0]:
            main_beam = []
            intf_beam = []

            for j, phase in enumerate(np.ones(20)):
                p = phase       # boresight, for simplicity

                if j < sig_options.main_antenna_count:
                    main_beam.append(p)
                else:
                    intf_beam.append(p)

            main_beams.append(main_beam)
            intf_beams.append(intf_beam)

        detail['num_beams'] = len(main_beams)

        slice_details.append(detail)
        main_beam_angles.append(main_beams)
        intf_beam_angles.append(intf_beams)

    # Different slices can have a different amount of beams used. Slices that use fewer beams
    # than the max number of beams are padded with zeros so that matrix calculations can be
    # used. The extra beams that are processed will be not be parsed for data writing.
    max_num_beams = max([len(x) for x in main_beam_angles])

    def pad_beams(angles, ant_count):
        for x in angles:
            if len(x) < max_num_beams:
                beam_pad = [0.0j] * ant_count
                for i in range(max_num_beams - len(x)):
                    x.append(beam_pad)

    pad_beams(main_beam_angles, sig_options.main_antenna_count)
    pad_beams(intf_beam_angles, sig_options.intf_antenna_count)

    main_beam_angles = np.array(main_beam_angles, dtype=np.complex64)
    intf_beam_angles = np.array(intf_beam_angles, dtype=np.complex64)
    mixing_freqs = np.array(mixing_freqs, dtype=np.float64)

    # TODO(Remington): Initialize some data!
    ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(total_antennas, -1)

    if cupy_available:
        cp.cuda.runtime.hostRegister(ringbuffer.ctypes.data, ringbuffer.size, 0)

    dm_msg = "Decimation rates: "
    taps_msg = "Number of filter taps per stage: "
    for stage in decimation_stages:
        dm_rates.append(stage.dm_rate)
        dm_scheme_taps.append(np.array(stage.filter_taps, dtype=np.complex64))

        dm_msg += str(stage.dm_rate) + " "
        taps_msg += str(len(stage.filter_taps)) + " "

    dm_rates = np.array(dm_rates, dtype=np.uint32)
    pprint(dm_msg)
    pprint(taps_msg)

    for dm, taps in zip(reversed(dm_rates), reversed(dm_scheme_taps)):
        extra_samples = (extra_samples * dm) + len(taps) / 2

    # This work is done in a thread
    def sequence_worker(**kwargs):
        sequence_num = kwargs['sequence_num']
        main_beam_angles = kwargs['main_beam_angles']
        intf_beam_angles = kwargs['intf_beam_angles']
        mixing_freqs = kwargs['mixing_freqs']
        slice_details = kwargs['slice_details']
        start_sample = kwargs['start_sample']
        end_sample = kwargs['end_sample']

        pprint(sm.COLOR('green', "Processing #{}".format(sequence_num)))
        pprint("Mixing freqs for #{}: {}".format(sequence_num, mixing_freqs))
        pprint("Main beams shape for #{}: {}".format(sequence_num, main_beam_angles.shape))
        pprint("Intf beams shape for #{}: {}".format(sequence_num, intf_beam_angles.shape))
        if cupy_available:
            cp.cuda.runtime.setDevice(0)

        start = time.time()

        if cupy_available:
            sequence_samples = cp.array(ringbuffer)
        else:
            sequence_samples = ringbuffer

        copy_end = time.time()
        time_diff = (copy_end - start) * 1000
        pprint("Time to copy samples for #{}: {}ms".format(sequence_num, time_diff))

        # Process main samples
        main_sequence_samples = sequence_samples[:sig_options.main_antenna_count, :]
        pprint("Main buffer shape: {}".format(main_sequence_samples.shape))
        processed_main_samples = dsp.DSP(main_sequence_samples, rx_rate, dm_rates,
                                         dm_scheme_taps, mixing_freqs, main_beam_angles)

        # If interferometer is used, process those samples too.
        if sig_options.intf_antenna_count > 0:
            intf_sequence_samples = sequence_samples[sig_options.main_antenna_count:, :]
            pprint("Intf buffer shape: {}".format(intf_sequence_samples.shape))
            processed_intf_samples = dsp.DSP(intf_sequence_samples, rx_rate, dm_rates,
                                             dm_scheme_taps, mixing_freqs, intf_beam_angles)

        end = time.time()

        time_diff = (end - copy_end) * 1000

        pprint("Time to decimate, beamform and correlate for #{}: {}ms".format(sequence_num,
                                                                               time_diff))

        time_diff = (end - start) * 1000
        pprint("Total time for #{}: {}ms".format(sequence_num, time_diff))

        # Extract outputs from processing into groups that will be put into proto fields.
        start = time.time()
        data_outputs = {}
        if cupy_available:
            filter_outputs_m = [cp.asnumpy(x) for x in processed_main_samples.filter_outputs]

            if sig_options.intf_antenna_count > 0:
                filter_outputs_i = [cp.asnumpy(x) for x in processed_intf_samples.filter_outputs]
        else:
            filter_outputs_m = processed_main_samples.filter_outputs

            if sig_options.intf_antenna_count > 0:
                filter_outputs_i = processed_intf_samples.filter_outputs

        if sig_options.intf_antenna_count > 0:
            filter_outputs = [np.hstack((x, y)) for x, y in zip(filter_outputs_m,
                                                                filter_outputs_i)]
        else:
            filter_outputs = filter_outputs_m

        data_outputs['debug_outputs'] = filter_outputs

        if cupy_available:
            beamformed_m = cp.asnumpy(processed_main_samples.beamformed_samples)

            if sig_options.intf_antenna_count > 0:
                beamformed_i = cp.asnumpy(processed_intf_samples.beamformed_samples)
        else:
            beamformed_m = processed_main_samples.beamformed_samples

            if sig_options.intf_antenna_count > 0:
                beamformed_i = processed_intf_samples.beamformed_samples

        data_outputs['beamformed_m'] = beamformed_m

        if sig_options.intf_antenna_count > 0:
            data_outputs['beamformed_i'] = beamformed_i

        save_to_file(slice_details, data_outputs)

        end = time.time()
        time_diff = (end - start) * 1000
        pprint("Time to serialize and send processed data for #{}: {}ms".format(sequence_num,
                                                                                time_diff))

    args = {"sequence_num": 0,
            "main_beam_angles": copy.deepcopy(main_beam_angles),
            "intf_beam_angles": copy.deepcopy(intf_beam_angles),
            "mixing_freqs": copy.deepcopy(mixing_freqs),
            "slice_details": copy.deepcopy(slice_details),
            "start_sample": 0,
            "end_sample": ringbuffer.shape[-1],     # [antennas, samps]
            }

    seq_thread = threading.Thread(target=sequence_worker, kwargs=args)
    seq_thread.daemon = True
    seq_thread.start()

    threads.append(seq_thread)
    seq_thread.join()
    if len(threads) > 1:
        thread = threads.pop(0)
        thread.join()


if __name__ == "__main__":
    main()














