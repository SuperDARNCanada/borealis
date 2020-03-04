#!/usr/bin/python

"""
    sequences
    ~~~~~~~~~
    This is the module containing the Sequence class. The Sequence class contains the
    ScanClassBase members, as well as a list of pulse dictionaries,
    the total_combined_pulses in the sequence, power_divider, last_pulse_len, ssdelay,
    seqtime, which together give sstime (scope synce time, or time for receiving,
    and numberofreceivesamples to sample during the receiving window (calculated using
    the receive sampling rate).

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

import math
import numpy as np
from scipy.constants import speed_of_light
import copy
from operator import itemgetter
import collections

from sample_building.sample_building import get_samples
from experiment_prototype.scan_classes.scan_class_base import ScanClassBase
from experiment_prototype.experiment_exception import ExperimentException
class Sequence(ScanClassBase):
    """
    Set up the sequence class.

    **The members of the sequence are:**

    pulses
        a list of pre-combined, pre-sampled pulse dictionaries (one dictionary = one
        basic pulse of single frequency). The dictionary keys are: isarepeat,
        pulse_timing_us, slice_id, slice_pulse_index, pulse_len,
        intra_pulse_start_time, combined_pulse_index, pulse_shift, iscombined,
        combine_total, and combine_index.
    total_combined_pulses
        the total number of pulses to be sent by the driver. This may not
        be the sum of pulses in all slices in the sequence, as some pulses may need to be
        combined because they are overlapping in timing. This is the number of pulses in the
        combined sequence, or the number of times T/R signal goes high in the sequence.
    power_divider
        the power ratio per slice. If there are multiple slices in the same
        pulse then we must reduce the output amplitude to potentially accommodate multiple
        frequencies.
    last_pulse_len
        the length of the last pulse (us)
    ssdelay
        delay past the end of the sequence to receive for (us) - function of num_ranges and
        pulse_len. ss stands for scope sync.
    seqtime
        the amount of time for the whole sequence to transmit, until the logic signal
        switches low on the last pulse in the sequence (us).
    sstime
        ssdelay + seqtime (total time for receiving) (us).
    numberofreceivesamples
        the number of receive samples to take, given the rx rate, during
        the sstime.
    first_rx_sample_time
        The location of the first sample for the RX data, in time, from the start of the TX data.
        This will be calculated as the time at centre sample of the first pulse. In seconds.
    blanks
        A list of sample indices that should not be used for acfs because they were samples
        taken when transmitting.

    **Pulses is a list of pulse dictionaries. The pulse dictionary keys are:**

    isarepeat
        Boolean, True if the pulse is exactly the same as the last pulse in the sequence.
    pulse_timing_us
        The time past the start of sequence for this pulse to start at (us).
    slice_id
        The slice_id that corresponds to this pulse and gives the information about the
        experiment and pulse information (frequency, num_ranges, first_range, etc.).
    slice_pulse_index
        The index of the pulse in its own slice's sequence.
    pulse_len
        The length of the pulse (us)
    intra_pulse_start_time
        If the pulse is combined with another pulse and they transmit in a single USRP
        burst, then we need to know if there is an offset from one pulse's samples being
        sent and the other pulse's samples being sent.
    combined_pulse_index
        The combined_pulse_index is the index corresponding with actual number of pulses
        that will be sent to driver, after combinations are completed. Multiple pulse
        dictionaries in self.pulses can have the same combined_pulse_index if they are
        combined together, ie are close enough in timing that T/R will not go low
        between them, and we will combine the samples of both pulses into one set to
        send to the driver.
    pulse_shift
        Phase shift for this pulse, for doing pulse coding.
    iscombined
        Boolean, true if there is another pulse with the same combined_pulse_index.
    combine_total
        Total number of pulse dictionaries that have the same combined_pulse_index as
        this one. (minimum number = 1, itself).
    combine_index
        Index of this pulse dictionary in regards to all the other pulse dictionaries that
        have the same combined_pulse_index.
    """

    def __init__(self, seqn_keys, sequence_slice_dict, sequence_interface, transmit_metadata):
        """
        :param seqn_keys: list of slice_ids that need to be included in this sequence.
        :param sequence_slice_dict: the slice dictionary that explains the parameters of each
         slice that is included in this sequence. Keys are the slice_ids included
         and values are dictionaries including all necessary slice parameters as keys.
        :param sequence_interface: the interfacing dictionary that describes how to interface the
         slices that are included in this sequence. Keys are tuples of format
         (slice_id_1, slice_id_2) and values are of interface_types set up in
         experiment_prototype.
        :param transmit_metadata: metadata from the config file that is useful here.
        """

        # TODO make diagram(s) for pulse combining algorithm
        # TODO make diagram for pulses that are repeats, showing clearly what intra_pulse_start_time,
        # and pulse_shift are.
        ScanClassBase.__init__(self, seqn_keys, sequence_slice_dict, sequence_interface,
                               transmit_metadata)


        txrate = self.transmit_metadata['txrate']
        txctrfreq = self.transmit_metadata['txctrfreq']
        main_antenna_count = self.transmit_metadata['main_antenna_count']
        main_antenna_spacing = self.transmit_metadata['main_antenna_spacing']
        pulse_ramp_time = self.transmit_metadata['pulse_ramp_time']
        max_usrp_dac_amplitude = self.transmit_metadata['max_usrp_dac_amplitude']
        tr_window_time = self.transmit_metadata['tr_window_time']

        self.basic_slice_pulses = {}
        single_pulse_timing = []
        for slice_id in self.slice_ids:

            exp_slice = self.slice_dict[slice_id]
            tx_freq_khz = float(exp_slice['txfreq'])
            wave_freq = tx_freq_khz - txctrfreq
            wave_freq_hz = wave_freq * 1000

            if not exp_slice['rxonly']:
                basic_samples, real_freq = get_samples(txrate, wave_freq_hz,
                                                   float(exp_slice['pulse_len']) / 1e6,
                                                   pulse_ramp_time,
                                                   1.0,
                                                   exp_slice['iwavetable'],
                                                   exp_slice['qwavetable'])
                if real_freq != wave_freq_hz:
                    errmsg = 'Actual Frequency {} is Not Equal to Intended Wave Freq {}'.format(real_freq,
                                                                                                wave_freq_hz)
                    raise ExperimentException(errmsg)  # TODO change to warning? only happens on non-SINE

                # this will zero out non tx antenna samples
                main_antennas = np.zeros(main_antenna_count, dtype=np.int)
                temp = np.arange(main_antenna_count)
                main_antennas[exp_slice['tx_antennas']] = temp[exp_slice['tx_antennas']]

                beam_rads = np.pi / 180 * np.array(exp_slice['beam_angle'], dtype=np.float64)

                x = ((main_antenna_count / 2.0 - main_antennas) * main_antenna_spacing)
                x *= 2 * np.pi * (tx_freq_khz * 1000)

                y = np.cos(np.pi / 2.0 - beam_rads) / speed_of_light

                phase_shift = np.fmod(np.outer(y, x), 2.0 * np.pi) # beams by antenna
                phase_shift = np.exp(1j * phase_shift)

                phased_samps_for_beams = np.outer(phase_shift.flatten(), basic_samples)

                #beams by antenna by samples
                phased_samps_for_beams = phased_samps_for_beams.reshape(phase_shift.shape + basic_samples.shape)

                self.basic_slice_pulses[slice_id] = phased_samps_for_beams
            else:
                self.basic_slice_pulses[slice_id] = []

            for pulse_time in exp_slice['pulse_sequence']:
                pulse_timing_us = pulse_time * exp_slice['tau_spacing'] + exp_slice['seqoffset']

                single_pulse_timing.append({'start_time_us' : pulse_timing_us,
                                            'pulse_len_us' : exp_slice['pulse_len'],
                                            'slice_id' : slice_id})


        single_pulse_timing = sorted(single_pulse_timing, key=lambda d: d['start_time_us'])

        def make_pulse_dict(pulse_timing_info):
            return {'start_time_us' : pulse_timing_info['start_time_us'],
                      'total_pulse_len' : pulse_timing_info['pulse_len_us'],
                      'component_info' : [pulse_timing_info]
                    }

        pulse_data = make_pulse_dict(single_pulse_timing[0])
        combined_pulses_metadata = []

        for pulse_time in single_pulse_timing[1:]:
            pulse_timing_us = pulse_time['start_time_us']
            pulse_len_us = pulse_time['pulse_len_us']

            cp_timing_us = pulse_data['start_time_us']
            cp_pulse_len_us = pulse_data['total_pulse_len']

            min_sep = self.transmit_metadata['minimum_pulse_separation']
            if pulse_timing_us < cp_timing_us + cp_pulse_len_us + min_sep:
                new_pulse_len = pulse_timing_us - cp_timing_us + pulse_len_us

                pulse_data['total_pulse_len'] = new_pulse_len
                pulse_data['component_info'].append(pulse_time)
            else:
                combined_pulses_metadata.append(pulse_data)
                pulse_data = make_pulse_dict(pulse_time)

        combined_pulses_metadata.append(pulse_data)

        power_divider = max([len(p['component_info']) for p in combined_pulses_metadata])
        all_antennas = []
        for slice_id in self.slice_ids:
            self.basic_slice_pulses[slice_id] *= max_usrp_dac_amplitude / power_divider

            slice_tx_antennas = self.slice_dict[slice_id]['tx_antennas']
            all_antennas.extend(slice_tx_antennas)


        sequence_antennas = list(set(all_antennas))
        num_pulses = len(combined_pulses_metadata)
        for i in range(num_pulses):
            combined_pulses_metadata[i]['transmit_metadata'] = {}
            pulse_transmit_data = combined_pulses_metadata[i]['transmit_metadata']

            pulse_transmit_data['startofburst'] = i == 0
            pulse_transmit_data['endofburst'] = i == (num_pulses - 1)

            pulse_transmit_data['pulse_antennas'] = sequence_antennas
            pulse_transmit_data['samples_array'] = None
            pulse_transmit_data['timing'] = combined_pulses_metadata[i]['start_time_us']
            pulse_transmit_data['isarepeat'] = False

        self.combined_pulses_metadata = combined_pulses_metadata


        # FIND the max scope sync time
        # The gc214 receiver card in the old system required 19 us for sample delay and another 10 us
        # as empirically discovered. in that case delay = (num_ranges + 19 + 10) * pulse_len.
        # Now we will remove those values. In the old design scope sync was used directly to
        # determine how long to sample. Now we will calculate the number of samples to receive
        # (numberofreceivesamples) using scope sync and send that to the driver to sample at
        # a specific rxrate (given by the config).

        # number of samples for the first range for all slice ids

        range_as_samples = lambda x,y: int(math.ceil(x/y))
        first_range_samples = {slice_id : range_as_samples(self.slice_dict[slice_id]['first_range'],
                                                            self.slice_dict[slice_id]['range_sep'])
                                for slice_id in self.slice_ids}

        # time for number of ranges given, in us, taking into account first_range and num_ranges.
        self.ssdelay = max([(self.slice_dict[slice_id]['num_ranges'] + first_range_samples[slice_id]) *
                            (1.0e6/self.transmit_metadata['output_rx_rate']) for slice_id in self.slice_ids])


        # The delay is long enough for any slice's pulse length and num_ranges to be accounted for.

        # FIND the sequence time. Time before the first pulse is 70 us when RX and TR set up for the first pulse. The
        # timing to the last pulse is added, as well as its pulse length and the RX/TR delay at the end of last pulse.
        # tr_window_time is originally in seconds, convert to us.
        self.seqtime = (2 * tr_window_time * 1.0e6 +
                        self.combined_pulses_metadata[-1]['start_time_us'] +
                       self.combined_pulses_metadata[-1]['total_pulse_len'])

        # FIND the total scope sync time and number of samples to receive.
        self.sstime = self.seqtime + self.ssdelay

        # number of receive samples will round down
        # This is the number of receive samples to receive for the entire duration of the
        # sequence and afterwards. This starts before first pulse is sent and goes until the
        # end of the scope sync delay which is there for the amount of time necessary to get
        # the echoes from the specified number of ranges.
        self.numberofreceivesamples = int(self.transmit_metadata['rx_sample_rate'] * self.sstime *
                                          1e-6)

        sample_time = (int(self.combined_pulses_metadata[0]['total_pulse_len'] / 2) + tr_window_time) / txrate
        self.first_rx_sample_time = sample_time

        self.blanks = self.find_blanks()

        self.output_encodings = collections.defaultdict(list)

    def make_sequence(self, beam_iter, sequence_num):
        main_antenna_count = self.transmit_metadata['main_antenna_count']
        txrate = self.transmit_metadata['txrate']
        tr_window_time = self.transmit_metadata['tr_window_time']

        sequence = np.zeros([main_antenna_count, self.numberofreceivesamples], dtype=np.complex64)

        for slice_id in self.slice_ids:
            exp_slice = self.slice_dict[slice_id]
            basic_samples = self.basic_slice_pulses[slice_id][beam_iter]

            num_pulses = len(exp_slice['pulse_sequence'])
            encode_fn = exp_slice['pulse_phase_offset']
            if encode_fn:
                num_samples = basic_samples.shape[1]
                phase_encoding = encode_fn(beam_iter, sequence_num, num_pulses, num_samples)

                # Reshape as vector if 1D, else stays the same.
                phase_encoding = phase_encoding.reshape((phase_encoding.shape[0],-1))

                self.output_encodings[slice_id].append(phase_encoding)

                phase_encoding = np.exp(1j * phase_encoding[:,np.newaxis,:])
                samples = phase_encoding * basic_samples

            else:
                samples = np.repeat(basic_samples[np.newaxis,:,:], num_pulses, axis=0)

            tr_window_num_samps = round(tr_window_time * txrate)
            for i, pulse in enumerate(self.combined_pulses_metadata):
                for component_info in pulse['component_info']:
                    if component_info['slice_id'] == slice_id:
                        pulse_timing_us = component_info['start_time_us']
                        pulse_sample_start = round(txrate * (pulse_timing_us * 1e-6))
                        pulse_samples_len = samples.shape[-1]

                        start = tr_window_num_samps + pulse_sample_start
                        end = start + pulse_samples_len
                        pulse_piece = sequence[...,start:end]

                        np.add(pulse_piece, samples[i], out=pulse_piece)

        pulse_data = []
        for i, pulse in enumerate(self.combined_pulses_metadata):
            pulse_timing_us = pulse['start_time_us']
            pulse_sample_start = round(txrate * (pulse_timing_us * 1e-6))

            pulse_len = pulse['total_pulse_len']
            num_samples = round(txrate * (pulse_len * 1e-6))
            start = pulse_sample_start
            end = start + num_samples + 2 * tr_window_num_samps
            samples = sequence[...,start:end]

            new_pulse_info = copy.deepcopy(pulse['transmit_metadata'])
            new_pulse_info['samples_array'] = samples

            if i != 0:
                last_pulse = pulse_data[i-1]['samples_array']
                if samples.shape == last_pulse.shape:
                    if np.isclose(samples, last_pulse).all():
                            new_pulse_info['isarepeat'] = True

            pulse_data.append(new_pulse_info)

        return pulse_data

    def find_blanks(self):
        """
        Sets the blanks. Must be run after first_rx_sample_time is set inside the
        build_pulse_transmit_data function. Called from inside the build_pulse_transmit_data
        function.
        """
        blanks = []
        sample_time = 1.0/float(self.transmit_metadata['output_rx_rate'])
        pulses_time = []

        for pulse in self.combined_pulses_metadata:
            start_time = pulse['start_time_us']
            pulse_len = pulse['total_pulse_len']
            pulse_start_stop = [start_time * 1.0e-6, (start_time + pulse_len) * 1.0e-6]
            pulses_time.append(pulse_start_stop)

        output_samples_in_sequence = int(self.sstime * 1.0e-6/sample_time)
        sample_times = [self.first_rx_sample_time + i*sample_time for i in
                        range(0, output_samples_in_sequence)]
        for sample_num, time_s in enumerate(sample_times):
            for pulse_start_stop in pulses_time:
                if pulse_start_stop[0] <= time_s <= pulse_start_stop[1]:
                    blanks.append(sample_num)

        return blanks
