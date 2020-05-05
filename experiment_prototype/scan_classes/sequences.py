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

from sample_building.sample_building import get_samples, get_phase_shift
from experiment_prototype.scan_classes.scan_class_base import ScanClassBase
from experiment_prototype.experiment_exception import ExperimentException

sys.path.append(os.environ["BOREALISPATH"])
import utils.shared_macros.shared_macros as sm

sequence_print = sm.MODULE_PRINT("sequence building", "magenta")
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
    basic_slice_pulses
        A dictionary that holds pre-computed tx samples for each slice. Each dictionary value is a
        multi-dimensional array that holds a beamformed set of samples for each antenna for all
        beam directions.
    combined_pulses_metadata
        This list holds dictionary metadata for all pulses in the sequence. This metadata holds all
        the info needed to combine pulses if pulses are mixed.
            start_time_us - start time of the pulse in us, relative to the first pulse in sqn
            total_pulse_len - total length of the pulse that includes len of all combined pulses.
            component_info - a list of all the pre-combined pulse components
                        (incl their length and start time) that are in the combined pulseAlso in us.
            transmit_metadata - dictionary hold the transmit metadata that will be sent to driver.
    output_encodings
        This dict will hold a list of all the encodings used during an aveperiod for each slice.
        These will be used for data write later.
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

        ScanClassBase.__init__(self, seqn_keys, sequence_slice_dict, sequence_interface,
                               transmit_metadata)


        txrate = self.transmit_metadata['txrate']
        txctrfreq = self.transmit_metadata['txctrfreq']
        main_antenna_count = self.transmit_metadata['main_antenna_count']
        main_antenna_spacing = self.transmit_metadata['main_antenna_spacing']
        intf_antenna_count = self.transmit_metadata['intf_antenna_count']
        intf_antenna_spacing = self.transmit_metadata['intf_antenna_spacing']
        pulse_ramp_time = self.transmit_metadata['pulse_ramp_time']
        max_usrp_dac_amplitude = self.transmit_metadata['max_usrp_dac_amplitude']
        tr_window_time = self.transmit_metadata['tr_window_time']
        output_rx_rate = self.transmit_metadata['output_rx_rate']

        self.basic_slice_pulses = {}
        self.rx_beam_phases = {}
        single_pulse_timing = []

        # For each slice calculate beamformed samples and place into the basic_slice_pulses dictionary.
        # Also populate the pulse timing metadata and place into single_pulse_timing
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

                main_phase_shift = get_phase_shift(exp_slice['beam_angle'], tx_freq_khz,
                                                    main_antenna_count, main_antenna_spacing)
                intf_phase_shift = get_phase_shift(exp_slice['beam_angle'], tx_freq_khz,
                                                    intf_antenna_count, intf_antenna_spacing,
                                                    intf_offset[0])
                # We want to apply all the phases to the basic samples. We can flatten the phases
                # so that can multiply them all with the basic samples. This can later be reshaped
                # so that each antenna has a set of phased samples for each beam.
                # If there are N antennas and M beams

                # if we let antennaxbeamy = abxy from the previous matrix (now flattened to be able to multiply)
                # and basic_samples[i] = bsi
                # And there are S basic_samples then...

                # Now we have:
                # [ab00bs0 ab00bs1 ... ab00bsS-1
                # ab10bs0 ab10bs1 ... ab10bsS-1
                # ...
                # ab(N-1)0bs0 ab(N-1)0bs1 ... ab(N-1)0bsS-1
                # ab01bs0 ab01bs1 ... ab01bsS-1
                # ...
                # ab(N-1)1bs0 ab(N-1)1bs1 ... ab(N-1)1*bsS-1
                # ...
                # ab(N-1)(M-1)*bs0 ... ... ...........ab(N-1)(M-1)*bsS-1]

                # And to access a sample for a specific beam and antenna:
                # phased_samps_for_beams[antenna+(N)*beam][sample]
                phased_samps_for_beams = np.outer(main_phase_shift.flatten(), basic_samples)

                # beams by antenna by samples
                phased_samps_for_beams = phased_samps_for_beams.reshape(main_phase_shift.shape +
                                                                        basic_samples.shape)

                # zero out the antennas not being used.
                temp = np.zeros_like(phased_samps_for_beams, dtype=phased_samps_for_beams.dtype)
                temp[:,exp_slice['tx_antennas'],:] = phased_samps_for_beams[:,exp_slice['tx_antennas'],:]
                phased_samps_for_beams = temp[:,exp_slice['tx_antennas'],:]

                self.basic_slice_pulses[slice_id] = phased_samps_for_beams
            else:
                rx_freq_khz = experiment.slice_dict[slice_id]['rxfreq']
                main_phase_shift = get_phase_shift(exp_slice['beam_angle'], rx_freq_khz,
                                                    main_antenna_count, main_antenna_spacing)
                intf_phase_shift = get_phase_shift(exp_slice['beam_angle'], rx_freq_khz,
                                                    intf_antenna_count, intf_antenna_spacing,
                                                    intf_offset[0])

                self.basic_slice_pulses[slice_id] = []

            self.rx_beam_phases[slice_id] = {'main' : main_phase_shift, 'intf' : intf_phase_shift}

            for pulse_time in exp_slice['pulse_sequence']:
                pulse_timing_us = pulse_time * exp_slice['tau_spacing'] + exp_slice['seqoffset']
                pulse_sample_start = round((pulse_timing_us * 1e-6) * txrate)

                pulse_num_samps = round((exp_slice['pulse_len'] * 1e-6) * txrate)

                single_pulse_timing.append({'start_time_us' : pulse_timing_us,
                                            'pulse_len_us' : exp_slice['pulse_len'],
                                            'pulse_sample_start' : pulse_sample_start,
                                            'pulse_num_samps' : pulse_num_samps,
                                            'slice_id' : slice_id})


        single_pulse_timing = sorted(single_pulse_timing, key=lambda d: d['start_time_us'])

        # Combine any pulses closer than the minimum separation time into a single pulse data
        # dictionary and append to the list of all combined pulses, combined_pulses_metadata.
        tr_window_num_samps = round((tr_window_time * 1e-6) * txrate)
        def initialize_combined_pulse_dict(pulse_timing_info):
            return {'start_time_us' : pulse_timing_info['start_time_us'],
                      'total_pulse_len' : pulse_timing_info['pulse_len_us'],
                      'pulse_sample_start' : pulse_timing_info['pulse_sample_start'],
                      'total_num_samps' : pulse_timing_info['pulse_num_samps'],
                      'tr_window_num_samps' : tr_window_num_samps,
                      'component_info' : [pulse_timing_info]
                    }

        pulse_data = initialize_combined_pulse_dict(single_pulse_timing[0])
        combined_pulses_metadata = []

        # determine where pulses occur in the sequence. This will be important if there are overlaps
        for pulse_time in single_pulse_timing[1:]:
            pulse_timing_us = pulse_time['start_time_us']
            pulse_len_us = pulse_time['pulse_len_us']
            pulse_sample_start = pulse_time['pulse_sample_start']
            pulse_num_samps = pulse_time['pulse_num_samps']

            last_timing_us = pulse_data['start_time_us']
            last_pulse_len_us = pulse_data['total_pulse_len']
            last_sample_start = pulse_data['pulse_sample_start']
            last_pulse_num_samps = pulse_data['total_num_samps']

            # If there are overlaps (two pulses within minimum separation time) then make them
            # into one single pulse
            min_sep = self.transmit_metadata['minimum_pulse_separation']
            if pulse_timing_us < last_timing_us + last_pulse_len_us + min_sep:
                new_pulse_len = pulse_timing_us - last_timing_us + pulse_len_us
                new_pulse_samps = pulse_sample_start - last_sample_start + pulse_num_samps

                pulse_data['total_pulse_len'] = new_pulse_len
                pulse_data['total_num_samps'] = new_pulse_samps
                pulse_data['component_info'].append(pulse_time)
            else: # pulses do not overlap
                combined_pulses_metadata.append(pulse_data)
                pulse_data = initialize_combined_pulse_dict(pulse_time)

        combined_pulses_metadata.append(pulse_data)

        # normalize all pulses to max usrp dac amplitude.
        power_divider = max([len(p['component_info']) for p in combined_pulses_metadata])
        all_antennas = []
        for slice_id in self.slice_ids:
            self.basic_slice_pulses[slice_id] *= max_usrp_dac_amplitude / power_divider

            slice_tx_antennas = self.slice_dict[slice_id]['tx_antennas']
            all_antennas.extend(slice_tx_antennas)

        sequence_antennas = list(set(all_antennas))

        # predetermine some of the transmit metadata.
        num_pulses = len(combined_pulses_metadata)
        for i in range(num_pulses):
            combined_pulses_metadata[i]['pulse_transmit_data'] = {}
            pulse_transmit_data = combined_pulses_metadata[i]['pulse_transmit_data']

            pulse_transmit_data['startofburst'] = i == 0
            pulse_transmit_data['endofburst'] = i == (num_pulses - 1)

            pulse_transmit_data['pulse_antennas'] = sequence_antennas
            pulse_transmit_data['samples_array'] = None
            pulse_transmit_data['timing'] = combined_pulses_metadata[i]['start_time_us']
            pulse_transmit_data['isarepeat'] = False

        # print out pulse information for logging.
        for i, cpm in enumerate(combined_pulses_metadata):
            message = "Pulse {}: start time(us) {}  start sample {}".format(i,
                                                                            cpm['start_time_us'],
                                                                            cpm['pulse_sample_start'])
            sequence_print(message)

            message = "          pulse length(us) {}  pulse num samples {}".format(i,
                                                                                   cpm['pulse_len_us'],
                                                                                   cpm['total_num_samps'])
            sequence_print(message)

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

        # FIND the sequence time. Time before the first pulse is 70 us when RX and TR set up for the first pulse. Thet
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

        first_slice_pulse_len = self.combined_pulses_metadata[0]['component_info'][0]['pulse_num_samps']
        offset_to_start = int(first_slice_pulse_len / 2) + tr_window_num_samps
        self.first_rx_sample_time = offset_to_start

        self.blanks = self.find_blanks()

        self.output_encodings = collections.defaultdict(list)


        # create debug dict for tx samples.
        debug_dict = {'txrate' : txfreq,
                      'txctrfreq' : txctrfreq,
                      'pulse_timing' : [],
                      'pulse_sample_start' : [],
                      'sequence_samples' : {},
                      'decimated_samples' : {},
                      'dmrate_error' : None,
                      'dmrate' : None
                      }

        dm_rate = txrate / output_rx_rate
        dm_rate_error = dm_rate - round(dm_rate)
        dm_rate = round(dm_rate)

        debug_dict['dmrate'] = dm_rate
        debug_dict['dmrate_error'] = dm_rate_error

        for i, cpm in enumerate(combined_pulses_metadata):
            debug_dict['pulse_timing'].append(cpm['start_time_us'])
            debug_dict['pulse_sample_start'].append(cpm['pulse_sample_start'])

        for i in range(main_antenna_count):
            debug_dict['sequence_samples'][i] = []
            debug_dict['decimated_samples'][i] = []

        self.debug_dict = debug_dict



    def make_sequence(self, beam_iter, sequence_num):
        """
        Create the samples needed for each pulse in the sequence. This function is optimized to
        be able to generate new samples every sequence if needed.

        :param      beam_iter:     The beam iterator
        :type       beam_iter:     int
        :param      sequence_num:  The sequence number in the ave period
        :type       sequence_num:  int

        :returns:   Transmit data for each pulse, including timing and samples
        :rtype:     list
        :returns:   The transmit sequence and related data to use for debug.
        :rtype:     Dict
        """
        main_antenna_count = self.transmit_metadata['main_antenna_count']
        txrate = self.transmit_metadata['txrate']
        tr_window_time = self.transmit_metadata['tr_window_time']

        buffer_len = int(txrate * self.sstime * 1e-6)
        # This is gonna act as buffer for mixing pulses. Its the length of the receive samples
        # since we know this will be large enough to hold samples at any pulse position. There will
        # be a buffer for each antenna.
        sequence = np.zeros([main_antenna_count, buffer_len], dtype=np.complex64)

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

                # we have [p,e] and [a,s], but we want [p,a,(e*s)]. Adding null axis to encoding
                # will produce this result.
                phase_encoding = np.exp(1j * phase_encoding[:,np.newaxis,:])
                samples = phase_encoding * basic_samples

            else:
                samples = np.repeat(basic_samples[np.newaxis,:,:], num_pulses, axis=0)

            # sum the samples into their position in the sequence buffer. Find where the relative
            # timing of each pulse matches the sample number in the buffer. Directly sum the samples
            # for each pulse into the buffer position. If any pulses overlap, this is how they will
            # be mixed.
            for i, pulse in enumerate(self.combined_pulses_metadata):
                for component_info in pulse['component_info']:
                    if component_info['slice_id'] == slice_id:
                        pulse_sample_start = component_info['pulse_sample_start']
                        pulse_samples_len = component_info['pulse_num_samps']

                        start = pulse['tr_window_num_samps'] + pulse_sample_start
                        end = start + pulse_samples_len
                        pulse_piece = sequence[...,start:end]

                        np.add(pulse_piece, samples[i], out=pulse_piece)

        # copy the encoded and combined samples into the metadata for the sequence.
        pulse_data = []
        for i, pulse in enumerate(self.combined_pulses_metadata):
            pulse_sample_start = pulse['pulse_sample_start']

            num_samples = pulse['total_num_samps']
            start = pulse_sample_start
            end = start + num_samples + 2 * pulse['tr_window_num_samps']
            samples = sequence[...,start:end]

            new_pulse_info = copy.deepcopy(pulse['pulse_transmit_data'])
            new_pulse_info['samples_array'] = samples

            if i != 0:
                last_pulse = pulse_data[i-1]['samples_array']
                if samples.shape == last_pulse.shape:
                    if np.isclose(samples, last_pulse).all():
                            new_pulse_info['isarepeat'] = True

            pulse_data.append(new_pulse_info)


        debug_dict = copy.copy(self.debug_dict)
        def fill_dbg_dict()
            """
            This needs major speed optimization to work at realtime
            """
            decimated_samples = sequence[:,debug_dict['dmrate']]
            for i in range(main_antenna_count):
                samples = sequence[i]
                deci_samples = decimated_samples[i]

                samples_dict = {'real' : samples.real.tolist(),
                                'imag' : samples.imag.tolist()}

                deci_samples_dict = {'real' : decimated_samples.real.tolist(),
                                     'imag' : decimated_samples.imag.tolist()}

                debug_dict['sequence_samples'][i] = samples_dict
                debug_dict['decimated_sequence'][i] = deci_samples_dict

        if __debug__:
            fill_dbg_dict()
        else:
            debug_dict = None

        return pulse_data, debug_dict

    def find_blanks(self):
        """
        Finds the blanked samples after all pulse positions are calculated.
        """
        blanks = []
        rx_sample_period = 1.0/float(self.transmit_metadata['output_rx_rate'])
        pulses_time = []

        for pulse in self.combined_pulses_metadata:
            start_time = pulse['start_time_us']
            pulse_len = pulse['total_pulse_len']
            pulse_start_stop = [start_time * 1.0e-6, (start_time + pulse_len) * 1.0e-6]
            pulses_time.append(pulse_start_stop)

        output_samples_in_sequence = int(self.sstime * 1.0e-6/rx_sample_period)
        sample_times = [self.first_rx_sample_time + i*rx_sample_period for i in
                        range(0, output_samples_in_sequence)]
        for sample_num, time_s in enumerate(sample_times):
            for pulse_start_stop in pulses_time:
                if pulse_start_stop[0] <= time_s <= pulse_start_stop[1]:
                    blanks.append(sample_num)

        return blanks

    def get_rx_phases(beam_iter):
        """
        Gets the receive phases for a given beam

        :param      beam_iter:  The beam iter in a scan.
        :type       beam_iter:  int

        :returns:   The receive phases.
        :rtype:     Array of phases for each antenna
        """

        temp_dict = copy.copy(self.rx_beam_phases)

        for k,v in temp_dict.items():
            beam_num = self.slice_dict['beam_order'][beam_iter]
            v['main'] = v['main'][beam_num]
            v['intf'] = v['intf'][beam_num]

        return temp_dict

