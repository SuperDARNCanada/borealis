#!/usr/bin/python

from operator import itemgetter

from sample_building.sample_building import make_pulse_samples
from experiment_prototype.scan_classes.scan_class_base import ScanClassBase


class Sequence(ScanClassBase):
    """
    Scans are made up of AveragingPeriods, these are typically a 3sec time of
    the same pulse sequence pointing in one direction.  AveragingPeriods are made
    up of Sequences, typically the same sequence run ave. 20-30 times after a clear
    frequency search.  Sequences are made up of pulses, which is a list of dictionaries 
    where each dictionary describes a pulse. 
    """

    def __init__(self, seqn_keys, sequence_slice_dict, sequence_interface, options):
        """
        Set up the sequence class. 
        
        This includes setting up the pulses. self.pulses is a list of
        dictionaries where one dictionary = one pulse. The dictionary keys are : isarepeat, 
        pulse_timing_us, slice_id, slice_pulse_index, pulse_len, intra_pulse_start_time, 
        combined_pulse_index, pulse_shift, iscombined, combine_total, and combine_index.
        
        Other specific members of the sequence are:
        total_combined_pulses: the total number of pulses to be sent by the driver. This may not
            be the sum of pulses in all slices in the sequence, as some pulses may need to be 
            combined because they are overlapping in timing. This is the number of pulses in the 
            combined sequence, or the number of times T/R signal goes high in the sequence.
        power_divider: the power ratio per slice. If there are multiple slices in the same 
            pulse then we must reduce the output amplitude to potentially accommodate multiple
            frequencies.
        last_pulse_len: the length of the last pulse (us)
        ssdelay: delay past the end of the sequence to receive for (us) - function of nrang and
            pulse_len. ss stands for scope sync.
        seqtime: the amount of time for the whole sequence to transmit, until the logic signal 
            switches low on the last pulse in the sequence.
        sstime: ssdelay + seqtime (total time for receiving.
        numberofreceivesamples: the number of receive samples to take, given the rx rate, during 
            the sstime.
        
        :param seqn_keys - slice ids in this sequence
        :param sequence_slice_dict
        :param sequence_interface
        :param options

        """
        # TODO describe pulse dictionary in docs
        # TODO make diagram(s) for pulse combining algorithm
        # TODO make diagram for pulses that are repeats, showing clearly what intra_pulse_start_time,
        # and pulse_shift are.
        ScanClassBase.__init__(self, seqn_keys, sequence_slice_dict, sequence_interface,
                               options)

        # TODO: pass clear frequencies to pass to pulses

        # All interfacing at this point is PULSE.

        # TODO: use number of frequencies to determine power output of each frequency (1/2, 1/3)


        # TODO add in seqoffset value from the slice to pulse timing!!
        pulses = []
        # Getting a list of pulses, where pulse is a dictionary
        for slice_id in self.slice_ids:
            for slice_pulse_index, pulse_time in enumerate(self.slice_dict[slice_id]['pulse_sequence']):
                pulse_timing_us = pulse_time*self.slice_dict[slice_id]['mpinc']
                pulses.append({'pulse_timing_us': pulse_timing_us, 'slice_id': slice_id,
                                   'slice_pulse_index': slice_pulse_index,
                                   'pulse_len': self.slice_dict[slice_id]['pulse_len'],
                                   'pulse_shift': self.slice_dict[slice_id]['pulse_shift'][slice_pulse_index]})

        self.pulses = sorted(pulses, key=itemgetter('pulse_timing_us', 'slice_id'))
        # Will sort by timing first and then by slice if timing =. This is all pulses in the sequence,
        # in a list of dictionaries.

        if __debug__:
            print(self.pulses)

        # Set up the combined pulse list
        this_pulse_index = 0
        combined_pulse_index = 0
        total_combined_pulses = 0
        while this_pulse_index < len(self.pulses):
            pulse = self.pulses[this_pulse_index]
            pulse['isarepeat'] = False  # Will change later if True
            pulse['intra_pulse_start_time'] = 0  # Will change later if combined with another pulse
            pulse['combined_pulse_index'] = combined_pulse_index
            # The combined_pulse_index is the index corresponding with actual number of pulses that
            # will be sent to driver, after combinations are completed. Multiple pulse dictionaries
            # in self.pulses can have the same combined_pulse_index if they are combined together,
            # ie are close enough in timing that T/R will not go low between them, and we will
            # combine the samples of both pulses into one set to send to the driver.

            # Now will determine if we are combining samples based on timing of pulses
            combine_pulses = True
            next_pulse_index = this_pulse_index + 1

            if this_pulse_index == len(self.pulses) - 1:  # last pulse is not combined - end
                combine_pulses = False
                pulse['iscombined'] = False
                pulse['combine_total'] = 1  # no pulses combined.
                pulse['combine_index'] = 0
                total_combined_pulses = pulse['combined_pulse_index'] + 1

            while combine_pulses:
                if self.pulses[next_pulse_index]['pulse_timing_us'] <= pulse['pulse_timing_us']\
                         + pulse['pulse_len'] + self.options.minimum_pulse_separation:
                    # combine pulse and next_pulse
                    next_pulse = self.pulses[next_pulse_index]

                    if 'iscombined' in pulse.keys():  # already combined with a previous pulse
                        # if iscombined key exists here it must be = True as we only set it false
                        # when combine_pulses = False therefore we would not get into this while
                        # loop were it false.
                        for index in range(this_pulse_index,next_pulse_index):
                            self.pulses[index]['combine_total'] += 1
                    else:
                        # We are combining pulses, set keys iscombined, combine_total, combine_index
                        pulse['iscombined'] = True
                        pulse['combine_total'] = 2  # 2 pulses are combined (so far), pulse and next_pulse
                        pulse['combine_index'] = 0

                    # Next pulse is being combined into the this pulse. That means that we need
                    # to set next_pulse[intra_pulse_start_time] = next_pulse[pulse_timing_us] -
                    # this_pulse[pulse_timing_us], and then pulse_timing_us can be set to the same
                    # value for both pulses. intra_pulse_start_time is the offset time after the
                    # combined_pulse has started, when we start transmitting the samples for the
                    # pulse.

                    next_pulse['iscombined'] = True
                    next_pulse['combine_total'] = pulse['combine_total']
                    next_pulse['combine_index'] = next_pulse['combine_total'] - 1
                    next_pulse['isarepeat'] = False
                    next_pulse['intra_pulse_start_time'] = next_pulse['pulse_timing_us'] \
                        - pulse['pulse_timing_us']
                    next_pulse['pulse_timing_us'] = pulse['pulse_timing_us']
                    next_pulse['combined_pulse_index'] = pulse['combined_pulse_index']
                    next_pulse_index = next_pulse_index + 1

                    if next_pulse_index == len(self.pulses):  # last pulse has been combined - end
                        combine_pulses = False
                        total_combined_pulses = pulse['combined_pulse_index'] + 1
                else:
                    combine_pulses = False
                    if 'iscombined' not in pulse.keys():
                        pulse['iscombined'] = False
                        pulse['combine_total'] = 1  # no pulses combined.
                        pulse['combine_index'] = 0

            this_pulse_index = next_pulse_index
            combined_pulse_index += 1
            # Jump ahead depending how many pulses we've combined.

        # Total combined pulses is the number of pulses in the sequence after some pulses have
        # been combined because they are at the same time or close in timing. This translates to
        # the number of times T/R would go high and the number of times we would send samples
        # but it may or may not be equal to the sum of pulses in all slices involved in this
        # sequence.
        self.total_combined_pulses = total_combined_pulses

        # max combine_total is = power_divider for this sequence. power divider is unique by
        # sequence. Note that if you have combined pulses for some pulses and not for others,
        # even the pulses that are not combined (for example, only running one frequency) will
        # have the lower power level 1/power_divider.
        # TODO should powers be adjusted by frequency, other factors outside of the specific
        # sequence. Should this be specifiable inside the experiment. (so you could give one
        # slice a higher power weighting if you wanted)

        self.power_divider = max([p['combine_total'] for p in self.pulses])

        # All pulse dictionaries with the same combined_pulse_index make up a combined pulse.
        # A repeat is when a combined pulse is after a combined pulse that is
        # exactly the same. The only thing that would be different is the start time.

        for pulse_index in range(1, len(self.pulses)):
            # 0th pulse can never be a repeat, but isarepeat key  was initialized to False so we can
            # leave it.
            pulse = self.pulses[pulse_index]
            last_pulse = self.pulses[pulse_index - 1]
            if pulse['iscombined']:
                if pulse['combine_index'] != 0:
                    # this pulse will be a repeat if the pulse with combine_index of 0 is a repeat.
                    first_pulse_in_combination_index = pulse_index - pulse['combine_index']
                    pulse['isarepeat'] = self.pulses[first_pulse_in_combination_index]['isarepeat']
                elif not last_pulse['iscombined']:
                    # pulse['iscombined'] = True, pulse['combine_index'] = 0, but lastpulse iscombined = False.
                    pulse['isarepeat'] = False  # the last pulse must be combined in some way as well.
                elif pulse['combine_total'] != last_pulse['combine_total']:
                    pulse['isarepeat'] = False  # must have same number of slices combined in this
                    # combined pulse as in last combined pulse.
                else:
                    # pulse['iscombined'] = True, pulse['combine_index'] = 0, lastpulse
                    # iscombined = True, pulse combine_total = lastpulse combine_total.
                    # We must now check that the same slices are combined, and that the intra pulse
                    # timing for the slices is the same and the pulse_shift.
                    last_combined_pulse_index = last_pulse['combined_pulse_index']
                    this_combined_pulse_index = pulse['combined_pulse_index']
                    # get all pulse dictionaries that are in this combined pulse
                    combined_pulse_1 = [] # the previous combined pulse
                    combined_pulse_2 = [] # this combined pulse, which may be a repeat.
                    for a_pulse in self.pulses:
                        if a_pulse['combined_pulse_index'] == last_combined_pulse_index:
                            combined_pulse_1.append(a_pulse)

                    for b_pulse in self.pulses:
                        if b_pulse['combined_pulse_index'] == this_combined_pulse_index:
                            combined_pulse_2.append(b_pulse)

                    for pulse_1, pulse_2 in zip(combined_pulse_1, combined_pulse_2):
                        # combine_index should be the same because they were sorted
                        # chronologically in self.pulses.
                        if pulse_1['slice_id'] != pulse_2['slice_id']:
                            pulse['isarepeat'] = False
                            break
                        if pulse_1['intra_pulse_start_time'] != pulse_2['intra_pulse_start_time']:
                            pulse['isarepeat'] = False
                            break
                        if pulse_1['pulse_shift'] != pulse_2['pulse_shift']:
                            pulse['isarepeat'] = False
                            break
                    else:  # no break
                        pulse['isarepeat'] = True

        if __debug__:
            print(self.pulses)


        last_pulse = self.pulses[-1]
        self.last_pulse_len = max([(p['pulse_len'] + p['intra_pulse_start_time']) for p in
                              self.pulses[-(last_pulse['combine_total']):]])  # TODO does this work? pycharm saying max() will return a list

        # FIND the max scope sync time
        # The gc214 receiver card in the old system required 19 us for sample delay and another 10 us
        # as empirically discovered. in that case delay = (nrang + 19 + 10) * pulse_len.
        # Now we will remove those values. In the old design scope sync was used directly to
        # determine how long to sample. Now we will calculate the number of samples to receive
        # (numberofreceivesamples) using scope sync and send that to the driver to sample at
        # a specific rxrate (given by the config).

        self.ssdelay = max([self.slice_dict[slice_id]['nrang'] *
                            self.slice_dict[slice_id]['pulse_len'] for slice_id in self.slice_ids])
        # The delay is long enough for any slice's pulse length and nrang to be accounted for.

        # FIND the sequence time. Time before the first pulse is 70 us when RX and TR set up for the first pulse. The
        # timing to the last pulse is added, as well as its pulse length and the RX/TR delay at the end of last pulse.
        self.seqtime = self.options.atten_window_time_start + 2*self.options.tr_window_time + \
                       self.options.atten_window_time_end + self.pulses[-1]['pulse_timing_us'] + \
                       self.last_pulse_len

        # FIND the total scope sync time and number of samples to receive.
        self.sstime = self.seqtime + self.ssdelay
        # number of receive samples will round down
        # This is the number of receive samples to receive for the entire duration of the
        # sequence and afterwards. This starts before first pulse is sent and goes until the
        # end of the scope sync delay which is there for the amount of time necessary to get
        # the echoes from the specified number of ranges.
        self.numberofreceivesamples = int(self.options.rx_sample_rate * self.sstime * 1e-6)

    def build_pulse_transmit_data(self, slice_to_beamdir_dict, txctrfreq, txrate, options):
        #TODO only take in things you need or add needed params from options in the init function.
        # These params would be main_antenna_count, main_antenna_spacing and some pulse building
        # parameters needed for basic_samples function in sample_mapping.
        # TODO consider rewriting options to have a mapping of transmit antennas and their
        # TODO ... orientation (do not assume main array all in a line at certain spacing.
        # TODO ... this orientation would also then be passed to signal processing.
        """
        Build a list of pulse dictionaries including samples data to send to driver.
        :param: slice_to_beamdir_dict:
        :param: txctrfreq:
        :param: txrate:
        :param: options:
        :return: sequence_list: list of pulse dictionaries in correct order.
        """

        sequence_list = []
        for pulse_index in range(0, self.total_combined_pulses):
            pulse_transmit_data = {}
            # Pulses are in order

            one_pulse_list = [pulse for pulse in self.pulses if
                              pulse['combined_pulse_index'] == pulse_index]

            if pulse_index == 0:
                startofburst = True
            else:
                startofburst = False
            if pulse_index == self.total_combined_pulses - 1:
                endofburst = True
            else:
                endofburst = False

            repeat = one_pulse_list[0]['isarepeat']
            timing = one_pulse_list[0]['pulse_timing_us']
            pulse_samples = []
            if repeat:
                pulse_antennas = []

            else:
                # Initialize a list of lists for samples on all channels.
                # TODO: modify this function if we put a weighting on powers instead of just a
                # simple power_divider integer

                pulse_samples, pulse_antennas = (
                    make_pulse_samples(one_pulse_list, self.power_divider, self.slice_dict,
                                       slice_to_beamdir_dict, txctrfreq,
                                       txrate, options))
                # Can plot for testing here
                # plot_samples('channel0.png', pulse_samples[0])
                # plot_fft('fftplot.png', pulse_samples[0], prog.txrate)

            # This is all the data required for a pulse.
            pulse_transmit_data['startofburst'] = startofburst
            pulse_transmit_data['endofburst'] = endofburst
            pulse_transmit_data['pulse_antennas'] = pulse_antennas
            pulse_transmit_data['samples_array'] = pulse_samples
            pulse_transmit_data['timing'] = timing
            pulse_transmit_data['isarepeat'] = repeat

            # Add pulse dictionary pulse_transmit_data at last place in sequence list
            sequence_list.append(pulse_transmit_data)

        return sequence_list
