#!/usr/bin/python

""" Scans are made up of AveragingPeriods, these are typically a 3sec time of 
the same pulse sequence pointing in one direction.  AveragingPeriods are made 
up of Sequences, typically the same sequence run ave. 21 times after a clear 
frequency search.  Sequences are made up of pulse_time lists, which give 
timing, CPObject, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype. 
"""

import sys
from operator import itemgetter
from scan_class_base import ScanClassBase

class Sequence(ScanClassBase):

    def __init__(self, seqn_keys, sequence_slice_dict, sequence_interface, options):
        """
        Set up the sequence class. This includes setting up the pulses. self.pulses is a list of
        dictionaries where one dictionary = one pulse. The dictionary keys are : isarepeat, 
        pulse_timing_us, slice_id, slice_pulse_index, pulse_len, intra_pulse_start_time, 
        combined_pulse_index, pulse_shift, iscombined, combine_total, and combine_index.
        :param seqn_keys
        :param sequence_slice_dict
        :param sequence_interface
        :param options
        """

        # TODO describe pulse dictionary in docs
        ScanClassBase.__init__(self, seqn_keys, sequence_slice_dict, sequence_interface,
                               options)

        # TODO: pass clear frequencies to pass to pulses

        # All interfacing at this point is PULSE.

        # TODO: use number of frequencies to determine power output of each frequency (1/2, 1/3)

        pulses = []
        # Getting a list of pulses, where pulse is a dictionary
        for slice_id in seqn_keys:
            for slice_pulse_index, pulse_time in enumerate(self.slice_dict[slice_id]['pulse_sequence']):
                pulse_timing_us = pulse_time*self.slice_dict[slice_id]['mpinc']
                pulses.append({'pulse_timing_us': pulse_timing_us, 'slice_id': slice_id,
                                   'slice_pulse_index': slice_pulse_index,
                                   'pulse_len': self.slice_dict[slice_id]['pulse_len'],
                                   'pulse_shift': self.slice_dict[slice_id]['pulse_shift'][slice_pulse_index]})

        self.pulses = sorted(pulses, key=itemgetter('pulse_timing_us', 'slice_id'))
        # Will sort by timing first and then by cpo if timing =. This is all pulses in the sequence,
        # in a list of dictionaries.

        print self.pulses  # check the sort

        # Set up the combined pulse list
        this_pulse_index = 0
        combined_pulse_index = 0
        total_combined_pulses = 0
        while this_pulse_index < len(self.pulses):
            pulse = self.pulses[this_pulse_index]
            pulse['isarepeat'] = False  # Will change later if True
            pulse['intra_pulse_start_time'] = 0  # Will change later if combined with another pulse
            pulse['combined_pulse_index'] = combined_pulse_index # this is the index corresponding with actual number of pulses that will be sent to driver, after combinations are completed.
            # Pulse is just a dictionary of keys isarepeat, pulse_timing_us, slice_id, and
            # slice_pulse_index, pulse_len, intra_pulse_start_time

            # Determine if we are combining samples based on timing of pulses
            # If we are combining pulses, set keys iscombined, combinetotal, combineindex
            combine_pulses = True
            next_pulse_index = this_pulse_index + 1

            if this_pulse_index == len(self.pulses) - 1:  # last pulse is not combined - end
                combine_pulses = False
                pulse['iscombined'] = False
                total_combined_pulses = pulse['combined_pulse_index'] + 1

            while combine_pulses:
                if self.pulses[next_pulse_index]['pulse_timing_us'] <= pulse['pulse_timing_us']\
                         + pulse['pulse_len'] + self.options.minimum_pulse_separation:
                    # combine pulse and next_pulse
                    next_pulse = self.pulses[next_pulse_index]

                    if 'iscombined' in pulse.keys():  # already combined with a previous pulse
                        for index in range(this_pulse_index,next_pulse_index):
                            self.pulses[index]['combine_total'] += 1
                    else:
                        pulse['iscombined'] = True
                        pulse['combine_total'] = 2  # 2 pulses are combined (so far), pulse and next_pulse
                        pulse['combine_index'] = 0

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

            this_pulse_index = next_pulse_index
            combined_pulse_index += 1
            # Jump ahead depending how many pulses we've combined.
            # Combined pulse list is a list of lists of pulses, 
            #   combined as to how they are sent as samples to 
            #   driver.

        self.total_combined_pulses = total_combined_pulses

        # Find repeats
        for pulse_index in range(1, len(self.pulses)):
            pulse = self.pulses[pulse_index]
            if pulse['iscombined']:
                if pulse['combine_index'] != 0:
                    # this pulse will be a repeat if the pulse with combine_index of 0 is a repeat.
                    first_pulse_in_combination_index = pulse_index - pulse['combine_index']
                    pulse['isarepeat'] = self.pulses[first_pulse_in_combination_index]['isarepeat']
                elif not self.pulses[pulse_index -1]['iscombined']:
                    # pulse['combine_index'] = 0, lastpulse iscombined = False.
                    pulse['isarepeat'] = False  # the last pulse must be combined in some way as well.
                elif pulse['combine_total'] != self.pulses[pulse_index -1]['combine_total']:
                    pulse['isarepeat'] = False  # must have same number of slices combined.
                else:
                    last_pulse_index = pulse_index -1
                    last_pulse = self.pulses[last_pulse_index]
                    # get subset for only this combined pulse
                    combined_pulse_1 = []
                    combined_pulse_2 = []
                    for a_pulse in self.pulses:
                        if a_pulse['combined_pulse_index'] == last_pulse['combined_pulse_index']:
                            combined_pulse_1.append(a_pulse)

                    for b_pulse in self.pulses:
                        if b_pulse['combined_pulse_index'] == pulse['combined_pulse_index']:
                            combined_pulse_2.append(b_pulse)

                    for pulse_1, pulse_2 in zip(combined_pulse_1, combined_pulse_2):
                        # combine_index should be the same because they were in order.
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

        last_pulse = self.pulses[-1]
        this_pulse_len = 0
        if last_pulse['iscombined']:
            for pulse_ind in range(-(last_pulse['combine_total']), 0):
                this_pulse = self.pulses[pulse_ind]
                if (this_pulse['pulse_len'] + this_pulse['intra_pulse_start_time']) > this_pulse_len:
                    this_pulse_len = this_pulse['pulse_len'] + this_pulse['intra_pulse_start_time']
        else:
            this_pulse_len = last_pulse['pulse_len']

        self.last_pulse_len = this_pulse_len


        # FIND the max scope sync time
        # 19 is the sample delay below; how do we calculate this? # REVIEW #6 TODO find out the reason for these magic numbers
        self.ssdelay = 0  # ssdelay is time required to wait after last pulse.
        for slice_id in seqn_keys:
            newdelay = (self.slice_dict[slice_id]['nrang'] + 19 + 10) * self.slice_dict[slice_id]['pulse_len']
            if newdelay > self.ssdelay:
                self.ssdelay = newdelay
        # The delay is long enough for any slice's pulse length and nrang to be accounted for.

        # FIND the sequence time. Time before the first pulse is 70 us when RX and TR set up for the first pulse. The
        # timing to the last pulse is added, as well as its pulse length and the RX/TR delay at the end of last pulse.
        self.seqtime = self.options.atten_window_time_start + 2*self.options.tr_window_time + \
                       self.options.atten_window_time_end + self.pulses[-1]['pulse_timing_us'] + \
                       self.last_pulse_len

        # FIND the total scope sync time and number of samples to receive.
        self.sstime = self.seqtime + self.ssdelay
        self.numberofreceivesamples = int(self.options.rx_sample_rate * self.sstime * 1e-6)
