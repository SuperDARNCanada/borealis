#!/usr/bin/python

# Copyright SuperDARN Canada 2022
# A simultaneous multifrequency widebeam mode.
# This experiment uses two pairs of adjacent antennas for transmitting,
# with each pair operating on its own frequency.
# The mode has zero phase (no beams) and receives on all antennas.
# There is no scan boundary, it simply sounds for 3.5 seconds at a time.
# It does not generate correlations and only produces antennas_iq data.

# Requested by Dr. Pasha Ponomarenko April 2022
import copy
import sys
import os

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


class MultifreqWidebeam(ExperimentPrototype):

    def __init__(self, **kwargs):
        """
        kwargs:

        freq1: int, first transmit frequency - kHz
        freq2: int, second transmit frequency - kHz

        """
        cpid = 3712

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        tx_freq_1 = scf.COMMON_MODE_FREQ_1
        tx_freq_2 = scf.COMMON_MODE_FREQ_2

        if kwargs:
            if 'freq1' in kwargs.keys():
                tx_freq_1 = int(kwargs['freq1'])

            if 'freq2' in kwargs.keys():
                tx_freq_2 = int(kwargs['freq2'])

        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": [0],
            "rx_beam_order": [0],
            "tx_beam_order": [0],
            "freq": tx_freq_1,  # kHz
            "tx_antennas": [6, 7],  # Using two tx antennas from the middle of array
            "align_sequences": True,
            "scanbound": [i * scf.INTT_7P * 1e-3 for i in range(len(scf.STD_16_BEAM_ANGLE))],
        }

        slice_2 = copy.deepcopy(slice_1)    # slice_id = 1, the second slice
        slice_2['freq'] = tx_freq_2
        slice_2['tx_antennas'] = [8, 9]     # Use separate pair of antennas near middle of array

        list_of_slices = [slice_1, slice_2]
        sum_of_freq = 0
        for slice in list_of_slices:
            sum_of_freq += slice['freq']  # kHz, oscillator mixer frequency on the USRP for TX
        rxctrfreq = txctrfreq = int(sum_of_freq / len(list_of_slices))

        super().__init__(cpid, txctrfreq=txctrfreq, rxctrfreq=rxctrfreq,
                         comment_string='Simultaneous multifrequency widebeam')

        self.add_slice(slice_1)

        self.add_slice(slice_2, interfacing_dict={0: 'CONCURRENT'})
