#!/usr/bin/python

# write an experiment that creates a new control program.
import os
import sys
import copy

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class TwoMultifsound(ExperimentPrototype):

    def __init__(self):
        cpid = 3570

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        freqs = (scf.COMMON_MODE_FREQ_1, scf.COMMON_MODE_FREQ_2)

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            "freq" : freqs[0], #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }

        slice_2 = copy.deepcopy(slice_1)
        slice_2['freq'] = freqs[1]

        list_of_slices = [slice_1, slice_2]
        sum_of_freq = 0
        for slice in list_of_slices:
            sum_of_freq += slice['freq']# kHz, oscillator mixer frequency on the USRP for TX
        rxctrfreq = txctrfreq = int(sum_of_freq/len(list_of_slices))


        super(TwoMultifsound, self).__init__(cpid, txctrfreq=txctrfreq, rxctrfreq=rxctrfreq,
                comment_string='Twofsound simultaneous in-sequence')

        self.add_slice(slice_1)

        self.add_slice(slice_2, interfacing_dict={0: 'CONCURRENT'})

