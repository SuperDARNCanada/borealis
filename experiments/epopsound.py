"""
Copyright SuperDARN Canada 2020

Keith Kotyk

This experiment depends on a complementary passes file. The file should have name
{radar}.epop.passes name and located under the directory stored in the BOREALISSCHEDULEPATH env
variable. Lines in the file follow the structure:

utctimestampfromepoch beam marker_period freq(khz)

The closest upcoming timestamp is used, so make sure this mode begins running before the required
line.
"""

import copy
import datetime
import math
import os
import sys


BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf

EPOP_PASS_FILE = os.environ['BOREALISSCHEDULEPATH'] + "/{}.epop.passes"


class Epopsound(ExperimentPrototype):
    """
    Experiment for conjunction with EPOP RRI. 
    This mode creates a transmission that is received
    by RRI. 

    Up to 4 frequencies can be used, and given a certain
    beam range the beams will be cycled through at the 
    frequency using 8 pulse sequence, followed by one 
    integration time of a 7 pulse sequence at the frequency
    before moving on to the next frequency. 
    """

    def __init__(self, **kwargs):
        cpid = 3371
        epop_file = EPOP_PASS_FILE.format(scf.opts.site_id)

        # default values
        freqs = [scf.COMMON_MODE_FREQ_1]
        startbeam = stopbeam = 7
        marker_period = 0
        
        if kwargs:
            if 'freq1' in kwargs.keys():
                freqs = [int(kwargs['freq1'])]
                if 'freq2' in kwargs.keys():
                    freqs.append(int(kwargs['freq2']))
                    if 'freq3' in kwargs.keys():
                        freqs.append(int(kwargs['freq3']))
                        if 'freq4' in kwargs.keys():
                            freqs.append(int(kwargs['freq4']))
            if 'startbeam' in kwargs.keys():
                startbeam = int(kwargs['startbeam'])
            if 'stopbeam' in kwargs.keys():
                stopbeam = int(kwargs['stopbeam'])
            if 'marker_period' in kwargs.keys():
                marker_period = int(kwargs['marker_period'])

        self.printing('Freqs (kHz): {}, Start Beam: {}, Stop Beam: {}, '
                      'Marker Period: {}, '
                .format(freqs, startbeam, stopbeam, marker_period))

        centre_freq = int(sum(freqs)/len(freqs))

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        basic_beams = list(range(startbeam, stopbeam + 1))
        if marker_period > 0:
            lotta_beams = basic_beams * (math.ceil(marker_period/len(basic_beams)) + 1)
            beams_to_use = lotta_beams[0:marker_period]
            marker_beam_to_use = [lotta_beams[marker_period]]
        else:
            beams_to_use = basic_beams

        slices = []
        base_slice = {
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 1000, #ms
            "scanbound": [1.0 * i for i in range(len(beams_to_use))],
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "acf": True,
            "xcf": True,
            "acfint": True
        }
        for freq in freqs:
            # for each freq add 
            base_slice.update({
                "txfreq": freq
                })
            slices.append(slice_0)

        super(Epopsound, self).__init__(cpid=cpid, txctrfreq=centre_freq, rxctrfreq=centre_freq,
                                        comment_string=Epopsound.__doc__)

        self.add_slice(slices[0])
        if len(slices) > 1:
            for a_slice in slices[1:]:
                self.add_slice(a_slice, interfacing_dict={0: 'INTTIME'})

        if marker_period > 0:
            # get the marker slice
            slice_1 = copy.deepcopy(base_slice)
            slice_1.update({
                "pulse_sequence": scf.SEQUENCE_7P,
                "tau_spacing": scf.TAU_SPACING_7P,
                "beam_order": marker_beam_to_use,
                "txfreq": freqs[0]
                })
            self.add_slice(slice_1, interfacing_dict={0: 'SCAN'})