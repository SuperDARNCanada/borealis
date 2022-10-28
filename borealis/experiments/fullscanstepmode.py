#!/usr/bin/python

# write an experiment that creates a new control program.
import os
import sys

from experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class FullScanStepMode(ExperimentPrototype):
    """
    Scan beam by beam while stepping in frequency within a band. If a step is within a restricted
    band, it will be moved to the closest available unrestricted frequency. An averaging period of
    each frequency is run on each beam before switching to the next beam.
    """

    def __init__(self):
        cpid = 3561

        top = 15000
        bottom = 10000
        step = 500
        all_steps = list(range(bottom, top, step))

        def move_freqs(direction):
            """
            Move frequency steps that lay in restricted bands into unrestricted bands. A 25 kHz
            buffer is used.

            :param      direction:  The direction of which to try move the frequency. 'up' or 'down'
            :type       direction:  str
            """

            for i in range(len(all_steps)):
                for restrict in scf.opts.restricted_ranges:
                    if all_steps[i] > (restrict[0] - 25) and all_steps[i] < (restrict[1] + 25):
                        moved = False
                        while not moved:
                            if direction == "up":
                                if all_steps[i] + 25 > top:
                                    break
                                elif not((all_steps[i] + 25) > (restrict[0] - 25) and (all_steps[i] + 25) < (restrict[1] + 25)):
                                    moved = True
                                    all_steps[i] += 25
                                else:
                                    all_steps[i] += 25

                            if direction == "down":
                                if all_steps[i] - 25 < bottom:
                                    break
                                elif not((all_steps[i] - 25) > (restrict[0] - 25) and (all_steps[i] - 25) < (restrict[1] + 25)):
                                    moved = True
                                    all_steps[i] -= 25
                                else:
                                    all_steps[i] -= 25

        # First try to move all restricted steps up. If not, then try move them down.
        move_freqs("up")
        move_freqs("down")
        all_steps = sorted(list(set(all_steps)))
        
        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        slices = []
        for step in all_steps:
            s = {
                "pulse_sequence": scf.SEQUENCE_7P,
                "tau_spacing": scf.TAU_SPACING_7P,
                "pulse_len": scf.PULSE_LEN_45KM,
                "num_ranges": num_ranges,
                "first_range": scf.STD_FIRST_RANGE,
                "intt": 3500,  # duration of an integration, in ms
                "beam_angle": scf.STD_16_BEAM_ANGLE,
                "rx_beam_order": beams_to_use,
                "tx_beam_order": beams_to_use,
                "scanbound" : [i * (3.5 * len(all_steps)) for i in range(len(beams_to_use))],
                "freq" : step, #kHz
                "acf": True,
                "xcf": True,  # cross-correlation processing
                "acfint" : True,  # interferometer acfs
                "comment" : FullScanStepMode.__doc__
            }
            slices.append(s)


        rxctrfreq = txctrfreq = sum(all_steps)/len(all_steps)


        super(FullScanStepMode, self).__init__(cpid, txctrfreq=txctrfreq, rxctrfreq=rxctrfreq,
                comment_string=FullScanStepMode.__doc__)


        self.add_slice(slices[0])
        interfacing_dict = {}
        for i in range(1, len(slices)):
            interfacing_dict[i-1] = 'AVEPERIOD'
            self.add_slice(slices[i], interfacing_dict=interfacing_dict)
