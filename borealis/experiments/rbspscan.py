#!/usr/bin/python3

#Copyright SuperDARN Canada 2019

import os
import sys

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class RBSPScan(ExperimentPrototype):
    """rbspscan was first run in 2012 to support the Van Allen probes satellite mission
    (initially called the RBSP mission or Radiation Belt Storm Probes). It had a trigger mode
    where the DST (Disturbance Storm Time) index would be checked every 30 minutes, and if it
    went below a threshold, then rbspscan would trigger for at least 30 minutes. This was
    implemented via a python script that would update the schedule file on the CDN radars. The
    scheduler for qnx4 was upgraded to allow this to happen. Both duration and priority of a
    particularly scheduled radar control program were capabilities added to the scheduler, so
    that a PI group could prioritize any discretionary time modes over rbspscan if they wanted to



    Tim Yeoman - Fri Sept 21st, 2012
    The proposed RBSP mode (in the first instance) is a follows:

    CT-TRIG
    An interleaved full scan and mini scan, giving 2 min full scan data for
    convection, as was done in the old common time. So it is essentially the
    Themis mode but with 3 camp beams, Standard lagfr and smsep, a 3 s dwell and
    2 min scan boundaries.

    So a mono beam pattern would go as follows for a forward scanning radar,
    where n is the meridional beam:

    0,n-1,1, n,2, n+2,3,n-1,4,n,5,n+2,6,n-1,7,n,8,n+2,9, ...

    A first suggestion for the mini-scan beam choices is attached, although
    individual PIs can change this if, for example, there is a key piece of
    other instrumentation in their f-o-v which they would like to cover.

    In the first instance, CT-TRIG and ST-APOG can be the same. The idea is
    that ST-APOG can have its lag to first range and range gate size adjusted to
    best match any apogee passes.
    """
    def __init__(self,):
        cpid = 200

        forward_beams = [0, "westbm", 1, "meridonalbm", 2, "eastbm", 3, "westbm", 4, "meridonalbm",
                         5, "eastbm", 6, "westbm", 7, "meridonalbm", 8, "eastbm", 9, "westbm", 10,
                         "meridonalbm", 11, "eastbm", 12, "westbm", 13, "meridonalbm", 14, "eastbm",
                         15]
        reverse_beams = [15, "eastbm", 14, "meridonalbm", 13, "westbm", 12, "westbm", 11,
                        "meridonalbm", 10, "westbm", 9, "eastbm", 8, "meridonalbm", 7, "westbm", 6,
                        "eastbm", 5, "meridonalbm", 4, "westbm", 3, "eastbm", 2, "meridonalbm", 1,
                        "westbm", 0]

        if scf.IS_FORWARD_RADAR:
            beams_to_use = forward_beams
        else:
            beams_to_use = reverse_beams

        if scf.opts.site_id in ["sas"]:
            westbm = 2
            meridonalbm = 3
            eastbm = 5
        if scf.opts.site_id in ["pgr"]:
            westbm = 12
            meridonalbm = 13
            eastbm = 15
        if scf.opts.site_id in ["inv", "rkn", "cly"]:
            westbm = 6
            meridonalbm = 7
            eastbm = 9

        if scf.opts.site_id in ["sas", "pgr", "cly"]:
            freq = 10500
        if scf.opts.site_id in ["rkn"]:
            freq = 12200
        if scf.opts.site_id in ["inv"]:
            freq = 12100

        beams_to_use = [westbm if bm == "westbm" else bm for bm in beams_to_use]
        beams_to_use = [meridonalbm if bm == "meridonalbm" else bm for bm in beams_to_use]
        beams_to_use = [eastbm if bm == "eastbm" else bm for bm in beams_to_use]

        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            "freq" : freq, #kHz
            "scanbound" : scf.easy_scanbound(scf.INTT_7P, beams_to_use), #2 min scanbound
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }
        super(RBSPScan, self).__init__(cpid)

        self.add_slice(slice_1)
