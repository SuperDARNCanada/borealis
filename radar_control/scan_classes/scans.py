#!/usr/bin/python

""" Scans are made up of AveragingPeriods, these are typically a 3sec time of
the same pulse sequence pointing in one direction.  AveragingPeriods are made
up of Sequences, typically the same sequence run ave. 21 times after a clear
frequency search.  Sequences are made up of pulse_time lists, which give
timing, CPObject, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype.
"""

import sys
import averaging_periods


class Scan:
    """ 
    Made up of AveragingPeriods at defined beam directions, and some other metadata for the scan itself.
    """

    def __init__(self, experiment, scan_keys):  # REVIEW #22/#41 Lots of duplicate code in init fn's of the scan
        # classes. Can this be written once in a parent class and then be inherited in children. REPLY OK. TODO
        self.rxrate = int(float(experiment.options.rx_sample_rate))  # in Hz
        self.keys = scan_keys
        self.slice_dict = {}
        for slice_id in scan_keys:
            self.slice_dict[slice_id] = experiment.slice_list[slice_id]

        # Create smaller interfacing dictionary for this scan to pass to the averaging periods.
        # This dictionary will only include the slices in this scan, therefore it will not include any SCAN interfacing.
        interface_keys = []
        for m in range(len(scan_keys)):
            for n in range(m + 1, len(scan_keys)):
                interface_keys.append([scan_keys[m], scan_keys[n]])
        self.interface = {}
        for p, q in interface_keys:
            self.interface[p, q] = experiment.interface[p, q]

        # scan metadata - must be the same between all slices combined in scan.  Metadata includes:
        self.scanboundf = self.slice_dict[self.keys[0]]['scanboundflag']
        for slice_id in self.keys:
            if self.slice_dict[slice_id]['scanboundflag'] != self.scanboundf:
                errmsg = """Scan Boundary Flag not the Same Between Slices {} and {} combined in Scan"""\
                    .format(self.keys[0], slice_id)
                sys.exit(errmsg)
        if self.scanboundf == 1:
            self.scanbound = self.slice_dict[self.keys[0]]['scanbound']
            for slice_id in self.keys:
                if self.slice_dict[slice_id]['scanbound'] != self.scanbound:
                    errmsg = """Scan Boundary not the Same Between Slices {} and {}
                         combined in Scan""".format(self.keys[0], slice_id)
                    sys.exit(errmsg)

        # NOTE: for now we assume that when INTTIME combined, the AveragingPeriods of the various slices in the scan are
        #   just interleaved 1 then the other.

        # Create a dictionary of beam directions for slice_id #
        self.beamdir = {}
        self.scan_beams = {}
        for slice_id in self.keys:
            self.beamdir[slice_id] = self.slice_dict[slice_id]['beamdir']
            self.scan_beams[slice_id] = self.slice_dict[slice_id]['scan']
            # REVIEW #26 Names are inconsistent, also difficult to tell what 'scan' is - is it the number? Is it the
            # order you're scanning on? indexed from 0? is beamdir the azimuth? in degrees?
            # REPLY: the slice['scan'] is the list of beamnums in order of desired for the scan. One element = one
            # inttime. One element could have multiple beamnums. Beamdir is the beam directions, where the direction
            # (off azimuth in degrees) = beamdir[beamnum] - so the third direction in the scan would be beamdir[scan[2]]
            # So what name should I change here

        # Determine how many averaging periods to make by separating but the INTTIME mixed.
        self.slice_id_inttime_lists = self.get_inttimes()

        # However we need to make sure the number of inttimes (as determined by length of slice['scan'] is the same
        # for slices combined in the averaging period.
        for inttime_list in self.slice_id_inttime_lists:
            for slice_id in inttime_list:
                if len(self.scan_beams[slice_id]) != len(self.scan_beams[inttime_list[0]]):
                    errmsg = """CPO {} and {} are mixed within the AveragingPeriod but do not have the same number of
                        AveragingPeriods in their scan""".format(self.keys[0], slice_id)
                    sys.exit(errmsg)
                    # REVIEW #6 need a todo for error handling this.

        if self.slice_id_inttime_lists:  # if list is not empty, can make aveperiods
            self.aveperiods = [averaging_periods.AveragingPeriod(self, inttime_list) for inttime_list in
                               self.slice_id_inttime_lists]
            # Each component is an inttime, we should create AveragingPeriods from those slice_ids.

            # order of the Averaging Periods - will be in slice_id # order.
            # self.aveperiods=sorted(self.aveperiods, key=operator.attrgetter('timing'))

    def get_inttimes(self):
        intt_combos = []

        for num1, num2 in self.interface.keys():
            if (self.interface[num1, num2] == "PULSE" or
                        self.interface[num1, num2] == "INTEGRATION"):
                intt_combos.append([num1, num2])
        # Save only the keys that are combinations within inttime.

        intt_combos = sorted(intt_combos)
        # if [2,4] and [1,4], then also must be [1,2] in the combos list
        i = 0
        while i < len(intt_combos):
            # REVIEW #22 This is duplicate code from experiment prototype. Can make a new fn or library for this?
            # REPLY OK TODO
            k = 0
            while k < len(intt_combos[i]):
                j = i + 1
                while j < len(intt_combos):
                    if intt_combos[i][k] == intt_combos[j][0]:
                        add_n = intt_combos[j][1]
                        intt_combos[i].append(add_n)
                        # Combine the indices if there are 3+ CPObjects
                        #   combining in same seq.
                        for m in range(0, len(intt_combos[i]) - 1):
                            # Try all values in seq_combos[i] except the
                            #   last value, which is = to add_n.
                            try:
                                intt_combos.remove([intt_combos[i][m], add_n])
                                # seq_combos[j][1] is the known last
                                #   value in seq_combos[i]
                            except ValueError:
                                errmsg = 'Interfacing not Valid: CPO %d and CPO \
                                    %d are combined in-scan and do not \
                                    interface the same with CPO %d' % (
                                    intt_combos[i][m], intt_combos[i][k],
                                    add_n)
                                sys.exit(errmsg)
                        j = j - 1
                        # This means that the former scan_combos[j] has
                        #   been deleted and there are new values at
                        #   index j, so decrement before
                        #   incrementing.
                    j = j + 1
                k = k + 1
            i = i + 1
        # Now scan_combos is a list of lists, where a cpobject occurs
        #   only once in the nested list.
        for i in range(len(self.keys)):
            found = False
            for k in range(len(intt_combos)):
                for j in range(len(intt_combos[k])):
                    if self.keys[i] == intt_combos[k][j]:
                        found = True
                        break
                if found == False:
                    continue
                break
            else:  # no break
                intt_combos.append([self.keys[i]])
                # Append the cpo on its own, is not scan combined.
        intt_combos = sorted(intt_combos)
        return intt_combos
