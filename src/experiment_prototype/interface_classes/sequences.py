#!/usr/bin/python

"""
    sequences
    ~~~~~~~~~
    This is the module containing the Sequence class. The Sequence class contains the InterfaceClassBase
    members, as well as a list of pulse dictionaries, the total_combined_pulses in the sequence,
    power_divider, last_pulse_len, ssdelay, seqtime, which together give sstime (scope synce time,
    or time for receiving, and numberofreceivesamples to sample during the receiving window
    (calculated using the receive sampling rate).

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
# built-in
import collections
import copy
from functools import reduce
import inspect
import math
from pathlib import Path

# third-party
import numpy as np
import structlog

# local
from experiment_prototype.interface_classes.interface_class_base import (
    InterfaceClassBase,
)
from experiment_prototype.experiment_exception import ExperimentException
from utils.signals import get_samples, get_phase_shift

# Obtain the module name that imported this log_config
caller = Path(inspect.stack()[-1].filename)
module_name = caller.name.split(".")[0]
log = structlog.getLogger(module_name)


class Sequence(InterfaceClassBase):
    """
    Set up the sequence class.

    **The members of the sequence are:**

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
    first_rx_sample_start
        The location of the first sample for the RX data from the start of the TX data (in number
        of samples, unitless). This will be calculated as the center sample of the first
        occurring pulse (uncombined).
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

        - start_time_us - start time of the pulse in us, relative to the first pulse in sqn.
        - total_pulse_len - total length of the pulse that includes len of all combined pulses.
        - pulse_sample_start - The tx sample number at which the pulse starts.
        - tr_window_num_samps - The number of tx samples of the tr window.
        - component_info - a list of all the pre-combined pulse components (incl their length and
          start time) that are in the combined pulseAlso in us.
        - pulse_transmit_data - dictionary hold the transmit metadata that will be sent to driver.

    output_encodings
        This dict will hold a list of all the encodings used during an aveperiod for each slice.
        These will be used for data write later.


    :param  seqn_keys:              slice_ids that need to be included in this sequence.
    :type   seqn_keys:              list
    :param  sequence_slice_dict:    the slice dictionary that explains the parameters of each slice
                                    that is included in this sequence. Keys are the slice_ids
                                    included and values are dictionaries including all necessary
                                    slice parameters as keys.
    :type   sequence_slice_dict:    dict
    :param  sequence_interface:     the interfacing dictionary that describes how to interface the
                                    slices that are included in this sequence. Keys are tuples of
                                    format (slice_id_1, slice_id_2) and values are of
                                    interface_types set up in experiment_prototype.
    :type   sequence_interface:     dict
    :param  transmit_metadata:      metadata from the config file that is useful here.
    :type   transmit_metadata:      dict
    """

    def __init__(
        self, seqn_keys, sequence_slice_dict, sequence_interface, transmit_metadata
    ):
        InterfaceClassBase.__init__(
            self, seqn_keys, sequence_slice_dict, sequence_interface, transmit_metadata
        )

        self.decimation_scheme = self.slice_dict[self.slice_ids[0]].decimation_scheme
        for slice_id in self.slice_ids:
            if self.slice_dict[slice_id].decimation_scheme != self.decimation_scheme:
                errmsg = (
                    f"Slices {self.slice_ids[0]} and {slice_id} are CONCURRENT "
                    f"interfaced and do not have the same decimation scheme"
                )
                raise ExperimentException(errmsg)

        dm_rate = 1
        for stage in self.decimation_scheme.stages:
            dm_rate *= stage.dm_rate

        self.output_rx_rate = self.transmit_metadata["rxrate"] / dm_rate
        txrate = self.transmit_metadata["txrate"]
        main_antenna_count = self.transmit_metadata["main_antenna_count"]
        main_antenna_spacing = self.transmit_metadata["main_antenna_spacing"]
        intf_antenna_count = self.transmit_metadata["intf_antenna_count"]
        intf_antenna_spacing = self.transmit_metadata["intf_antenna_spacing"]
        self.tx_main_antennas = self.transmit_metadata["tx_main_antennas"]
        self.rx_main_antennas = self.transmit_metadata["rx_main_antennas"]
        self.rx_intf_antennas = self.transmit_metadata["rx_intf_antennas"]
        pulse_ramp_time = self.transmit_metadata["pulse_ramp_time"]
        max_usrp_dac_amplitude = self.transmit_metadata["max_usrp_dac_amplitude"]
        tr_window_time = self.transmit_metadata["tr_window_time"]
        intf_offset = self.transmit_metadata["intf_offset"]

        self.basic_slice_pulses = {}
        self.rx_beam_phases = {}
        self.tx_main_phase_shifts = {}
        self.rx_main_antenna_indices = {}
        self.rx_intf_antenna_indices = {}
        self.tx_antenna_indices = {}
        self.txctrfreq = self.slice_dict[self.slice_ids[0]].txctrfreq
        self.rxctrfreq = self.slice_dict[self.slice_ids[0]].rxctrfreq
        single_pulse_timing = []

        # For each slice calculate beamformed samples and place into the basic_slice_pulses
        # dictionary. Also populate the pulse timing metadata and place into single_pulse_timing
        for slice_id in self.slice_ids:
            exp_slice = self.slice_dict[slice_id]
            freq_khz = float(exp_slice.freq)
            wave_freq = freq_khz - self.txctrfreq
            wave_freq_hz = wave_freq * 1000

            # Now we set up the phases for receive side
            if exp_slice.rx_antenna_pattern is not None:
                # Returns an array of size [beam_angle] of complex numbers of magnitude <= 1
                rx_main_phase_shift = exp_slice.rx_antenna_pattern(
                    exp_slice.beam_angle,
                    freq_khz,
                    main_antenna_count,
                    main_antenna_spacing,
                )
                rx_intf_phase_shift = exp_slice.rx_antenna_pattern(
                    exp_slice.beam_angle,
                    freq_khz,
                    intf_antenna_count,
                    intf_antenna_spacing,
                    intf_offset[0],
                )
            else:
                rx_main_phase_shift = get_phase_shift(
                    exp_slice.beam_angle,
                    freq_khz,
                    main_antenna_count,
                    main_antenna_spacing,
                )
                rx_intf_phase_shift = get_phase_shift(
                    exp_slice.beam_angle,
                    freq_khz,
                    intf_antenna_count,
                    intf_antenna_spacing,
                    intf_offset[0],
                )

            # The antenna indices for receiving by this slice
            slice_rx_main_antennas = exp_slice.rx_main_antennas
            slice_rx_intf_antennas = exp_slice.rx_intf_antennas

            # The index of the antennas for this slice, within the list of all antennas from the config file
            main_indices = [
                self.rx_main_antennas.index(ant) for ant in slice_rx_main_antennas
            ]
            intf_indices = [
                self.rx_intf_antennas.index(ant) for ant in slice_rx_intf_antennas
            ]
            self.rx_main_antenna_indices[slice_id] = main_indices
            self.rx_intf_antenna_indices[slice_id] = intf_indices

            # Zero out the complex phase for any antenna that isn't used in this slice
            main_phases = np.zeros(
                (rx_main_phase_shift.shape[0], len(self.rx_main_antennas)),
                dtype=rx_main_phase_shift.dtype,
            )
            intf_phases = np.zeros(
                (rx_intf_phase_shift.shape[0], len(self.rx_intf_antennas)),
                dtype=rx_intf_phase_shift.dtype,
            )
            main_phases[:, main_indices] = rx_main_phase_shift[
                :, slice_rx_main_antennas
            ]
            intf_phases[:, intf_indices] = rx_intf_phase_shift[
                :, slice_rx_intf_antennas
            ]

            self.rx_beam_phases[slice_id] = {"main": main_phases, "intf": intf_phases}

            # Set up the tx pulses if transmitting
            if not exp_slice.rxonly:
                basic_samples = get_samples(
                    txrate,
                    wave_freq_hz,
                    float(exp_slice.pulse_len) / 1e6,
                    pulse_ramp_time,
                    1.0,
                )

                if exp_slice.tx_antenna_pattern is not None:
                    # Returns an array of size [tx_antennas] of complex numbers of magnitude <= 1
                    tx_main_phase_shift = exp_slice.tx_antenna_pattern(
                        freq_khz, exp_slice.tx_antennas, main_antenna_spacing
                    )
                else:
                    tx_main_phase_shift = get_phase_shift(
                        exp_slice.beam_angle,
                        freq_khz,
                        main_antenna_count,
                        main_antenna_spacing,
                    )

                # The antennas used for transmitting this slice
                slice_tx_antennas = exp_slice.tx_antennas

                # The index of the antennas for this slice, within the list of all antennas from the config file
                tx_indices = [
                    self.tx_main_antennas.index(ant) for ant in slice_tx_antennas
                ]
                self.tx_antenna_indices[slice_id] = tx_indices

                # Zero out the complex phase of any antenna that isn't used in this slice
                tx_phases = np.zeros(
                    (tx_main_phase_shift.shape[0], len(self.tx_main_antennas)),
                    dtype=tx_main_phase_shift.dtype,
                )
                tx_phases[:, tx_indices] = tx_main_phase_shift[:, slice_tx_antennas]

                # tx_phases:        [num_beams, num_antennas]
                # basic_samples:    [num_samples]
                # phased_samps_for_beams: [num_beams, num_antennas, num_samples]
                log.verbose(
                    "slice information",
                    slice_id=slice_id,
                    tx_main_phases=tx_phases,
                    tx_main_magnitudes=np.abs(tx_phases),
                    tx_main_angles=np.rad2deg(np.angle(tx_phases)),
                )
                phased_samps_for_beams = np.einsum(
                    "ij,k->ijk", tx_phases, basic_samples
                )
                self.basic_slice_pulses[slice_id] = phased_samps_for_beams
            else:
                self.basic_slice_pulses[slice_id] = []
                tx_phases = np.zeros(
                    (rx_main_phase_shift.shape[0], len(self.tx_main_antennas)),
                    dtype=np.complex64,
                )
            self.tx_main_phase_shifts[slice_id] = tx_phases

            for pulse_time in exp_slice.pulse_sequence:
                pulse_timing_us = (
                    pulse_time * exp_slice.tau_spacing + exp_slice.seqoffset
                )
                pulse_sample_start = round((pulse_timing_us * 1e-6) * txrate)
                pulse_num_samps = round((exp_slice.pulse_len * 1e-6) * txrate)

                single_pulse_timing.append(
                    {
                        "start_time_us": pulse_timing_us,
                        "pulse_len_us": exp_slice.pulse_len,
                        "pulse_sample_start": pulse_sample_start,
                        "pulse_num_samps": pulse_num_samps,
                        "slice_id": slice_id,
                    }
                )

        single_pulse_timing = sorted(
            single_pulse_timing, key=lambda d: d["start_time_us"]
        )

        # Combine any pulses closer than the minimum separation time into a single pulse data
        # dictionary and append to the list of all combined pulses, combined_pulses_metadata.
        tr_window_num_samps = round(tr_window_time * txrate)

        def initialize_combined_pulse_dict(pulse_timing_info):
            return {
                "start_time_us": pulse_timing_info["start_time_us"],
                "total_pulse_len": pulse_timing_info["pulse_len_us"],
                "pulse_sample_start": pulse_timing_info["pulse_sample_start"],
                "total_num_samps": pulse_timing_info["pulse_num_samps"],
                "tr_window_num_samps": tr_window_num_samps,
                "component_info": [pulse_timing_info],
            }

        pulse_data = initialize_combined_pulse_dict(single_pulse_timing[0])
        combined_pulses_metadata = []

        # Determine where pulses occur in the sequence. This will be important if there are overlaps
        for pulse_time in single_pulse_timing[1:]:
            pulse_timing_us = pulse_time["start_time_us"]
            pulse_len_us = pulse_time["pulse_len_us"]
            pulse_sample_start = pulse_time["pulse_sample_start"]
            pulse_num_samps = pulse_time["pulse_num_samps"]

            last_timing_us = pulse_data["start_time_us"]
            last_pulse_len_us = pulse_data["total_pulse_len"]
            last_sample_start = pulse_data["pulse_sample_start"]
            last_pulse_num_samps = pulse_data["total_num_samps"]

            # If there are overlaps (two pulses within minimum separation time) then make them into one single pulse
            min_sep = self.transmit_metadata["min_pulse_separation"]
            if pulse_timing_us < last_timing_us + last_pulse_len_us + min_sep:
                # If the current pulse is completely enveloped by the previous pulse,
                # these values won't change or else we are truncating the previous pulse.
                new_pulse_len = max(
                    pulse_timing_us - last_timing_us + pulse_len_us, last_pulse_len_us
                )
                new_pulse_samps = max(
                    pulse_sample_start - last_sample_start + pulse_num_samps,
                    last_pulse_num_samps,
                )

                pulse_data["total_pulse_len"] = new_pulse_len
                pulse_data["total_num_samps"] = new_pulse_samps
                pulse_data["component_info"].append(pulse_time)
            else:  # pulses do not overlap
                combined_pulses_metadata.append(pulse_data)
                pulse_data = initialize_combined_pulse_dict(pulse_time)

        combined_pulses_metadata.append(pulse_data)

        # Store the overlapping antennas between all pairs of slices in this sequence. This will be
        # used to determine the power divider for each slice in the sequence, if any two slices have
        # overlapping pulses and use the same antennas.
        slice_shared_antennas = dict()
        for i in range(len(self.slice_ids)):
            slice_1_id = self.slice_ids[i]
            slice_1_antennas = set(self.slice_dict[slice_1_id].tx_antennas)
            for j in range(i + 1, len(self.slice_ids)):
                slice_2_id = self.slice_ids[j]
                slice_2_antennas = set(self.slice_dict[slice_2_id].tx_antennas)
                slice_shared_antennas[(slice_1_id, slice_2_id)] = (
                    slice_1_antennas.intersection(slice_2_antennas)
                )

        # Dictionary to keep track of which slices share antennas and transmit at the same time
        slice_overlaps = {slice_id: set() for slice_id in self.slice_ids}

        # Now we can figure out the power divider for each slice
        for combined_pulse in combined_pulses_metadata:
            num_pulses = len(combined_pulse["component_info"])
            if num_pulses == 1:
                # Only one pulse here, no need to check for overlap
                continue

            # Look at each possible pair of pulses in this combined pulse
            for i in range(num_pulses):
                pulse_1 = combined_pulse["component_info"][i]
                for j in range(i + 1, num_pulses):
                    pulse_2 = combined_pulse["component_info"][j]
                    if pulse_1["slice_id"] == pulse_2["slice_id"]:
                        # This is possible if pulses overlap like 1 -> 2 -> 1, so that 1 doesn't
                        # overlap with itself but is still combined with itself.
                        continue
                    min_slice_id = min(pulse_1["slice_id"], pulse_2["slice_id"])
                    max_slice_id = max(pulse_1["slice_id"], pulse_2["slice_id"])
                    if len(slice_shared_antennas[(min_slice_id, max_slice_id)]) != 0:
                        # These two pulses share antennas, and are also combined in a pulse.
                        # Now we check if they actually transmit at the same time, or are just
                        # combined because they almost overlap.
                        if (
                            pulse_2["start_time_us"]
                            < pulse_1["start_time_us"] + pulse_1["pulse_len_us"]
                        ):
                            slice_overlaps[pulse_1["slice_id"]].add(pulse_2["slice_id"])
                            slice_overlaps[pulse_2["slice_id"]].add(pulse_1["slice_id"])

        # Get the naive power divider - total number slices which overlap with slice under consideration.
        power_divider = {
            slice_id: len(ids) + 1 for slice_id, ids in slice_overlaps.items()
        }

        # Now we iterate through each slice, and check if the slices it overlaps with overlap with each
        # other. If they don't, we can subtract 1 from the power divider for the slice.
        for ref_slice, overlaps in slice_overlaps.items():
            overlap_list = list(overlaps)
            for i in range(len(overlap_list)):
                for j in range(i + 1, len(overlap_list)):
                    slice_1 = overlap_list[i]
                    slice_2 = overlap_list[j]
                    if slice_2 not in slice_overlaps[slice_1]:
                        # No overlap, so we decrement.
                        power_divider[ref_slice] -= 1

        # Normalize all combined pulses to the max USRP DAC amplitude
        all_antennas = []
        for slice_id in self.slice_ids:
            if not self.slice_dict[slice_id].rxonly:
                self.basic_slice_pulses[slice_id] *= (
                    max_usrp_dac_amplitude / power_divider[slice_id]
                )

                slice_tx_antennas = self.slice_dict[slice_id].tx_antennas
                all_antennas.extend(slice_tx_antennas)

        sequence_antennas = list(set(all_antennas))

        # predetermine some of the transmit metadata.
        num_pulses = len(combined_pulses_metadata)
        for i in range(num_pulses):
            combined_pulses_metadata[i]["pulse_transmit_data"] = {}
            pulse_transmit_data = combined_pulses_metadata[i]["pulse_transmit_data"]

            pulse_transmit_data["startofburst"] = i == 0
            pulse_transmit_data["endofburst"] = i == (num_pulses - 1)

            pulse_transmit_data["pulse_antennas"] = sequence_antennas
            # the samples array is populated as needed during operations
            pulse_transmit_data["samples_array"] = None
            pulse_transmit_data["timing"] = combined_pulses_metadata[i]["start_time_us"]
            # isarepeat is set as needed during operations
            pulse_transmit_data["isarepeat"] = False

        # print out pulse information for logging.
        for i, cpm in enumerate(combined_pulses_metadata):
            # message = f"Pulse {i}: start time(us) {cpm['start_time_us']}  start sample {cpm['pulse_sample_start']}"
            # message += f"          pulse length(us) {cpm['total_pulse_len']}  pulse num samples {cpm['total_num_samps']}"
            log.verbose("pulse information", **cpm)

        self.combined_pulses_metadata = combined_pulses_metadata

        # FIND the max scope sync time
        # The gc214 receiver card in the old system required 19 us for sample delay and another 10
        # us as empirically discovered. in that case delay = (num_ranges + 19 + 10) * pulse_len. Now
        # we will remove those values. In the old design scope sync was used directly to determine
        # how long to sample. Now we will calculate the number of samples to receive
        # (numberofreceivesamples) using scope sync and send that to the driver to sample at a
        # specific rxrate (given by the config).

        # number ranges to the first range for all slice ids

        range_as_samples = lambda x, y: int(math.ceil(x / y))
        num_ranges_to_first_range = {
            slice_id: range_as_samples(
                self.slice_dict[slice_id].first_range,
                self.slice_dict[slice_id].range_sep,
            )
            for slice_id in self.slice_ids
        }

        # time for number of ranges given, in us, taking into account first_range and num_ranges.
        # pulse_len is the amount of time for any range.
        self.ssdelay = max(
            [
                (
                    self.slice_dict[slice_id].num_ranges
                    + num_ranges_to_first_range[slice_id]
                )
                * self.slice_dict[slice_id].pulse_len
                for slice_id in self.slice_ids
            ]
        )

        # The delay is long enough for any slice's pulse length and num_ranges to be accounted for.

        # Find the sequence time. Add some TR setup time before the first pulse. The
        # timing to the last pulse is added, as well as its pulse length and the TR delay
        # at the end of last pulse.

        # tr_window_time is originally in seconds, convert to us.
        self.seqtime = (
            2 * tr_window_time * 1.0e6
            + self.combined_pulses_metadata[-1]["start_time_us"]
            + self.combined_pulses_metadata[-1]["total_pulse_len"]
        )

        # FIND the total scope sync time and number of samples to receive.
        self.sstime = self.seqtime + self.ssdelay

        # number of receive samples will round down
        # This is the number of receive samples to receive for the entire duration of the sequence
        # and afterwards. This starts before first pulse is sent and goes until the end of the scope
        # sync delay which is there for the amount of time necessary to get the echoes from the
        # specified number of ranges.
        self.numberofreceivesamples = int(
            self.transmit_metadata["rx_sample_rate"] * self.sstime * 1e-6
        )

        self.output_encodings = collections.defaultdict(list)

        # create debug dict for tx samples.
        debug_dict = {
            "txrate": txrate,
            "txctrfreq": self.txctrfreq,
            "pulse_timing": [],
            "pulse_sample_start": [],
            "sequence_samples": {},
            "decimated_samples": {},
            "dmrate": dm_rate,
        }

        for i, cpm in enumerate(combined_pulses_metadata):
            debug_dict["pulse_timing"].append(cpm["start_time_us"])
            debug_dict["pulse_sample_start"].append(cpm["pulse_sample_start"])

        for i in range(main_antenna_count):
            debug_dict["sequence_samples"][i] = []
            debug_dict["decimated_samples"][i] = []

        self.debug_dict = debug_dict

        first_slice_pulse_len = self.combined_pulses_metadata[0]["component_info"][0][
            "pulse_num_samps"
        ]
        full_pulse_samps = first_slice_pulse_len + 2 * tr_window_num_samps
        offset_to_start = int(full_pulse_samps / 2)
        self.first_rx_sample_start = offset_to_start

        self.blanks = self.find_blanks()

        self.align_sequences = reduce(
            lambda a, b: a or b, [s.align_sequences for s in self.slice_dict.values()]
        )
        if self.align_sequences:
            log.info("aligning sequences to 0.1 s boundaries.")

    def make_sequence(self, beam_iter, sequence_num):
        """
        Create the samples needed for each pulse in the sequence. This function is optimized to be
        able to generate new samples every sequence if needed. Modifies the samples_array and
        isarepeat fields of all pulse dictionaries needed for this sequence for radar_control to use
        in operation.

        :param      beam_iter:     The beam iterator
        :type       beam_iter:     int
        :param      sequence_num:  The sequence number in the ave period
        :type       sequence_num:  int

        :returns:   Transmit data for each pulse where each pulse is a dict, including timing and
                    samples
        :rtype:     list
        :returns:   The transmit sequence and related data to use for debug.
        :rtype:     dict
        """
        txrate = self.transmit_metadata["txrate"]

        buffer_len = int(txrate * self.sstime * 1e-6)
        # This is going to act as buffer for mixing pulses. It is the length of the receive samples
        # since we know this will be large enough to hold samples at any pulse position. There will
        # be a buffer for each antenna.
        sequence = np.zeros(
            [len(self.tx_main_antennas), buffer_len], dtype=np.complex64
        )

        for slice_id in self.slice_ids:
            exp_slice = self.slice_dict[slice_id]
            if exp_slice.rxonly:
                continue
            beam_num = exp_slice.tx_beam_order[beam_iter]
            # basic_samples: [num_antennas, num_samps]
            basic_samples = self.basic_slice_pulses[slice_id][beam_num]

            num_pulses = len(exp_slice.pulse_sequence)
            encode_fn = exp_slice.pulse_phase_offset
            if encode_fn:
                # Must return 1D array of length [pulses].
                phase_encoding = encode_fn(beam_num, sequence_num, num_pulses)

                # dimensions: [pulses]
                # Append list of phase encodings for this sequence, one per pulse.
                # output_encodings contains a list of lists for each slice id
                self.output_encodings[slice_id].append(phase_encoding)

                # phase_encoding: [pulses]
                # basic_samples: [antennas, samples]
                # samples: [pulses, antennas, samples]
                phase_encoding = np.radians(phase_encoding)
                phase_encoding = np.exp(1j * phase_encoding)
                samples = np.einsum("i,jk->ijk", phase_encoding, basic_samples)

            else:  # no encodings, all pulses in the slice are all the same
                samples = np.repeat(basic_samples[np.newaxis, :, :], num_pulses, axis=0)

            # sum the samples into their position in the sequence buffer. Find where the relative
            # timing of each pulse matches the sample number in the buffer. Directly sum the samples
            # for each pulse into the buffer position. If any pulses overlap, this is how they will
            # be mixed.
            for i, pulse in enumerate(self.combined_pulses_metadata):
                for component_info in pulse["component_info"]:
                    if component_info["slice_id"] == slice_id:
                        pulse_sample_start = component_info["pulse_sample_start"]
                        pulse_samples_len = component_info["pulse_num_samps"]
                        start = pulse["tr_window_num_samps"] + pulse_sample_start
                        end = start + pulse_samples_len

                        # samples: [pulses, tx_antenna_count, samples]
                        # sequence: [tx_antenna_count, buffer_len]
                        sequence[:, start:end] += samples[i, :, :]

        # copy the encoded and combined samples into the metadata for the sequence.
        pulse_data = []
        for i, pulse in enumerate(self.combined_pulses_metadata):
            pulse_sample_start = pulse["pulse_sample_start"]

            num_samples = pulse["total_num_samps"]
            start = pulse_sample_start
            end = start + num_samples + 2 * pulse["tr_window_num_samps"]
            samples = sequence[:, start:end]

            new_pulse_info = copy.deepcopy(pulse["pulse_transmit_data"])
            new_pulse_info["samples_array"] = samples

            if i != 0:
                last_pulse = pulse_data[i - 1]["samples_array"]
                if samples.shape == last_pulse.shape:
                    if np.isclose(samples, last_pulse).all():
                        new_pulse_info["isarepeat"] = True

            pulse_data.append(new_pulse_info)

        if __debug__:
            debug_dict = copy.deepcopy(self.debug_dict)
            debug_dict["sequence_samples"] = sequence
            debug_dict["decimated_samples"] = sequence[:, :: debug_dict["dmrate"]]
        else:
            debug_dict = None

        return pulse_data, debug_dict

    def find_blanks(self):
        """
        Finds the blanked samples after all pulse positions are calculated.
        """
        blanks = []
        dm_rate = self.debug_dict["dmrate"]
        for pulse in self.combined_pulses_metadata:
            pulse_start = pulse["pulse_sample_start"]
            num_samples = pulse["total_num_samps"] + 2 * pulse["tr_window_num_samps"]

            rx_sample_start = int(pulse_start / dm_rate)
            rx_num_samps = math.ceil(num_samples / dm_rate)

            pulse_blanks = np.arange(rx_sample_start, rx_sample_start + rx_num_samps)
            pulse_blanks += int(self.first_rx_sample_start / dm_rate)
            blanks.extend(pulse_blanks)

        return blanks

    def get_rx_phases(self, beam_iter):
        """
        Gets the receive phases for a given beam

        :param      beam_iter:  The beam iter in a scan.
        :type       beam_iter:  int

        :returns:   The receive phases for each possible beam for every main and intf antenna
        :rtype:     dict for both main and intf
        """

        temp_dict = copy.deepcopy(self.rx_beam_phases)
        for k, v in temp_dict.items():
            beam_num = self.slice_dict[k].rx_beam_order[beam_iter]
            if not isinstance(beam_num, list):
                beam_num = [beam_num]
            v["main"] = v["main"][beam_num, :]
            v["intf"] = v["intf"][beam_num, :]

        return temp_dict
