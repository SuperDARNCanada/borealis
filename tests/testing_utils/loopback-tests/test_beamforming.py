#!/usr/bin/env python3

"""
    test_beamforming
    ~~~~~~~~~~~~~~~~
    Testing script for analyzing output data.

    :copyright: 2018 SuperDARN Canada
    :authors: Marci Detwiller, Keith Kotyk

"""

import json
import matplotlib

matplotlib.use("TkAgg")
import sys
import os
import argparse
import traceback


sys.path.append(os.environ["BOREALISPATH"])

from testing.testing_utils.plot_borealis_hdf5_data.plotting_borealis_data_utils import *
from testing.testing_utils.beamforming_utils import *

borealis_path = os.environ["BOREALISPATH"]
config_file = borealis_path + "/config.ini"


def testing_parser():
    """
    Creates the parser for this script.

    :returns: parser, the argument parser for the testing script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="The name of a file that you want to analyze. This \
                                          can be a bfiq, iq, or testing file. The script will use the \
                                          timestamp to find the other related files, if they exist.",
    )
    parser.add_argument(
        "--record", help="The record to look at, if you don't want a random record."
    )

    return parser


def main():
    parser = testing_parser()
    args = parser.parse_args()
    data_file_path = args.filename

    data_file = os.path.basename(data_file_path)

    try:
        with open(config_file) as config_data:
            config = json.load(config_data)
    except IOError:
        errmsg = "Cannot open config file at {}".format(config_file)
        raise Exception(errmsg)

    data_directory = os.path.dirname(data_file_path)
    antenna_spacing = float(config["main_antenna_spacing"])
    intf_antenna_spacing = float(config["interferometer_antenna_spacing"])

    data_file_metadata = data_file.split(".")

    date_of_file = data_file_metadata[0]
    timestamp_of_file = ".".join(data_file_metadata[0:3])
    station_name = data_file_metadata[3]
    slice_id_number = data_file_metadata[4]

    file_suffix = data_file_metadata[-1]

    if file_suffix not in ["hdf5", "site"]:
        raise Exception("Incorrect File Suffix: {}".format(file_suffix))

    if file_suffix == "hdf5":
        type_of_file = data_file_metadata[-2]  # XX.hdf5
    else:  # site
        type_of_file = data_file_metadata[-3]  # XX.hdf5.site
        file_suffix = data_file_metadata[-2] + "." + data_file_metadata[-1]

    if type_of_file == slice_id_number:
        slice_id_number = (
            "0"  # choose the first slice to search for other available files.
        )
    else:
        type_of_file = slice_id_number + "." + type_of_file

    output_samples_filetype = slice_id_number + ".antennas_iq"
    bfiq_filetype = slice_id_number + ".bfiq"
    rawrf_filetype = "rawrf"
    tx_filetype = "txdata"
    file_types_avail = [
        bfiq_filetype,
        output_samples_filetype,
        tx_filetype,
        rawrf_filetype,
    ]

    if type_of_file not in file_types_avail:
        raise Exception(
            "Data type: {} not incorporated in script. Allowed types: {}".format(
                type_of_file, file_types_avail
            )
        )

    data = {}
    for file_type in list(
        file_types_avail
    ):  # copy of file_types_avail so we can modify it within.
        try:
            filename = (
                data_directory
                + "/"
                + timestamp_of_file
                + "."
                + station_name
                + "."
                + file_type
                + "."
                + file_suffix
            )
            data[file_type] = deepdish.io.load(filename)
        except:
            file_types_avail.remove(file_type)
            if file_type == type_of_file:  # if this is the filename you provided.
                raise

    # Choose a record from the provided file, and get that record for each filetype to analyze side by side.
    # Also reshaping data to correct dimensions - if there is a problem with reshaping, we will also not use that record.
    good_record_found = False
    record_attempts = 0
    while not good_record_found:
        if args.record:
            record_name = args.record
        else:
            record_name = random.choice(list(data[type_of_file].keys()))
        print("Record Name: {}".format(record_name))

        record_data = {}

        try:
            for file_type in file_types_avail:
                record_data[file_type] = data[file_type][record_name]

                if file_type == bfiq_filetype:
                    bf_iq = record_data[bfiq_filetype]
                    number_of_beams = len(bf_iq["beam_azms"])
                    number_of_arrays = len(bf_iq["antenna_arrays_order"])
                    flat_data = np.array(bf_iq["data"])
                    # reshape to 2 (main, intf) x nave x number_of_beams x number_of_samples
                    bf_iq_data = np.reshape(
                        flat_data,
                        (
                            number_of_arrays,
                            bf_iq["num_sequences"],
                            number_of_beams,
                            bf_iq["num_samps"],
                        ),
                    )
                    bf_iq["data"] = bf_iq_data
                    beam_azms = bf_iq["beam_azms"]
                    pulses = bf_iq["pulses"]
                    decimated_rate = bf_iq["rx_sample_rate"]
                    tau_spacing = bf_iq["tau_spacing"]
                    freq = bf_iq["freq"]
                    nave = bf_iq["num_sequences"]
                    main_antenna_count = bf_iq["main_antenna_count"]
                    intf_antenna_count = bf_iq["intf_antenna_count"]

                if file_type == output_samples_filetype:
                    output_samples_iq = record_data[output_samples_filetype]
                    number_of_antennas = len(output_samples_iq["antenna_arrays_order"])

                    flat_data = np.array(output_samples_iq["data"])
                    # reshape to number of antennas (M0..... I3) x nave x number_of_samples
                    output_samples_iq_data = np.reshape(
                        flat_data,
                        (
                            number_of_antennas,
                            output_samples_iq["num_sequences"],
                            output_samples_iq["num_samps"],
                        ),
                    )
                    output_samples_iq["data"] = output_samples_iq_data
                    antennas_present = [
                        int(i.split("_")[-1])
                        for i in output_samples_iq["antenna_arrays_order"]
                    ]
                    output_samples_iq["antennas_present"] = antennas_present

                if file_type == rawrf_filetype:
                    rawrf = record_data[rawrf_filetype]
                    number_of_antennas = (
                        rawrf["main_antenna_count"] + rawrf["intf_antenna_count"]
                    )
                    # number_of_antennas = len(rawrf['antenna_arrays_order'])
                    flat_data = np.array(rawrf["data"])
                    # reshape to num_sequences x number_of_antennas x number_of_samples
                    rawrf_data = np.reshape(
                        flat_data,
                        (
                            rawrf["num_sequences"],
                            number_of_antennas,
                            rawrf["num_samps"],
                        ),
                    )
                    rawrf["data"] = rawrf_data
                    rawrf["antennas_present"] = range(
                        0, rawrf["main_antenna_count"] + rawrf["intf_antenna_count"]
                    )
                    # these are based on filter size. TODO test with modified filter sizes and
                    # build this based on filter size.

                    # determined by : 0.5 * filter_3_num_taps * dm_rate_1 * dm_rate_2 + 0.5 *
                    # filter_3_num_taps. First term is indicative of the number of samples
                    # that were added on so that we don't miss the first pulse, second term
                    # aligns the filter so that the largest part of it (centre) is over the pulse.

                    # This needs to be tested.
                    rawrf["dm_start_sample"] = 180 * 10 * 5 + 180

                # tx data does not need to be reshaped.
                if file_type == tx_filetype:
                    tx = record_data[tx_filetype]
                    try:
                        tx["tx_rate"] = tx["tx_rate"][0]
                        tx["dm_rate"] = tx["dm_rate"][0]
                    except IndexError:
                        pass
                    tx["rx_sample_rate"] = tx["tx_rate"] / tx["dm_rate"]
                    print("Decimation rate error: {}".format(tx["dm_rate_error"]))
                    print(tx["rx_sample_rate"])
                    tx["data_descriptors"] = [
                        "num_sequences",
                        "num_antennas",
                        "num_samps",
                    ]
                    tx["data"] = tx["decimated_tx_samples"]
                    tx["antennas_present"] = tx["decimated_tx_antennas"][0]
                    tx["dm_start_sample"] = 0

        except ValueError as e:
            print(
                "Record {} raised an exception in filetype {}:\n".format(
                    record_name, file_type
                )
            )
            traceback.print_exc()
            print("\nA new record will be selected.")
            record_attempts += 1
            if record_attempts == 3:
                print("FILES FAILED WITH 3 FAILED ATTEMPTS TO LOAD RECORDS.")
                raise  # something is wrong with the files
        else:  # no errors
            good_record_found = True

    if bfiq_filetype not in file_types_avail:
        raise Exception(
            "Cannot do beamforming tests without beamformed iq to compare to."
        )

    print(file_types_avail)
    # find pulse points in data that is decimated.

    # plot_output_samples_iq_data(record_data[output_samples_filetype], output_samples_filetype)
    plot_bf_iq_data(record_data[bfiq_filetype], bfiq_filetype)
    # plot_antenna_data(record_data[tx_filetype]['tx_samples'][0,:,:], record_data[tx_filetype]['tx_antennas'][0], tx_filetype, main_antenna_count)
    # plot_antenna_data(record_data[output_samples_filetype]['data'][:,0,:], record_data[output_samples_filetype]['antenna_arrays_order'], output_samples_filetype, main_antenna_count, separate_plots=False)
    # plot_antenna_data(record_data[rawrf_filetype]['data'][0,:,18000:20000], record_data[rawrf_filetype]['antennas_present'], rawrf_filetype, main_antenna_count,separate_plots=False)
    # for antenna in range(0,record_data[tx_filetype]['tx_samples'].shape[1]):
    #    fft_and_plot(record_data[tx_filetype]['tx_samples'][0,antenna,:], 5000000.0)

    beamforming_dict = {}
    # print('BEAM AZIMUTHS: {}'.format(beam_azms))
    for sequence_num in range(0, nave):
        # print('SEQUENCE NUMBER {}'.format(sequence_num))
        sequence_dict = beamforming_dict[sequence_num] = {}
        for filetype, record_dict in record_data.items():
            print(filetype)
            sequence_filetype_dict = sequence_dict[filetype] = {}
            data_description_list = list(record_dict["data_descriptors"])
            # STEP 1: DECIMATE IF NECESSARY
            if not math.isclose(
                record_dict["rx_sample_rate"], decimated_rate, abs_tol=0.001
            ):
                # print(decimated_rate)
                # print(record_dict['rx_sample_rate'])
                # we aren't at 3.3 kHz - need to decimate.
                # print(record_dict['rx_sample_rate'])
                dm_rate = int(record_dict["rx_sample_rate"] / decimated_rate)
                # print(dm_rate)
                dm_start_sample = record_dict["dm_start_sample"]
                dm_end_sample = -1 - dm_start_sample  # this is the filter size
                if data_description_list == [
                    "num_antenna_arrays",
                    "num_sequences",
                    "num_beams",
                    "num_samps",
                ]:
                    decimated_data = record_dict["data"][0][sequence_num][:][
                        dm_start_sample:dm_end_sample:dm_rate
                    ]  # grab only main array data, first sequence, all beams.
                    intf_decimated_data = record_dict["data"][1][sequence_num][:][
                        dm_start_sample:dm_end_sample:dm_rate
                    ]
                elif data_description_list == [
                    "num_antennas",
                    "num_sequences",
                    "num_samps",
                ]:
                    decimated_data = record_dict["data"][
                        :, sequence_num, dm_start_sample:dm_end_sample:dm_rate
                    ]  # all antennas.
                elif data_description_list == [
                    "num_sequences",
                    "num_antennas",
                    "num_samps",
                ]:
                    if filetype == tx_filetype:  # tx data has sequence number 0 for all
                        decimated_data = record_dict["data"][
                            0, :, dm_start_sample:dm_end_sample:dm_rate
                        ]
                    else:
                        decimated_data = record_dict["data"][
                            sequence_num, :, dm_start_sample:dm_end_sample:dm_rate
                        ]
                else:
                    raise Exception(
                        "Not sure how to decimate with the dimensions of this data: {}".format(
                            record_dict["data_descriptors"]
                        )
                    )

            else:
                if data_description_list == [
                    "num_antenna_arrays",
                    "num_sequences",
                    "num_beams",
                    "num_samps",
                ]:
                    decimated_data = record_dict["data"][
                        0, sequence_num, :, :
                    ]  # only main array
                    intf_decimated_data = record_dict["data"][1, sequence_num, :, :]
                elif data_description_list == [
                    "num_antennas",
                    "num_sequences",
                    "num_samps",
                ]:
                    decimated_data = record_dict["data"][
                        :, sequence_num, :
                    ]  # first sequence only, all antennas.
                elif data_description_list == [
                    "num_sequences",
                    "num_antennas",
                    "num_samps",
                ]:
                    if filetype == tx_filetype:
                        decimated_data = record_dict["data"][
                            0, :, :
                        ]  # first sequence only, all antennas.
                    else:
                        decimated_data = record_dict["data"][
                            sequence_num, :, :
                        ]  # first sequence only, all antennas.
                else:
                    raise Exception(
                        "Unexpected data dimensions: {}".format(
                            record_dict["data_descriptors"]
                        )
                    )

            sequence_filetype_dict["decimated_data"] = decimated_data

            # STEP 2: BEAMFORM ANY UNBEAMFORMED DATA
            if filetype != bfiq_filetype:
                # need to beamform the data.
                antenna_list = []
                # print(decimated_data.shape)
                if data_description_list == [
                    "num_antennas",
                    "num_sequences",
                    "num_samps",
                ]:
                    for antenna in range(0, record_dict["data"].shape[0]):
                        antenna_list.append(decimated_data[antenna, :])
                    antenna_list = np.array(antenna_list)
                elif data_description_list == [
                    "num_sequences",
                    "num_antennas",
                    "num_samps",
                ]:
                    for antenna in range(0, record_dict["data"].shape[1]):
                        antenna_list.append(decimated_data[antenna, :])
                    antenna_list = np.array(antenna_list)
                else:
                    raise Exception(
                        "Not sure how to beamform with the dimensions of this data: {}".format(
                            record_dict["data_descriptors"]
                        )
                    )

                # beamform main array antennas only.
                main_antennas_mask = (
                    record_dict["antennas_present"] < main_antenna_count
                )
                intf_antennas_mask = (
                    record_dict["antennas_present"] >= main_antenna_count
                )
                decimated_beamformed_data = beamform(
                    antenna_list[main_antennas_mask][:].copy(),
                    beam_azms,
                    freq,
                    antenna_spacing,
                )  # TODO test
                # without
                # .copy()
                intf_decimated_beamformed_data = beamform(
                    antenna_list[intf_antennas_mask][:].copy(),
                    beam_azms,
                    freq,
                    intf_antenna_spacing,
                )
            else:
                decimated_beamformed_data = decimated_data
                intf_decimated_beamformed_data = intf_decimated_data

            sequence_filetype_dict["main_bf_data"] = (
                decimated_beamformed_data  # this has 2 dimensions: num_beams x num_samps for this sequence.
            )
            sequence_filetype_dict["intf_bf_data"] = intf_decimated_beamformed_data

            # STEP 3: FIND THE PULSES IN THE DATA
            for beamnum in range(0, sequence_filetype_dict["main_bf_data"].shape[0]):

                len_of_data = sequence_filetype_dict["main_bf_data"].shape[1]
                pulse_indices = find_pulse_indices(
                    sequence_filetype_dict["main_bf_data"][beamnum], 0.3
                )
                if len(pulse_indices) > len(
                    pulses
                ):  # sometimes we get two samples from the same pulse.
                    if math.fmod(len(pulse_indices), len(pulses)) == 0.0:
                        step_size = int(len(pulse_indices) / len(pulses))
                        pulse_indices = pulse_indices[step_size - 1 :: step_size]

                pulse_points = [
                    False if i not in pulse_indices else True
                    for i in range(0, len_of_data)
                ]
                sequence_filetype_dict["pulse_indices"] = pulse_indices

                # verify pulse indices make sense.
                # tau_spacing is in microseconds
                num_samples_in_tau_spacing = int(
                    round(tau_spacing * 1.0e-6 * decimated_rate)
                )
                pulse_spacing = pulses * num_samples_in_tau_spacing
                expected_pulse_indices = list(pulse_spacing + pulse_indices[0])
                if expected_pulse_indices != pulse_indices:
                    sequence_filetype_dict["calculate_offsets"] = False
                    print(expected_pulse_indices)
                    print(pulse_indices)
                    print(
                        "Pulse Indices are Not Equal to Expected for filetype {} sequence {}".format(
                            filetype, sequence_num
                        )
                    )
                    print(
                        "Phase Offsets Cannot be Calculated for this filetype {} sequence {}".format(
                            filetype, sequence_num
                        )
                    )
                else:
                    sequence_filetype_dict["calculate_offsets"] = True

                # get the phases of the pulses for this data.
                pulse_data = sequence_filetype_dict["main_bf_data"][beamnum][
                    pulse_points
                ]
                sequence_filetype_dict["pulse_samples"] = pulse_data
                pulse_phases = np.angle(pulse_data) * 180.0 / math.pi
                sequence_filetype_dict["pulse_phases"] = pulse_phases
                # print('Pulse Indices:\n{}'.format(pulse_indices))
                # print('Pulse Phases:\n{}'.format(pulse_phases))

        # plot_antenna_data(record_data[tx_filetype]['tx_samples'][sequence_num,:,:], record_data[tx_filetype]['tx_antennas'][0], tx_filetype, main_antenna_count)
        # plot_antenna_data(sequence_dict[rawrf_filetype]['decimated_data'], record_data[rawrf_filetype]['antennas_present'], rawrf_filetype, main_antenna_count, separate_plots=False)

        # Compare phases from pulses in the various datasets.
        if (
            output_samples_filetype in file_types_avail
            and bfiq_filetype in file_types_avail
        ):
            if (
                sequence_dict[output_samples_filetype]["calculate_offsets"]
                and sequence_dict[bfiq_filetype]["calculate_offsets"]
            ):
                beamforming_phase_offset = get_offsets(
                    sequence_dict[output_samples_filetype]["pulse_samples"],
                    sequence_dict[bfiq_filetype]["pulse_samples"],
                )
                print(
                    "There are the following phase offsets (deg) between the prebf and bf iq data pulses on sequence {}: {}".format(
                        sequence_num, beamforming_phase_offset
                    )
                )

        if (
            rawrf_filetype in file_types_avail
            and output_samples_filetype in file_types_avail
        ):
            if (
                sequence_dict[output_samples_filetype]["calculate_offsets"]
                and sequence_dict[rawrf_filetype]["calculate_offsets"]
            ):
                decimation_phase_offset = get_offsets(
                    sequence_dict[rawrf_filetype]["pulse_samples"],
                    sequence_dict[output_samples_filetype]["pulse_samples"],
                )
                print(
                    "There are the following phase offsets (deg) between the rawrf and prebf iq data pulses on sequence {}: {}".format(
                        sequence_num, decimation_phase_offset
                    )
                )

        if tx_filetype in file_types_avail and rawrf_filetype in file_types_avail:
            if (
                sequence_dict[tx_filetype]["calculate_offsets"]
                and sequence_dict[rawrf_filetype]["calculate_offsets"]
            ):
                decimation_phase_offset = get_offsets(
                    sequence_dict[tx_filetype]["pulse_samples"],
                    sequence_dict[rawrf_filetype]["pulse_samples"],
                )
                print(
                    "There are the following phase offsets (deg) between the tx and rawrf iq data pulses on sequence {}: {}".format(
                        sequence_num, decimation_phase_offset
                    )
                )
    # print('Raw RF')
    # check_for_equal_samples_across_channels(record_data[rawrf_filetype], 18000, 20000, 0.02, main_antenna_count)
    # print('Output Samples Iq')
    # check_for_equal_samples_across_channels(record_data[output_samples_filetype], 0, record_data[output_samples_filetype]['num_samps'] - 1, 0.002, main_antenna_count)


if __name__ == "__main__":
    main()


# def find_tx_rx_delay_offset():  # use pulse points to do this.
# TODO


# def make_dict(stuff):
#     antenna_samples_dict = {}
#     for dsetk, dsetv in stuff.iteritems():
#         real = dsetv['real']
#         imag = dsetv['imag']
#         antenna_samples_dict[dsetk] = np.array(real) + 1.0j*np.array(imag)
#     return antenna_samples_dict

# iq_ch0_list = make_dict(iq)['antenna_0']
# stage_1_ch0_list = make_dict(stage_1)['antenna_0'][::300][6::]
# stage_2_ch0_list = make_dict(stage_2)['antenna_0'][::30][6::]
# #stage_3_ch0_list = make_dict(stage_3)['antenna_0']

# ant_samples = []
# pulse_offset_errors = tx_samples['pulse_offset_error']
# decimated_sequences = tx_samples['decimated_sequence']
# all_tx_samples = tx_samples['sequence_samples']
# tx_rate = tx_samples['txrate']
# tx_ctr_freq = tx_samples['txctrfreq']
# pulse_sequence_timings = tx_samples['pulse_sequence_timing']
# dm_rate_error = tx_samples['dm_rate_error']
# dm_rate = tx_samples['dm_rate']

# decimated_sequences = collections.OrderedDict(sorted(decimated_sequences.items(), key=lambda t: t[0]))

# # for antk, antv in decimated_sequences.iteritems():
# #     real = antv['real']
# #     imag = antv['imag']

# #     cmplx = np.array(real) + 1.0j*np.array(imag)
# #     ant_samples.append(cmplx)


# # ant_samples = np.array(ant_samples)

# #combined_tx_samples = np.sum(ant_samples,axis=0)


# #print(combined_tx_samples.shape, main_beams.shape)

# #aligned_bf_samples, bf_tx_rx_offset = correlate_and_align_tx_samples(combined_tx_samples, main_beams[0])

# tx_samples_dict = {}
# for antk, antv in all_tx_samples.items():
#     real = antv['real']
#     imag = antv['imag']

#     cmplx = np.array(real) + 1.0j*np.array(imag)
#     tx_samples_dict[antk] = cmplx

# #figx = plt.plot(np.arange(len(tx_samples_dict['0'])), tx_samples_dict['0'])
# #plt.show()
# # tr window time = 300 samples at start of pulses.


# # This correlation is not aligning properly
# #undec_aligned_samples, undec_tx_rx_offset = correlate_and_align_tx_samples(tx_samples_dict['0'], raw_iq['antenna_0'])

# undec_tx_rx_offset = 18032

# tx_dec_samples_dict = {}
# for antk, antv in tx_samples_dict.items():
#      offset_samples = align_tx_samples(antv, undec_tx_rx_offset, (len(main_beams[0])+6)*1500)
#      tx_dec_samples_dict[antk] = offset_samples[::1500][6::]

# # Beamform the tx samples only at the pulse points.
# pulse_points = (tx_dec_samples_dict['0'] != 0.0)
# tx_pulse_points_dict = {}
# for antk, antv in tx_dec_samples_dict.items():
#     tx_pulse_points_dict[antk] = antv[pulse_points]
# beamformed_tx_pulses = np.sum(np.array([antv for (antk, antv) in tx_pulse_points_dict.items()]), axis=0)

# decimated_raw_iq = collections.defaultdict(dict)

# for k,v in raw_iq.items():
#     decimated_raw_iq[k] = v[::1500][6::]

# aligned_to_raw_samples, raw_tx_rx_offset = correlate_and_align_tx_samples(tx_dec_samples_dict['0'],decimated_raw_iq['antenna_0'])
# #good_snr = np.ma.masked_where(main_beams[0] > 0.3*(np.max(main_beams[0])), main_beams[0])


# #print('Difference between offsets: {} samples'.format(raw_tx_rx_offset - undec_tx_rx_offset/1500))
# #pulse_points = (aligned_bf_samples != 0.0)
# #raw_pulse_points = (aligned_to_raw_samples != 0.0)


# raw_pulse_points = (np.abs(decimated_raw_iq['antenna_0']) > 0.1)
# print('Raw RF Channel 0 Pulses at Indices: {}'.format(np.where(raw_pulse_points==True)))
# stage_1_pulse_points = (np.abs(stage_1_ch0_list) > 0.1)
# stage_2_pulse_points = (np.abs(stage_2_ch0_list) > 0.1)
# #stage_3_pulse_points = (np.abs(stage_3_ch0_list) > 0.08)
# print('Stage 1 Channel 0 Pulses at Indices: {}'.format(np.where(stage_1_pulse_points==True)))
# print('Stage 2 Channel 0 Pulses at Indices: {}'.format(np.where(stage_2_pulse_points==True)))
# #print('Stage 3 Channel 0 Pulses at Indices: {}'.format(np.where(stage_3_pulse_points==True)))
# iq_pulse_points = (np.abs(iq_ch0_list) > 0.08)
# print('Iq Channel 0 Pulses at Indices: {}'.format(np.where(iq_pulse_points==True)))
# print('Transmitted Decimated Samples Channel 0 Pulses at Indices: {}'.format(np.where(pulse_points==True)))

# #beamformed_tx_pulses = aligned_bf_samples[pulse_points]
# beamformed_rx_pulses = main_beams[0][pulse_points]
# iq_ch0_pulses = iq_ch0_list[pulse_points]
# raw_ch0_pulse_samples = decimated_raw_iq['antenna_0'][raw_pulse_points]
# stage_1_ch0_pulses = stage_1_ch0_list[raw_pulse_points]
# stage_2_ch0_pulses = stage_2_ch0_list[raw_pulse_points]
# #stage_3_ch0_pulses = stage_3_ch0_list[raw_pulse_points]

# #except:
# #    print('File {} issues'.format(bf_iq_samples))


#     #plt.savefig('/home/radar/borealis/tools/tmp/beamforming-plots/{}.png'.format(bf_iq_samples))

#     #tx_angle = np.angle(beamformed_tx_pulses)
#     #rx_angle = np.angle(beamformed_rx_pulses)
#     #phase_offsets = rx_angle - tx_angle
#     # normalize the samples
#     #beamformed_tx_pulses_norm = beamformed_tx_pulses / np.abs(beamformed_tx_pulses)

#     #beamformed_rx_pulses_norm = beamformed_rx_pulses / np.abs(beamformed_rx_pulses)
#     #samples_diff = np.subtract(beamformed_rx_pulses_norm, beamformed_tx_pulses_norm)

# fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2, sharex='col', figsize=(18, 24))

# #ax1.plot(np.arange(len(combined_tx_samples)),combined_tx_samples.real, np.arange(len(combined_tx_samples)), combined_tx_samples.imag)
# ax1.set_ylabel('Transmit Channel 0\nSamples Aligned to RX')
# ax1.plot(np.arange(len(tx_dec_samples_dict['0'])),tx_dec_samples_dict['0'].real, np.arange(len(tx_dec_samples_dict['0'])), tx_dec_samples_dict['0'].imag)
# ax2.set_ylabel('Beamformed Received Samples I+Q')
# ax2.plot(np.arange(len(main_beams[0])),main_beams[0].real,np.arange(len(main_beams[0])),main_beams[0].imag)

# #ax4.set_ylabel('Transmit Samples Aligned to Raw')
# #ax4.plot(np.arange(len(aligned_to_raw_samples)), aligned_to_raw_samples.real, np.arange(len(aligned_to_raw_samples)), aligned_to_raw_samples.imag)

# ax5.set_ylabel('Decimated raw iq')
# ax5.plot(np.arange(len(decimated_raw_iq['antenna_0'])), decimated_raw_iq['antenna_0'].real, np.arange(len(decimated_raw_iq['antenna_0'])), decimated_raw_iq['antenna_0'].imag)

# ax6.set_ylabel('Stage 1 samples')
# ax6.plot(np.arange(len(stage_1_ch0_list)), stage_1_ch0_list.real, np.arange(len(stage_1_ch0_list)), stage_1_ch0_list.imag)

# ax7.set_ylabel('Stage 2 samples')
# ax7.plot(np.arange(len(stage_2_ch0_list)), stage_2_ch0_list.real, np.arange(len(stage_2_ch0_list)), stage_2_ch0_list.imag)

# ax8.set_ylabel('Channel 0 Received Samples I+Q')
# ax8.plot(np.arange(len(iq_ch0_list)), iq_ch0_list.real, np.arange(len(iq_ch0_list)),iq_ch0_list.imag)

# #ax8.set_ylabel('Stage 3 samples')
# #ax8.plot(np.arange(len(stage_3_ch0_list)), stage_3_ch0_list.real, np.arange(len(stage_3_ch0_list)), stage_3_ch0_list.imag)
# #ax3.plot(np.arange(len(good_snr)),np.angle(good_snr))
# #ax4.plot(np.arange(len(corr)),np.abs(corr))

# # #plt.savefig('/home/radar/borealis/tools/tmp/beamforming-plots/{}.png'.format(timestamp_of_file))
# # #plt.close()

# # stage_1_all = make_dict(stage_3)['antenna_0']
# #fig2 = fft_and_plot(stage_1_all, 1.0e6)
# plt.show()
# #fig2 , fig2ax1 = plt.subplots(1,1, figsize=(50,5))

# #fig2ax1.plot(np.arange(len(stage_1_all)), stage_1_all.real, np.arange(len(stage_1_all)), stage_1_all.imag)

# #plt.show()
# #plt.savefig('/home/radar/borealis/tools/tmp/beamforming-plots/{}.png'.format(timestamp_of_file + '-stage-1'))

# phase_offset_dict = {'phase_offsets': bf_phase_offsets.tolist(), 'tx_rx_offset': undec_tx_rx_offset}
# with open("/home/radar/borealis/tools/tmp/beamforming-plots/{}.offsets".format(timestamp_of_file), 'w') as f:
#     json.dump(phase_offset_dict, f)
# with open("/home/radar/borealis/tools/tmp/rx-pulse-phase-offsets/{}.offsets".format(timestamp_of_file), 'w') as f:
#     json.dump({'rx_pulse_phase_offsets': rx_pulse_phase_offsets.tolist()}, f)


# #for pulse_index, offset_time in enumerate(pulse_offset_errors):
# #    if offset_time != 0.0
# #        phase_rotation = offset_time * 2 * cmath.pi *
# #        phase_offsets[pulse_index] = phase_offsets[pulse_index] * cmath.exp()
