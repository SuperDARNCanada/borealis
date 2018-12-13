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
from scipy.fftpack import fft
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections
import os
import deepdish
import argparse
import random
import traceback


sys.path.append(os.environ["BOREALISPATH"])

borealis_path = os.environ['BOREALISPATH']
config_file = borealis_path + '/config.ini'
from sample_building.sample_building import get_phshift, shift_samples


def testing_parser():
    """
    Creates the parser for this script.
    
    :returns: parser, the argument parser for the testing script. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The name of a file that you want to analyze. This \
                                          can be a bfiq, iq, or testing file. The script will use the \
                                          timestamp to find the other related files, if they exist.")

    return parser


def fft_and_plot(samples, rate):
    fft_samps = fft(samples)
    T = 1.0/float(rate)
    num_samps = len(samples)
    xf = np.linspace(-1.0/(2.0*T),1.0/(2.0*T),num_samps)

    fig, smpplt = plt.subplots(1,1)

    fft_to_plot = np.empty([num_samps],dtype=np.complex64)
    halfway = int(math.ceil(float(num_samps)/2))
    fft_to_plot = np.concatenate([fft_samps[halfway:], fft_samps[:halfway]])
    #xf = xf[halfway-200:halfway+200]
    #fft_to_plot = fft_to_plot[halfway-200:halfway+200]
    smpplt.plot(xf, 1.0/num_samps * np.abs(fft_to_plot))
    return fig


def align_tx_samples(tx_samples, offset, array_len):
    zeros_offset = np.array([0.0]*(offset))
    #print('Shapes of arrays:{}, {}'.format(zeros_offset.shape, ant_samples[0].shape))
    aligned_ant_samples = np.concatenate((zeros_offset, tx_samples))

    if array_len > len(aligned_ant_samples):
        zeros_extender = np.array([0.0]*(array_len-len(aligned_ant_samples)))
        #print(len(zeros_extender))
        aligned_ant_samples = np.concatenate((aligned_ant_samples,zeros_extender))
    else:
        aligned_ant_samples = aligned_ant_samples[:array_len]

    return aligned_ant_samples


def correlate_and_align_tx_samples(tx_samples, some_other_samples):
    """
    :param tx_samples: array of tx samples
    :param some_other_samples: an arry of other samples at same sampling rate as tx_samples
    """
    corr = np.correlate(tx_samples,some_other_samples,mode='full')
    max_correlated_index = np.argmax(np.abs(corr))
    #print('Max index {}'.format(max_correlated_index))
    correlated_offset = max_correlated_index - len(tx_samples) + 2
    # TODO: why plus 2? figured out based on plotting. symptom of the 'full' correlation?
    #print('Correlated offset = {}'.format(correlated_offset))
    aligned_ant_samples = align_tx_samples(tx_samples, correlated_offset, len(some_other_samples))

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True)
    # ax1.plot(np.arange(len(tx_samples)),np.abs(tx_samples))
    # ax2.plot(np.arange(len(aligned_ant_samples)),np.abs(aligned_ant_samples))
    # ax3.plot(np.arange(len(some_other_samples)),np.abs(some_other_samples))
    # ax4.plot(np.arange(len(corr)),np.abs(corr))
    # plt.show()
    return aligned_ant_samples, correlated_offset


def beamform(antennas_data, beamdirs, rxfreq, antenna_spacing):
    """
    :param antennas_data: numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be 
    from the same array and are assumed to be side by side with antenna spacing 15.24 m, pulse_shift = 0.0
    :param beamdirs: list of azimuthal beam directions in degrees off boresite
    :param rxfreq: frequency to beamform at.
    """

    beamformed_data = []
    for beam_direction in beamdirs:
        #print(beam_direction)
        antenna_phase_shifts = []
        for antenna in range(0, antennas_data.shape[0]):
            #phase_shift = get_phshift(beam_direction, rxfreq, antenna, 0.0, 16, 15.24)
            phase_shift = math.fmod((-1 * get_phshift(beam_direction, rxfreq, antenna, 0.0,
                                                  antennas_data.shape[0], antenna_spacing)),
                                    2*math.pi)
            antenna_phase_shifts.append(phase_shift)
        phased_antenna_data = [shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0) for i in range(0, antennas_data.shape[0])]
        phased_antenna_data = np.array(phased_antenna_data)
        one_beam_data = np.sum(phased_antenna_data, axis=0)
        beamformed_data.append(one_beam_data)
    beamformed_data = np.array(beamformed_data)

    return beamformed_data


def get_offsets(samples_a, samples_b):
    """
    Return the offset of samples b from samples a.
    :param samples_a: numpy array of complex samples
    :param samples_b: another numpy array of complex samples
    """
    samples_diff = samples_a * np.conj(samples_b)
    # Gives you an array with numbers that have the magnitude of |samples_a| * |samples_b| but
    # the angle of angle(samples_a) - angle(samples_b)
    phase_offsets = np.angle(samples_diff)
    phase_offsets = phase_offsets * 180.0/math.pi
    return list(phase_offsets)


def find_pulse_indices(data, threshold):  # TODO change to not normalized
    """
    :param data: a numpy array of complex values to find the pulses within. 
    :param threshold: a magnitude valude threshold (absolute value) that pulses will be defined as being greater than.
    """
    absolute_max = max(abs(data))
    normalized_data = abs(data)/absolute_max
    pulse_points = (normalized_data > threshold)
    pulse_indices = list(np.where(pulse_points == True)[0])
    return pulse_indices


def plot_output_samples_iq_data(record_dict, record_filetype):
    """
    :param record_dict: a record containing data_dimensions, data_descriptors, antenna_array_names, and data (reshaped)
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be output_samples_iq type, but there might be multiple slices.
    """

    # data dimensions are num_antennas, num_sequences, num_samps

    antennas_present = [int(i.split('_')[-1]) for i in record_dict['antenna_arrays_order']]

    for sequence in range(0, record_dict['data'].shape[1]):
        print('Sequence number: {}'.format(sequence))
         
        for index in range(0, record_dict['data'].shape[0]):
            antenna = antennas_present[index]
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            if antenna < record_dict['main_antenna_count']:
                ax1.set_title('Main Antennas {}'.format(record_filetype))
                ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].real, label='Real {}'.format(antenna))
                ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].imag, label="Imag {}".format(antenna))
                ax1.legend()
            else:
                ax2.set_title('Intf Antennas {}'.format(record_filetype))
                ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].real, label='Real {}'.format(antenna))
                ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].imag, label="Imag {}".format(antenna))
                ax2.legend()                       
            plt.show()



def plot_bf_iq_data(record_dict, record_filetype):
    """
    :param record_dict: a record containing data_dimensions, data_descriptors, antenna_array_names, and data (reshaped)
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be bf_iq type, but there might be multiple slices.
    """

    # data dimensions are num_antenna_arrays, num_sequences, num_beams, num_samps

    beam = 0
    for sequence in range(0, record_dict['data'].shape[1]):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for index in range(0, record_dict['data'].shape[0]):
            antenna_array = record_dict['antenna_arrays_order'][index]

            if antenna_array == 'main':
                ax1.set_title('Main Array {} sequence {}'.format(record_filetype, sequence))
                ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].real, label='Real {}'.format(antenna_array))
                ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].imag, label="Imag {}".format(antenna_array))
                ax1.legend()
            else:
                ax2.set_title('Intf Array {} sequence {}'.format(record_filetype, sequence))
                ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].real, label='Real {}'.format(antenna_array))
                ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].imag, label="Imag {}".format(antenna_array))
                ax2.legend()                       
        plt.show()


def plot_all_bf_data(record_data):
    """
    :param record_data: dictionary with keys = filetypes, each value is a record dictionaries from that filetype, 
     for the same record identifier. (eg. '1543525820193')
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for filetype, record_dict in record_data.items():
        # normalized_real = record_dict['main_bf_data'][0].real / max(abs(record_dict['main_bf_data'][0]))
        # normalized_imag = record_dict['main_bf_data'][0].imag / max(abs(record_dict['main_bf_data'][0]))
        # ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), normalized_real, label='Normalized Real {}'.format(filetype))
        # ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), normalized_imag, label="Normalized Imag {}".format(filetype))
        # ax1.legend()
        # normalized_real = record_dict['intf_bf_data'][0].real / max(abs(record_dict['intf_bf_data'][0]))
        # normalized_imag = record_dict['intf_bf_data'][0].imag / max(abs(record_dict['intf_bf_data'][0]))
        # ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), normalized_real, label='INTF Normalized Real {}'.format(filetype))
        # ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), normalized_imag, label="INTF Normalized Imag {}".format(filetype))
        # ax2.legend()
        ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), record_dict['main_bf_data'][0].real, label='Real {}'.format(filetype))
        ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), record_dict['main_bf_data'][0].imag, label="Imag {}".format(filetype))
        ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), record_dict['intf_bf_data'][0].real, label='INTF Real {}'.format(filetype))
        ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), record_dict['intf_bf_data'][0].imag, label="INTF Imag {}".format(filetype))
    ax1.legend()
    ax2.legend()
    plt.show()


def main():
    parser = testing_parser()
    args = parser.parse_args()
    data_file_path = args.filename

    data_file = os.path.basename(data_file_path)

    try:
        with open(config_file) as config_data:
            config = json.load(config_data)
    except IOError:
        errmsg = 'Cannot open config file at {}'.format(config_file)
        raise Exception(errmsg)

    data_directory = config['data_directory']
    antenna_spacing = config['main_antenna_spacing']
    intf_antenna_spacing = config['interferometer_antenna_spacing']

    data_file_metadata = data_file.split('.')

    date_of_file = data_file_metadata[0]
    timestamp_of_file = '.'.join(data_file_metadata[0:3])
    station_name = data_file_metadata[3]
    slice_id_number = data_file_metadata[4]
    type_of_file = data_file_metadata[-2]  # XX.hdf5
    if type_of_file == slice_id_number:
        slice_id_number = '0'  # choose the first slice to search for other available files.
    else:
        type_of_file = slice_id_number + '.' + type_of_file
    file_suffix = data_file_metadata[-1]

    if file_suffix != 'hdf5':
        raise Exception('Incorrect File Suffix: {}'.format(file_suffix))

    output_samples_filetype = slice_id_number + ".output_samples_iq"
    bfiq_filetype = slice_id_number + ".bfiq"
    rawrf_filetype = "rawrf"
    tx_filetype = "txdata"
    file_types_avail = [bfiq_filetype, output_samples_filetype] #, tx_filetype, rawrf_filetype]

    if type_of_file not in file_types_avail:
        raise Exception(
            'Data type: {} not incorporated in script. Allowed types: {}'.format(type_of_file,
                                                                                 file_types_avail))

    data = {}
    for file_type in list(file_types_avail):  # copy of file_types_avail so we can modify it within.
        try:
            filename = data_directory + '/' + date_of_file + '/' + timestamp_of_file + \
                        '.' + station_name + '.' + file_type + '.hdf5'
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
        #record_name = '1543525820193' 
        record_name = random.choice(list(data[type_of_file].keys()))
        print('Record Name: {}'.format(record_name))

        record_data = {}

        try:
            for file_type in file_types_avail:
                record_data[file_type] = data[file_type][record_name]

                if file_type == bfiq_filetype:
                    bf_iq = record_data[bfiq_filetype]
                    number_of_beams = len(bf_iq['beam_azms'])
                    number_of_arrays = len(bf_iq['antenna_arrays_order'])
                    flat_data = np.array(bf_iq['data'])  
                    # reshape to 2 (main, intf) x nave x number_of_beams x number_of_samples
                    bf_iq_data = np.reshape(flat_data, (number_of_arrays, bf_iq['num_sequences'], number_of_beams, bf_iq['num_samps']))
                    bf_iq['data'] = bf_iq_data
                    beam_azms = bf_iq['beam_azms']
                    pulses = bf_iq['pulses']
                    decimated_rate = bf_iq['rx_sample_rate']
                    tau_spacing = bf_iq['tau_spacing']
                    freq = bf_iq['freq']
                    nave = bf_iq['num_sequences']
                    main_antenna_count = bf_iq['main_antenna_count']
                    intf_antenna_count = bf_iq['intf_antenna_count']

                if file_type == output_samples_filetype:
                    output_samples_iq = record_data[output_samples_filetype]
                    number_of_antennas = len(output_samples_iq['antenna_arrays_order'])

                    flat_data = np.array(output_samples_iq['data'])  
                    # reshape to number of antennas (M0..... I3) x nave x number_of_samples
                    output_samples_iq_data = np.reshape(flat_data, (number_of_antennas, output_samples_iq['num_sequences'], output_samples_iq['num_samps']))
                    output_samples_iq['data'] = output_samples_iq_data
                    antennas_present = [int(i.split('_')[-1]) for i in output_samples_iq['antenna_arrays_order']]
                    output_samples_iq['antennas_present'] = antennas_present

                if file_type == rawrf_filetype:
                    rawrf = record_data[rawrf_filetype]
                    number_of_antennas = rawrf['main_antenna_count'] + rawrf['intf_antenna_count']
                    #number_of_antennas = len(rawrf['antenna_arrays_order'])
                    flat_data = np.array(rawrf['data'])  
                    # reshape to number_of_antennas x number_of_samples x num_sequences
                    rawrf_data = np.reshape(flat_data, (rawrf['num_sequences'], number_of_antennas, rawrf['num_samps']))
                    rawrf['data'] = rawrf_data
                    rawrf['antennas_present'] = range(0,rawrf['main_antenna_count'] + rawrf['intf_antenna_count'])
                    # these are based on filter size. TODO test with modified filter sizes and
                    # build this based on filter size.

                    # determined by : 0.5 * filter_3_num_taps * dm_rate_1 * dm_rate_2 + 0.5 *
                    # filter_3_num_taps. First term is indicative of the number of samples
                    # that were added on so that we don't miss the first pulse, second term
                    # aligns the filter so that the largest part of it (centre) is over the pulse.

                    # This needs to be tested.
                    rawrf['dm_start_sample'] = 180*10*5 + 180

                # tx data does not need to be reshaped.
                if file_type == tx_filetype:
                    tx = record_data[tx_filetype]
                    tx['rx_sample_rate'] = int(tx['tx_rate'][0]/tx['dm_rate'])
                    print('Decimation rate error: {}'.format(tx['dm_rate_error']))
                    print(tx['rx_sample_rate'])
                    tx['data_descriptors'] = ['num_sequences', 'num_antennas', 'num_samps']
                    tx['data'] = tx['decimated_tx_samples']
                    tx['antennas_present'] = tx['decimated_tx_antennas'][0]
                    tx['dm_start_sample'] = 0


        except ValueError as e:
            print('Record {} raised an exception in filetype {}:\n'.format(record_name, file_type))
            traceback.print_exc()
            print('\nA new record will be selected.')
            record_attempts +=1
            if record_attempts == 3:
                print('FILES FAILED WITH 3 FAILED ATTEMPTS TO LOAD RECORDS.')
                raise # something is wrong with the files 
        else:  # no errors
            good_record_found = True

    if bfiq_filetype not in file_types_avail:
        raise Exception('Cannot do beamforming tests without beamformed iq to compare to.')

    # find pulse points in data that is decimated. 

    #plot_output_samples_iq_data(record_data[output_samples_filetype], output_samples_filetype)
    #plot_bf_iq_data(record_data[bfiq_filetype], bfiq_filetype)

    beamforming_dict = {}
    print('BEAM AZIMUTHS: {}'.format(beam_azms))
    for sequence_num in range(0,nave):
        print('SEQUENCE NUMBER {}'.format(sequence_num))
        sequence_dict = beamforming_dict[sequence_num] = {}
        for filetype, record_dict in record_data.items():
            #print(filetype)
            sequence_filetype_dict = sequence_dict[filetype] = {}
            data_description_list = list(record_dict['data_descriptors'])
            # STEP 1: DECIMATE IF NECESSARY
            if not math.isclose(record_dict['rx_sample_rate'], decimated_rate, abs_tol=0.001):
                # we aren't at 3.3 kHz - need to decimate.
                #print(record_dict['rx_sample_rate'])
                dm_rate = int(record_dict['rx_sample_rate']/decimated_rate)
                #print(dm_rate)
                dm_start_sample = record_dict['dm_start_sample']
                dm_end_sample = -1 - dm_start_sample # this is the filter size 
                if data_description_list == ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']:
                    decimated_data = record_dict['data'][0][sequence_num][:][dm_start_sample:dm_end_sample:dm_rate] # grab only main array data, first sequence, all beams.
                    intf_decimated_data = record_dict['data'][1][sequence_num][:][dm_start_sample:dm_end_sample:dm_rate]
                elif data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                    decimated_data = record_dict['data'][:,sequence_num,
                                     dm_start_sample:dm_end_sample:dm_rate]  # all antennas.
                elif data_description_list == ['num_sequences', 'num_antennas', 'num_samps']:
                    if filetype == tx_filetype:  # tx data has sequence number 0 for all 
                        decimated_data = record_dict['data'][0,:,dm_start_sample:dm_end_sample:dm_rate]
                    else:
                        decimated_data = record_dict['data'][sequence_num,:,dm_start_sample:dm_end_sample:dm_rate]
                else:
                    raise Exception('Not sure how to decimate with the dimensions of this data: {}'.format(record_dict['data_descriptors']))
                
            else:
                if data_description_list == ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']:
                    decimated_data = record_dict['data'][0,sequence_num,:,:] # only main array
                    intf_decimated_data = record_dict['data'][1,sequence_num,:,:]
                elif data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                    decimated_data = record_dict['data'][:,sequence_num,:] # first sequence only, all antennas.
                elif data_description_list == ['num_sequences', 'num_antennas', 'num_samps']:
                    if filetype == tx_filetype:
                        decimated_data = record_dict['data'][0,:,:] # first sequence only, all antennas.
                    else:
                        decimated_data = record_dict['data'][sequence_num,:,:] # first sequence only, all antennas.
                else:
                    raise Exception('Unexpected data dimensions: {}'.format(record_dict['data_descriptors']))

            # STEP 2: BEAMFORM ANY UNBEAMFORMED DATA
            if filetype != bfiq_filetype:
                # need to beamform the data. 
                antenna_list = []
                #print(decimated_data.shape)
                if data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                    for antenna in range(0, record_dict['data'].shape[0]):
                        antenna_list.append(decimated_data[antenna,:])
                    antenna_list = np.array(antenna_list)
                elif data_description_list == ['num_sequences', 'num_antennas', 'num_samps']:
                    for antenna in range(0, record_dict['data'].shape[1]):
                        antenna_list.append(decimated_data[antenna,:])
                    antenna_list = np.array(antenna_list)
                else:
                    raise Exception('Not sure how to beamform with the dimensions of this data: {}'.format(record_dict['data_descriptors']))

                # beamform main array antennas only. 
                main_antennas_mask = (record_dict['antennas_present'] < main_antenna_count)
                intf_antennas_mask = (record_dict['antennas_present'] >= main_antenna_count)
                decimated_beamformed_data = beamform(antenna_list[main_antennas_mask][:],
                                                     beam_azms, freq, antenna_spacing) # TODO test
                # without
                # .copy()
                intf_decimated_beamformed_data = beamform(antenna_list[intf_antennas_mask][:],
                                                          beam_azms, freq, intf_antenna_spacing)
            else:
                decimated_beamformed_data = decimated_data  
                intf_decimated_beamformed_data = intf_decimated_data

            sequence_filetype_dict['main_bf_data'] = decimated_beamformed_data # this has 2 dimensions: num_beams x num_samps for this sequence.
            sequence_filetype_dict['intf_bf_data'] = intf_decimated_beamformed_data


            # STEP 3: FIND THE PULSES IN THE DATA
            for beamnum in range(0, sequence_filetype_dict['main_bf_data'].shape[0]):

                len_of_data = sequence_filetype_dict['main_bf_data'].shape[1]
                pulse_indices = find_pulse_indices(sequence_filetype_dict['main_bf_data'][beamnum], 0.09)
                if len(pulse_indices) > len(pulses): # sometimes we get two samples from the same pulse.
                    if math.fmod(len(pulse_indices), len(pulses)) == 0.0:
                        step_size = int(len(pulse_indices)/len(pulses))
                        pulse_indices = pulse_indices[step_size-1::step_size]
                
                pulse_points = [False if i not in pulse_indices else True for i in range(0,len_of_data)]
                sequence_filetype_dict['pulse_indices'] = pulse_indices

                # verify pulse indices make sense.
                # tau_spacing is in microseconds
                num_samples_in_tau_spacing = int(round(tau_spacing * 1.0e-6 * decimated_rate))
                pulse_spacing = pulses * num_samples_in_tau_spacing
                expected_pulse_indices = list(pulse_spacing + pulse_indices[0])
                if expected_pulse_indices != pulse_indices:
                    sequence_filetype_dict['calculate_offsets'] = False
                    print(expected_pulse_indices)
                    print(pulse_indices)
                    print('Pulse Indices are Not Equal to Expected for filetype {} sequence {}'.format(filetype, sequence_num))
                    #print('Phase Offsets Cannot be Calculated for this filetype {} sequence {}'.format(filetype, sequence_num))
                else:
                    sequence_filetype_dict['calculate_offsets'] = True

                # get the phases of the pulses for this data.
                pulse_data = sequence_filetype_dict['main_bf_data'][beamnum][pulse_points]
                sequence_filetype_dict['pulse_samples'] = pulse_data
                pulse_phases = np.angle(pulse_data) * 180.0/math.pi
                sequence_filetype_dict['pulse_phases'] = pulse_phases
                #print('Pulse Indices:\n{}'.format(pulse_indices))
                #print('Pulse Phases:\n{}'.format(pulse_phases))

        # Compare phases from pulses in the various datasets.
        if output_samples_filetype in file_types_avail and bfiq_filetype in file_types_avail:
            if sequence_dict[output_samples_filetype]['calculate_offsets'] and sequence_dict[bfiq_filetype]['calculate_offsets']:
                beamforming_phase_offset = get_offsets(sequence_dict[output_samples_filetype]['pulse_samples'], sequence_dict[bfiq_filetype]['pulse_samples'])
                print('There are the following phase offsets (deg) between the prebf and bf iq data pulses on sequence {}: {}'.format(sequence_num, beamforming_phase_offset))

        if rawrf_filetype in file_types_avail and output_samples_filetype in file_types_avail:
            if sequence_dict[output_samples_filetype]['calculate_offsets'] and sequence_dict[rawrf_filetype]['calculate_offsets']:
                decimation_phase_offset = get_offsets(sequence_dict[rawrf_filetype]['pulse_samples'], sequence_dict[output_samples_filetype]['pulse_samples'])
                print('There are the following phase offsets (deg) between the rawrf and prebf iq data pulses on sequence {}: {}'.format(sequence_num, decimation_phase_offset))

        if tx_filetype in file_types_avail and rawrf_filetype in file_types_avail:
            if sequence_dict[tx_filetype]['calculate_offsets'] and sequence_dict[rawrf_filetype]['calculate_offsets']:
                decimation_phase_offset = get_offsets(sequence_dict[tx_filetype]['pulse_samples'], sequence_dict[rawrf_filetype]['pulse_samples'])
                print('There are the following phase offsets (deg) between the tx and rawrf iq data pulses on sequence {}: {}'.format(sequence_num, decimation_phase_offset))


if __name__ == '__main__':
    main()


#def find_tx_rx_delay_offset():  # use pulse points to do this.
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


#     #plt.savefig('/home/radar/borealis/testing/tmp/beamforming-plots/{}.png'.format(bf_iq_samples))

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

# # #plt.savefig('/home/radar/borealis/testing/tmp/beamforming-plots/{}.png'.format(timestamp_of_file))
# # #plt.close()

# # stage_1_all = make_dict(stage_3)['antenna_0']
# #fig2 = fft_and_plot(stage_1_all, 1.0e6)
# plt.show()
# #fig2 , fig2ax1 = plt.subplots(1,1, figsize=(50,5)) 

# #fig2ax1.plot(np.arange(len(stage_1_all)), stage_1_all.real, np.arange(len(stage_1_all)), stage_1_all.imag)

# #plt.show()
# #plt.savefig('/home/radar/borealis/testing/tmp/beamforming-plots/{}.png'.format(timestamp_of_file + '-stage-1'))

# phase_offset_dict = {'phase_offsets': bf_phase_offsets.tolist(), 'tx_rx_offset': undec_tx_rx_offset}
# with open("/home/radar/borealis/testing/tmp/beamforming-plots/{}.offsets".format(timestamp_of_file), 'w') as f:
#     json.dump(phase_offset_dict, f)    
# with open("/home/radar/borealis/testing/tmp/rx-pulse-phase-offsets/{}.offsets".format(timestamp_of_file), 'w') as f:
#     json.dump({'rx_pulse_phase_offsets': rx_pulse_phase_offsets.tolist()}, f)



# #for pulse_index, offset_time in enumerate(pulse_offset_errors):
# #    if offset_time != 0.0
# #        phase_rotation = offset_time * 2 * cmath.pi * 
# #        phase_offsets[pulse_index] = phase_offsets[pulse_index] * cmath.exp()



