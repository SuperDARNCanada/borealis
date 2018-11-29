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
    #print('Correlated offset = {}'.format(correlated_offset))
    aligned_ant_samples = align_tx_samples(tx_samples, correlated_offset, len(some_other_samples))

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True)
    # ax1.plot(np.arange(len(tx_samples)),np.abs(tx_samples))
    # ax2.plot(np.arange(len(aligned_ant_samples)),np.abs(aligned_ant_samples))
    # ax3.plot(np.arange(len(some_other_samples)),np.abs(some_other_samples))
    # ax4.plot(np.arange(len(corr)),np.abs(corr))
    # plt.show()
    return aligned_ant_samples, correlated_offset


def beamform(antennas_data, beamdirs, rxfreq):
    """
    :param antennas_data: numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be 
    from the same array and are assumed to be side by side with antenna spacing 15.24 m, pulse_shift = 0.0
    :param beamdirs: list of azimuthal beam directions in degrees off boresite
    """

    beamformed_data = []
    for beam_direction in beamdirs:
        #print(beam_direction)
        antenna_phase_shifts = []
        for antenna in range(0, antennas_data.shape[0]):
            #phase_shift = get_phshift(beam_direction, rxfreq, antenna, 0.0, 16, 15.24)
            phase_shift = math.fmod((0.0 - get_phshift(beam_direction, rxfreq, antenna, 0.0, antennas_data.shape[0], 15.24)), 2*math.pi)
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
    return np.angle(samples_diff)


def find_pulse_indices(data, threshold):
    """
    :param data: a numpy array of complex values to find the pulses within. 
    :param threshold: a magnitude valude threshold (absolute value) that pulses will be defined as being greater than.
    """
    absolute_max = max(abs(data))
    normalized_data = abs(data)/absolute_max
    pulse_points = (normalized_data > threshold)
    pulse_indices = list(np.where(pulse_points == True)[0])
    return pulse_indices


parser = testing_parser()
args = parser.parse_args()
data_file_path = args.filename

data_file = os.path.basename(data_file_path)

data_file_metadata = data_file.split('.')

date_of_file = data_file_metadata[0]
timestamp_of_file = '.'.join(data_file_metadata[0:3])
station_name = data_file_metadata[3]
slice_id_number = data_file_metadata[4]
type_of_file = data_file_metadata[-2]  # XX.hdf5
if type_of_file == slice_id_number:
    slice_id_number = 0  # choose the first slice to search for other available files.
else:
    type_of_file = slice_id_number + '.' + type_of_file
file_suffix = data_file_metadata[-1]

if file_suffix != 'hdf5':
    raise Exception('Incorrect File Suffix: {}'.format(file_suffix))

output_samples_filetype = slice_id_number + ".output_samples_iq"
bfiq_filetype = slice_id_number + ".bfiq"
rawrf_filetype = "rawrf"
file_types_avail = [output_samples_filetype, bfiq_filetype]

if type_of_file not in file_types_avail:
    raise Exception('Type of Data Not Incorporated in Script: {}'.format(type_of_file))

data = {}
for file_type in list(file_types_avail):  # copy of file_types_avail so we can modify it within.
    try:
        filename = '/data/borealis_data/' + date_of_file + '/' + timestamp_of_file + \
                    '.' + station_name + '.' + file_type + '.hdf5'
        data[file_type] = deepdish.io.load(filename)
    except:
        file_types_avail.remove(file_type)
        if file_type == type_of_file:  # if this is the filename you provided.
            raise

# choose a record from the provided file. 
good_record_found = False
while not good_record_found:
    record_name = '1543525820193' 
    # record_name = random.choice(list(data[type_of_file].keys()))
    print(record_name)

    record_data = {}
    for file_type in file_types_avail:
        record_data[file_type] = data[file_type][record_name]

    try:
        if bfiq_filetype in file_types_avail:
            bf_iq = record_data[bfiq_filetype]
            number_of_beams = len(bf_iq['beam_azms'])
            number_of_arrays = len(bf_iq['antenna_arrays_order'])

            flat_data = np.array(bf_iq['data'])  
            # reshape to 2 (main, intf) x nave x number_of_beams x number_of_samples
            bf_iq_data = np.reshape(flat_data, (number_of_arrays, bf_iq['num_sequences'], number_of_beams, bf_iq['num_samps']))
            bf_iq['data'] = bf_iq_data

        if output_samples_filetype in file_types_avail:
            output_samples_iq = record_data[output_samples_filetype]
            number_of_antennas = len(output_samples_iq['antenna_arrays_order'])

            flat_data = np.array(output_samples_iq['data'])  
            # reshape to nave x number of antennas (M0..... I3) x number_of_samples
            output_samples_iq_data = np.reshape(flat_data, (number_of_antennas, output_samples_iq['num_sequences'], output_samples_iq['num_samps']))
            output_samples_iq['data'] = output_samples_iq_data
            output_samples_iq['data_descriptors'] = ['num_antennas', 'num_sequences', 'num_samps'] # TODO REMOVE

            # for sequence in range(0, output_samples_iq_data.shape[1]):
            #     print('Sequence number: {}'.format(sequence))
            #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            #     ax1.plot(np.arange(output_samples_iq['num_samps']), output_samples_iq['data'][0][sequence].real, label='Real {}'.format(output_samples_filetype))
            #     ax1.plot(np.arange(output_samples_iq['num_samps']), output_samples_iq['data'][0][sequence].imag, label="Imag {}".format(output_samples_filetype))
            #     ax1.legend()
            #     ax2.plot(np.arange(output_samples_iq['num_samps']), output_samples_iq['data'][1][sequence].real, label='Real {}'.format(output_samples_filetype))
            #     ax2.plot(np.arange(output_samples_iq['num_samps']), output_samples_iq['data'][1][sequence].imag, label="Imag {}".format(output_samples_filetype))
            #     ax2.legend()
            #     plt.show()

        if 'rawrf' in file_types_avail:
            rawrf = record_data[rawrf_filetype]
            number_of_antennas = rawrf['main_antenna_count'] + rawrf['intf_antenna_count']
            flat_data = np.array(rawrf['data'])  
            # reshape to number_of_antennas x number_of_samples
            rawrf_data = np.reshape(flat_data, (rawrf['num_sequences'], number_of_antennas, rawrf['num_samps']))
            rawrf['data'] = rawrf_data
    except ValueError as e:
        print('Record {} raised an exception:\n'.format(record_name))
        traceback.print_exc()
        print('\nA new record will be selected.')

    else:  # no errors
        good_record_found = True


# find pulse points in data that is decimated. 
nave = record_data[type_of_file]['num_sequences']
for sequence_num in range(0,nave):
    print('SEQUENCE NUMBER {}'.format(sequence_num))
    for filetype, record_dict in record_data.items():

        data_description_list = list(record_dict['data_descriptors'])
        
        # STEP 1: DECIMATE IF NECESSARY
        if record_dict['rx_sample_rate'] != 3333.0:
            # we aren't at 3.3 kHz - need to decimate.
            dm_rate = int(record_data['rx_sample_rate']/3333.0)
            print(dm_rate)
            if data_description_list == ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']:
                decimated_data = record_dict['data'][0][sequence_num][:][0::dm_rate] # grab only main array data, first sequence, all beams.
                intf_decimated_data = record_dict['data'][1][sequence_num][:][0::dm_rate]
            elif data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                decimated_data = record_dict['data'][:][sequence_num][0::dm_rate] # first sequence only, all antennas.
            else:
                raise Exception('Not sure how to decimate with the dimensions of this data: {}'.format(record_dict['data_descriptors']))
            
        else:
            if data_description_list == ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']:
                decimated_data = record_dict['data'][0][sequence_num][:][:] # only main array
                intf_decimated_data = record_dict['data'][1][sequence_num][:][:]
            elif data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                decimated_data = record_dict['data'][:][sequence_num][:] # first sequence only, all antennas.
            else:
                raise Exception('Unexpected data dimensions: {}'.format(record_dict['data_descriptors']))
        
        # if filetype != bfiq_filetype:
        #     # need to beamform the data. 
        #     if data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
        #         decimated_beamformed_data = [np.array(decimated_data[:][0])]
        #         decimated_beamformed_data = np.array(decimated_beamformed_data)
        #         #np.reshape(decimated_beamformed_data, (1, record_dict['num_samps']))
        #         intf_decimated_beamformed_data = [np.array(decimated_data[:][1])]
        #         intf_decimated_beamformed_data = np.array(intf_decimated_beamformed_data)
        #         #np.reshape(intf_decimated_beamformed_data, (1, record_dict['num_samps']))
        #     else:
        #         raise Exception('Not sure how to beamform with the dimensions of this data: {}'.format(record_dict['data_descriptors']))

        # else:
        #     decimated_beamformed_data = decimated_data  
        #     intf_decimated_beamformed_data = intf_decimated_data


        # STEP 2: BEAMFORM ANY UNBEAMFORMED DATA
        if filetype != bfiq_filetype:
            # need to beamform the data. 
            antenna_list = []
            if data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                for antenna in range(0, record_dict['data'].shape[1]):
                    antenna_list.append(decimated_data[:][antenna])
                antenna_list = np.array(antenna_list)
            else:
                raise Exception('Not sure how to beamform with the dimensions of this data: {}'.format(record_dict['data_descriptors']))

            # beamform main array antennas only. 
            print('Main antenna count: {}'.format(record_dict['main_antenna_count']))
            antennas_present = [int(i.split('_')[-1]) for i in record_dict['antenna_arrays_order']]
            main_antennas_mask = (antennas_present < record_dict['main_antenna_count'])
            intf_antennas_mask = (antennas_present >= record_dict['main_antenna_count'])
            decimated_beamformed_data = beamform(antenna_list[main_antennas_mask][:].copy(), record_dict['beam_azms'], record_dict['freq']) 
            intf_decimated_beamformed_data = beamform(antenna_list[intf_antennas_mask][:].copy(), record_dict['beam_azms'], record_dict['freq']) 
            summed_data = np.sum(antenna_list[main_antennas_mask][:], axis=0)
            record_dict['straight_summed_data'] = summed_data
        else:
            decimated_beamformed_data = decimated_data  
            intf_decimated_beamformed_data = intf_decimated_data

        record_dict['main_bf_data'] = decimated_beamformed_data # this has 2 dimensions: num_beams x num_samps
        record_dict['intf_bf_data'] = intf_decimated_beamformed_data

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
        print(type(record_dict['main_bf_data'][0]))
        ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), record_dict['main_bf_data'][0].real, label='Real {}'.format(filetype))
        ax1.plot(np.arange(record_dict['main_bf_data'].shape[1]), record_dict['main_bf_data'][0].imag, label="Imag {}".format(filetype))
        ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), record_dict['intf_bf_data'][0].real, label='INTF Real {}'.format(filetype))
        ax2.plot(np.arange(record_dict['intf_bf_data'].shape[1]), record_dict['intf_bf_data'][0].imag, label="INTF Imag {}".format(filetype))
    ax1.legend()
    ax2.legend()
    plt.show()

for sequence_num in range(0,nave):
    for filetype, record_dict in record_data.items():
        # STEP 3: FIND THE PULSES IN THE DATA
        for beamnum in range(0, record_dict['main_bf_data'].shape[0]):

            len_of_data = record_dict['num_samps']
            pulse_indices = find_pulse_indices(record_dict['main_bf_data'][beamnum], 0.55)
            if len(pulse_indices) > len(record_dict['pulses']): # sometimes we get two samples from the same pulse.
                if math.fmod(len(pulse_indices), len(record_dict['pulses'])) == 0.0:
                    step_size = int(len(pulse_indices)/len(record_dict['pulses']))
                    pulse_indices = pulse_indices[step_size-1::step_size]
            
            pulse_points = [False if i not in pulse_indices else True for i in range(0,len_of_data)]
            record_dict['pulse_indices'] = pulse_indices

            # verify pulse indices make sense.
            num_samples_in_tau_spacing = int(round(record_dict['tau_spacing'] * 1.0e-6 * record_dict['rx_sample_rate']))  # us
            pulse_spacing = record_dict['pulses'] * num_samples_in_tau_spacing
            expected_pulse_indices = list(pulse_spacing + pulse_indices[0])
            if expected_pulse_indices != pulse_indices:
                print(expected_pulse_indices)
                print(pulse_indices)
                raise Exception('Pulse Indices are Not Equal to Expected.')

            # get the phases of the pulses for this data.
            pulse_data = record_dict['main_bf_data'][beamnum][pulse_points]
            record_dict['pulse_samples'] = pulse_data
            pulse_phases = np.angle(pulse_data)
            record_dict['pulse_phases'] = pulse_phases
            print(filetype)
            print(pulse_phases)

        # Straight summed pre-bf data has been the same as running beamform() function on the prebf data every time. The below was a test.
        if filetype == output_samples_filetype:
            pulse_phases_straight_sum = np.angle(record_dict['straight_summed_data'][pulse_indices])
            print(pulse_phases_straight_sum)
            print(np.subtract(pulse_phases, pulse_phases_straight_sum))
# Compare phases from pulses in the various datasets.
    beamforming_phase_offset = get_offsets(record_data[output_samples_filetype]['pulse_samples'], record_data[bfiq_filetype]['pulse_samples'])
    print('There are the following phase offsets between the prebf and bf iq data pulses: {}'.format(beamforming_phase_offset))




#def check_beamforming(bf_iq_record, prebf_iq_record):
    # """
    # Combine the pre-beamformed samples to verify the beamformed samples produced.

    # :param bf_iq_record: beamformed record from bfiq file, data reshaped to num_arrays 
    #  x nave x number_of_beams x number_of_samples
    # :param prebf_iq_record: unbeamformed record from output_samples_iq file, data reshaped 
    #  to num_antennas x nave x number_of_beams x number_of_samples
    # """

    #for antenna in range(0, bf_iq_record['main_antenna_count']):



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

# # corr = np.correlate(ant_samples[0],main_beams[0],mode='full')
# # max_correlated_index = np.argmax(np.abs(corr))
# # #print('Max index {}'.format(max_correlated_index))
# # correlated_offset = max_correlated_index - len(ant_samples[0]) +1
# # #print('Correlated offset = {}'.format(correlated_offset))
# # zeros_offset = np.array([0.0]*(correlated_offset))
# # #print('Shapes of arrays:{}, {}'.format(zeros_offset.shape, ant_samples[0].shape))
# # aligned_ant_samples = np.concatenate((zeros_offset, ant_samples[0]))

# # ax1.plot(np.arange(len(combined_tx_samples)),np.abs(combined_tx_samples))
# # #ax2.plot(np.arange(len(aligned_ant_samples)),np.abs(aligned_ant_samples))
# # ax3.plot(np.arange(len(main_beams[0])),np.abs(main_beams[0]))
# # ax4.plot(np.arange(len(corr)),np.abs(corr))

# # plt.show()


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


# ch0_tx_phase = np.angle(tx_pulse_points_dict['0'])
# tx_phase = np.angle(beamformed_tx_pulses)
# raw_rf_ch0_phase = np.angle(raw_ch0_pulse_samples)
# stage_1_phase = np.angle(stage_1_ch0_pulses)
# bf_phase_offsets = get_offsets(beamformed_rx_pulses, beamformed_tx_pulses)
# raw_ch0_phase_offsets = get_offsets(raw_ch0_pulse_samples, tx_pulse_points_dict['0'])
# stage_1_ch0_offsets = get_offsets(stage_1_ch0_pulses, tx_pulse_points_dict['0'])
# stage_2_ch0_offsets = get_offsets(stage_2_ch0_pulses, tx_pulse_points_dict['0'])

# stage_1_raw_offsets = get_offsets(stage_1_ch0_pulses, raw_ch0_pulse_samples)
# #stage_3_ch0_offsets = get_offsets(stage_3_ch0_pulses, tx_pulse_points_dict['0'])
# iq_ch0_offsets = get_offsets(iq_ch0_pulses, tx_pulse_points_dict['0'])

# tx_pulse_phase_offsets = np.max(np.angle(beamformed_tx_pulses)) - np.min(np.angle(beamformed_tx_pulses))
# rx_pulse_phase_offsets = np.max(np.angle(beamformed_rx_pulses)) - np.min(np.angle(beamformed_rx_pulses))  
# raw_pulse_phase_offsets = np.max(np.angle(raw_ch0_pulse_samples)) - np.min(np.angle(raw_ch0_pulse_samples))
# stage_1_ch0_pulse_offsets = np.max(np.angle(stage_1_ch0_pulses)) - np.min(np.angle(stage_1_ch0_pulses))
# stage_2_ch0_pulse_offsets = np.max(np.angle(stage_2_ch0_pulses)) - np.min(np.angle(stage_2_ch0_pulses))
# #stage_3_ch0_pulse_offsets = np.angle(stage_3_ch0_pulses) - np.angle(stage_3_ch0_pulses[0])
# iq_ch0_pulse_offsets = np.max(np.angle(iq_ch0_pulses)) - np.min(np.angle(iq_ch0_pulses))

# processing_phase_offset = bf_phase_offsets - raw_ch0_phase_offsets
# processing_phase_error = np.max(processing_phase_offset) - np.min(processing_phase_offset)

# print('Offset to align undec tx and raw Rx (in number of samples): {}'.format(undec_tx_rx_offset))
# print('Offset to align undec tx and raw (in us): {}\n'.format(undec_tx_rx_offset / 5.0)) # 5.0 MHz is sampling rate

# print('Ch0 TX Phase: {}'.format(ch0_tx_phase))
# print('Beamformed TX Phases: {}'.format(tx_phase))
# print('Phase Erro between Transmitted Pulses: {}\n'.format(tx_pulse_phase_offsets))

# print('Raw Channel 0 Phase: {}'.format(raw_rf_ch0_phase))
# print('Raw Channel 0 Tx/Rx Phase Offsets: {}'.format(raw_ch0_phase_offsets))
# print('Phase Error between Raw Received Pulses: {}\n'.format(raw_pulse_phase_offsets))

# print('Stage 1 Phase: {}'.format(stage_1_phase))
# print('Stage 1 Offset from Raw: {}'.format(stage_1_raw_offsets))
# print('Stage 1 tx/rx Phase Offsets: {}'.format(stage_1_ch0_offsets))
# print('Phase Error between stage 1 pulses: {}\n'.format(stage_1_ch0_pulse_offsets))

# print('Stage 2 tx/rx Phase Offsets: {}'.format(stage_2_ch0_offsets))
# print('Phase Error between stage 2 pulses: {}\n'.format(stage_2_ch0_pulse_offsets))

# print('IQ data tx/rx Phase offsets: {}'.format(iq_ch0_offsets))
# print('Phase Error between Channel 0 IQ Received Pulses: {}\n'.format(iq_ch0_pulse_offsets))

# print('Phase Offset between TX Beamformed and RX Beamformed data: {}'.format(bf_phase_offsets))
# print('Phase Error between Beamformed Data: {}\n'.format(rx_pulse_phase_offsets))

# print('Processing Phase Offset: {}'.format(processing_phase_offset))
# print('Error in Processing Phase Offset: {}'.format(processing_phase_error))








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



