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

tx_samples_file = sys.argv[1]
timestamp_of_file = sys.argv[2]
date_of_file = timestamp_of_file.split('.')[0]

bf_iq_samples = '/data/borealis_data/' + date_of_file + '/' + timestamp_of_file + '.sas.bfiq.hdf5'
iq_samples = '/data/borealis_data/' + date_of_file + '/' + timestamp_of_file + '.sas.iq.hdf5'
raw_iq_samples = '/data/borealis_data/' + date_of_file + '/' + timestamp_of_file + '.sas.rawrf.hdf5'

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


with open(tx_samples_file, 'r') as f:
    tx_samples = json.load(f)

#with open(bf_iq_samples, 'r') as f:
#    bf_iq = json.load(f)

#with open(iq_samples, 'r') as f:
#    iq_file_data = json.load(f)

bf_iq = deepdish.io.load(bf_iq_samples)

iq_file_data = deepdish.io.load(iq_samples)

iq = iq_file_data['output_samples']['frequency_0']

stage_1 = iq_file_data['stage_1']['frequency_0']

stage_2 = iq_file_data['stage_2']['frequency_0']

stage_3 = iq_file_data['stage_3']['frequency_0']

raw_iq = deepdish.io.load(raw_iq_samples)

bf_iq_samples = os.path.basename(bf_iq_samples)

main_beams = []
for dsetk, dsetv in bf_iq.iteritems():
    for beamk, beamv in dsetv.iteritems():
        main_data = beamv['main']

        real = main_data['real']
        imag = main_data['imag']

        cmplx = np.array(real) + 1.0j*np.array(imag)
        main_beams.append(cmplx)

main_beams = np.array(main_beams)


def make_dict(stuff):
    antenna_samples_dict = {}
    for dsetk, dsetv in stuff.iteritems():
        real = dsetv['real']
        imag = dsetv['imag']
        antenna_samples_dict[dsetk] = np.array(real) + 1.0j*np.array(imag)
    return antenna_samples_dict

iq_ch0_list = make_dict(iq)['antenna_0']
stage_1_ch0_list = make_dict(stage_1)['antenna_0'][::300][6::]
stage_2_ch0_list = make_dict(stage_2)['antenna_0'][::30][6::]
#stage_3_ch0_list = make_dict(stage_3)['antenna_0']
    
ant_samples = []
pulse_offset_errors = tx_samples['pulse_offset_error']
decimated_sequences = tx_samples['decimated_sequence']
all_tx_samples = tx_samples['sequence_samples']
tx_rate = tx_samples['txrate']
tx_ctr_freq = tx_samples['txctrfreq']
pulse_sequence_timings = tx_samples['pulse_sequence_timing']
dm_rate_error = tx_samples['dm_rate_error']
dm_rate = tx_samples['dm_rate']

decimated_sequences = collections.OrderedDict(sorted(decimated_sequences.items(), key=lambda t: t[0]))



# for antk, antv in decimated_sequences.iteritems():
#     real = antv['real']
#     imag = antv['imag']

#     cmplx = np.array(real) + 1.0j*np.array(imag)
#     ant_samples.append(cmplx)


# ant_samples = np.array(ant_samples)

#combined_tx_samples = np.sum(ant_samples,axis=0)




#print(combined_tx_samples.shape, main_beams.shape)

# corr = np.correlate(ant_samples[0],main_beams[0],mode='full')
# max_correlated_index = np.argmax(np.abs(corr))
# #print('Max index {}'.format(max_correlated_index))
# correlated_offset = max_correlated_index - len(ant_samples[0]) +1
# #print('Correlated offset = {}'.format(correlated_offset))
# zeros_offset = np.array([0.0]*(correlated_offset))
# #print('Shapes of arrays:{}, {}'.format(zeros_offset.shape, ant_samples[0].shape))
# aligned_ant_samples = np.concatenate((zeros_offset, ant_samples[0]))

# ax1.plot(np.arange(len(combined_tx_samples)),np.abs(combined_tx_samples))
# #ax2.plot(np.arange(len(aligned_ant_samples)),np.abs(aligned_ant_samples))
# ax3.plot(np.arange(len(main_beams[0])),np.abs(main_beams[0]))
# ax4.plot(np.arange(len(corr)),np.abs(corr))

# plt.show()


#aligned_bf_samples, bf_tx_rx_offset = correlate_and_align_tx_samples(combined_tx_samples, main_beams[0])

tx_samples_dict = {}
for antk, antv in all_tx_samples.items():
    real = antv['real']
    imag = antv['imag']

    cmplx = np.array(real) + 1.0j*np.array(imag)
    tx_samples_dict[antk] = cmplx

#figx = plt.plot(np.arange(len(tx_samples_dict['0'])), tx_samples_dict['0'])
#plt.show()
# tr window time = 300 samples at start of pulses.


# This correlation is not aligning properly
#undec_aligned_samples, undec_tx_rx_offset = correlate_and_align_tx_samples(tx_samples_dict['0'], raw_iq['antenna_0'])

undec_tx_rx_offset = 18032

tx_dec_samples_dict = {}
for antk, antv in tx_samples_dict.items():
     offset_samples = align_tx_samples(antv, undec_tx_rx_offset, (len(main_beams[0])+6)*1500)
     tx_dec_samples_dict[antk] = offset_samples[::1500][6::]

# Beamform the tx samples only at the pulse points. 
pulse_points = (tx_dec_samples_dict['0'] != 0.0)
tx_pulse_points_dict = {}
for antk, antv in tx_dec_samples_dict.items():
    tx_pulse_points_dict[antk] = antv[pulse_points]
beamformed_tx_pulses = np.sum(np.array([antv for (antk, antv) in tx_pulse_points_dict.items()]), axis=0)

decimated_raw_iq = collections.defaultdict(dict)

for k,v in raw_iq.items():
    decimated_raw_iq[k] = v[::1500][6::]

aligned_to_raw_samples, raw_tx_rx_offset = correlate_and_align_tx_samples(tx_dec_samples_dict['0'],decimated_raw_iq['antenna_0'])
#good_snr = np.ma.masked_where(main_beams[0] > 0.3*(np.max(main_beams[0])), main_beams[0])




#print('Difference between offsets: {} samples'.format(raw_tx_rx_offset - undec_tx_rx_offset/1500))
#pulse_points = (aligned_bf_samples != 0.0)
#raw_pulse_points = (aligned_to_raw_samples != 0.0)



raw_pulse_points = (np.abs(decimated_raw_iq['antenna_0']) > 0.1)
print('Raw RF Channel 0 Pulses at Indices: {}'.format(np.where(raw_pulse_points==True)))
stage_1_pulse_points = (np.abs(stage_1_ch0_list) > 0.1)
stage_2_pulse_points = (np.abs(stage_2_ch0_list) > 0.1)
#stage_3_pulse_points = (np.abs(stage_3_ch0_list) > 0.08)
print('Stage 1 Channel 0 Pulses at Indices: {}'.format(np.where(stage_1_pulse_points==True)))
print('Stage 2 Channel 0 Pulses at Indices: {}'.format(np.where(stage_2_pulse_points==True)))
#print('Stage 3 Channel 0 Pulses at Indices: {}'.format(np.where(stage_3_pulse_points==True)))
iq_pulse_points = (np.abs(iq_ch0_list) > 0.08)
print('Iq Channel 0 Pulses at Indices: {}'.format(np.where(iq_pulse_points==True)))
print('Transmitted Decimated Samples Channel 0 Pulses at Indices: {}'.format(np.where(pulse_points==True)))

#beamformed_tx_pulses = aligned_bf_samples[pulse_points]
beamformed_rx_pulses = main_beams[0][pulse_points]
iq_ch0_pulses = iq_ch0_list[pulse_points]
raw_ch0_pulse_samples = decimated_raw_iq['antenna_0'][raw_pulse_points]
stage_1_ch0_pulses = stage_1_ch0_list[raw_pulse_points]
stage_2_ch0_pulses = stage_2_ch0_list[raw_pulse_points]
#stage_3_ch0_pulses = stage_3_ch0_list[raw_pulse_points]

#except:
#    print('File {} issues'.format(bf_iq_samples))


    #plt.savefig('/home/radar/borealis/testing/tmp/beamforming-plots/{}.png'.format(bf_iq_samples))

    #tx_angle = np.angle(beamformed_tx_pulses)
    #rx_angle = np.angle(beamformed_rx_pulses)
    #phase_offsets = rx_angle - tx_angle
    # normalize the samples
    #beamformed_tx_pulses_norm = beamformed_tx_pulses / np.abs(beamformed_tx_pulses)
    
    #beamformed_rx_pulses_norm = beamformed_rx_pulses / np.abs(beamformed_rx_pulses)
    #samples_diff = np.subtract(beamformed_rx_pulses_norm, beamformed_tx_pulses_norm)

def get_offsets(samples, tx_samples):
    samples_diff = samples * np.conj(tx_samples)
    return np.angle(samples_diff)

ch0_tx_phase = np.angle(tx_pulse_points_dict['0'])
tx_phase = np.angle(beamformed_tx_pulses)
raw_rf_ch0_phase = np.angle(raw_ch0_pulse_samples)
stage_1_phase = np.angle(stage_1_ch0_pulses)
bf_phase_offsets = get_offsets(beamformed_rx_pulses, beamformed_tx_pulses)
raw_ch0_phase_offsets = get_offsets(raw_ch0_pulse_samples, tx_pulse_points_dict['0'])
stage_1_ch0_offsets = get_offsets(stage_1_ch0_pulses, tx_pulse_points_dict['0'])
stage_2_ch0_offsets = get_offsets(stage_2_ch0_pulses, tx_pulse_points_dict['0'])

stage_1_raw_offsets = get_offsets(stage_1_ch0_pulses, raw_ch0_pulse_samples)
#stage_3_ch0_offsets = get_offsets(stage_3_ch0_pulses, tx_pulse_points_dict['0'])
iq_ch0_offsets = get_offsets(iq_ch0_pulses, tx_pulse_points_dict['0'])

tx_pulse_phase_offsets = np.max(np.angle(beamformed_tx_pulses)) - np.min(np.angle(beamformed_tx_pulses))
rx_pulse_phase_offsets = np.max(np.angle(beamformed_rx_pulses)) - np.min(np.angle(beamformed_rx_pulses))  
raw_pulse_phase_offsets = np.max(np.angle(raw_ch0_pulse_samples)) - np.min(np.angle(raw_ch0_pulse_samples))
stage_1_ch0_pulse_offsets = np.max(np.angle(stage_1_ch0_pulses)) - np.min(np.angle(stage_1_ch0_pulses))
stage_2_ch0_pulse_offsets = np.max(np.angle(stage_2_ch0_pulses)) - np.min(np.angle(stage_2_ch0_pulses))
#stage_3_ch0_pulse_offsets = np.angle(stage_3_ch0_pulses) - np.angle(stage_3_ch0_pulses[0])
iq_ch0_pulse_offsets = np.max(np.angle(iq_ch0_pulses)) - np.min(np.angle(iq_ch0_pulses))

processing_phase_offset = bf_phase_offsets - raw_ch0_phase_offsets
processing_phase_error = np.max(processing_phase_offset) - np.min(processing_phase_offset)

print('Offset to align undec tx and raw Rx (in number of samples): {}'.format(undec_tx_rx_offset))
print('Offset to align undec tx and raw (in us): {}\n'.format(undec_tx_rx_offset / 5.0)) # 5.0 MHz is sampling rate

print('Ch0 TX Phase: {}'.format(ch0_tx_phase))
print('Beamformed TX Phases: {}'.format(tx_phase))
print('Phase Erro between Transmitted Pulses: {}\n'.format(tx_pulse_phase_offsets))

print('Raw Channel 0 Phase: {}'.format(raw_rf_ch0_phase))
print('Raw Channel 0 Tx/Rx Phase Offsets: {}'.format(raw_ch0_phase_offsets))
print('Phase Error between Raw Received Pulses: {}\n'.format(raw_pulse_phase_offsets))

print('Stage 1 Phase: {}'.format(stage_1_phase))
print('Stage 1 Offset from Raw: {}'.format(stage_1_raw_offsets))
print('Stage 1 tx/rx Phase Offsets: {}'.format(stage_1_ch0_offsets))
print('Phase Error between stage 1 pulses: {}\n'.format(stage_1_ch0_pulse_offsets))

print('Stage 2 tx/rx Phase Offsets: {}'.format(stage_2_ch0_offsets))
print('Phase Error between stage 2 pulses: {}\n'.format(stage_2_ch0_pulse_offsets))

print('IQ data tx/rx Phase offsets: {}'.format(iq_ch0_offsets))
print('Phase Error between Channel 0 IQ Received Pulses: {}\n'.format(iq_ch0_pulse_offsets))

print('Phase Offset between TX Beamformed and RX Beamformed data: {}'.format(bf_phase_offsets))
print('Phase Error between Beamformed Data: {}\n'.format(rx_pulse_phase_offsets))

print('Processing Phase Offset: {}'.format(processing_phase_offset))
print('Error in Processing Phase Offset: {}'.format(processing_phase_error))








fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2, sharex='col', figsize=(18, 24))

#ax1.plot(np.arange(len(combined_tx_samples)),combined_tx_samples.real, np.arange(len(combined_tx_samples)), combined_tx_samples.imag)
ax1.set_ylabel('Transmit Channel 0\nSamples Aligned to RX')
ax1.plot(np.arange(len(tx_dec_samples_dict['0'])),tx_dec_samples_dict['0'].real, np.arange(len(tx_dec_samples_dict['0'])), tx_dec_samples_dict['0'].imag)
ax2.set_ylabel('Beamformed Received Samples I+Q')
ax2.plot(np.arange(len(main_beams[0])),main_beams[0].real,np.arange(len(main_beams[0])),main_beams[0].imag)

#ax4.set_ylabel('Transmit Samples Aligned to Raw')
#ax4.plot(np.arange(len(aligned_to_raw_samples)), aligned_to_raw_samples.real, np.arange(len(aligned_to_raw_samples)), aligned_to_raw_samples.imag)

ax5.set_ylabel('Decimated raw iq')
ax5.plot(np.arange(len(decimated_raw_iq['antenna_0'])), decimated_raw_iq['antenna_0'].real, np.arange(len(decimated_raw_iq['antenna_0'])), decimated_raw_iq['antenna_0'].imag)

ax6.set_ylabel('Stage 1 samples')
ax6.plot(np.arange(len(stage_1_ch0_list)), stage_1_ch0_list.real, np.arange(len(stage_1_ch0_list)), stage_1_ch0_list.imag)

ax7.set_ylabel('Stage 2 samples')
ax7.plot(np.arange(len(stage_2_ch0_list)), stage_2_ch0_list.real, np.arange(len(stage_2_ch0_list)), stage_2_ch0_list.imag)

ax8.set_ylabel('Channel 0 Received Samples I+Q')
ax8.plot(np.arange(len(iq_ch0_list)), iq_ch0_list.real, np.arange(len(iq_ch0_list)),iq_ch0_list.imag)

#ax8.set_ylabel('Stage 3 samples')
#ax8.plot(np.arange(len(stage_3_ch0_list)), stage_3_ch0_list.real, np.arange(len(stage_3_ch0_list)), stage_3_ch0_list.imag)
#ax3.plot(np.arange(len(good_snr)),np.angle(good_snr))
#ax4.plot(np.arange(len(corr)),np.abs(corr))

# #plt.savefig('/home/radar/borealis/testing/tmp/beamforming-plots/{}.png'.format(timestamp_of_file))
# #plt.close()

# stage_1_all = make_dict(stage_3)['antenna_0']
#fig2 = fft_and_plot(stage_1_all, 1.0e6)
plt.show()
#fig2 , fig2ax1 = plt.subplots(1,1, figsize=(50,5)) 

#fig2ax1.plot(np.arange(len(stage_1_all)), stage_1_all.real, np.arange(len(stage_1_all)), stage_1_all.imag)

#plt.show()
#plt.savefig('/home/radar/borealis/testing/tmp/beamforming-plots/{}.png'.format(timestamp_of_file + '-stage-1'))

phase_offset_dict = {'phase_offsets': bf_phase_offsets.tolist(), 'tx_rx_offset': undec_tx_rx_offset}
with open("/home/radar/borealis/testing/tmp/beamforming-plots/{}.offsets".format(timestamp_of_file), 'w') as f:
    json.dump(phase_offset_dict, f)    
with open("/home/radar/borealis/testing/tmp/rx-pulse-phase-offsets/{}.offsets".format(timestamp_of_file), 'w') as f:
    json.dump({'rx_pulse_phase_offsets': rx_pulse_phase_offsets.tolist()}, f)



#for pulse_index, offset_time in enumerate(pulse_offset_errors):
#    if offset_time != 0.0
#        phase_rotation = offset_time * 2 * cmath.pi * 
#        phase_offsets[pulse_index] = phase_offsets[pulse_index] * cmath.exp()



