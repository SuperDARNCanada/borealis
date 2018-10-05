import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys


tx_samples_file = sys.argv[1]
bf_iq_samples = sys.argv[2]

with open(tx_samples_file, 'r') as f:
    tx_samples = json.load(f)

with open(bf_iq_samples, 'r') as f:
    bf_iq = json.load(f)


main_beams = []
for dsetk, dsetv in bf_iq.iteritems():
    for beamk, beamv in dsetv.iteritems():
        main_data = beamv['main']

        real = main_data['real']
        imag = main_data['imag']

        cmplx = np.array(real) + 1.0j*np.array(imag)
        main_beams.append(cmplx)

main_beams = np.array(main_beams)


ant_samples = []
pulse_offset_errors = tx_samples['pulse_offset_error']
decimated_sequences = tx_samples['decimated_sequence']
tx_rate = tx_samples['txrate']
tx_ctr_freq = tx_samples['txctrfreq']
pulse_sequence_timings = tx_samples['pulse_sequence_timing']
dm_rate_error = tx_samples['dm_rate_error']
dm_rate = tx_samples['dm_rate']

for antk, antv in decimated_sequences.iteritems():
    real = antv['real']
    imag = antv['imag']

    cmplx = np.array(real) + 1.0j*np.array(imag)
    ant_samples.append(cmplx)



ant_samples = np.array(ant_samples)
combined_tx_samples = np.sum(ant_samples,axis=0)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, sharex=True)


#print(combined_tx_samples.shape, main_beams.shape)

corr = np.correlate(combined_tx_samples,main_beams[0],mode='full')
max_correlated_index = np.argmax(np.abs(corr))
#print('Max index {}'.format(max_correlated_index))
correlated_offset = max_correlated_index - len(combined_tx_samples) +1
#print('Correlated offset = {}'.format(correlated_offset))
zeros_offset = np.array([0.0]*(correlated_offset))
#print('Shapes of arrays:{}, {}'.format(zeros_offset.shape, ant_samples[0].shape))
aligned_ant_samples = np.concatenate((zeros_offset, combined_tx_samples))
zeros_extender = np.array([0.0]*(len(main_beams[0])-len(aligned_ant_samples)))
aligned_ant_samples = np.concatenate((aligned_ant_samples,zeros_extender))
pulse_points = (aligned_ant_samples != 0.0)
tx_pulse_samples = aligned_ant_samples[pulse_points]
rx_pulse_samples = main_beams[0][pulse_points]
samples_diff = np.subtract(rx_pulse_samples, tx_pulse_samples)
phase_offsets = np.angle(samples_diff)
print('Phase offsets {}'.format(phase_offsets))

for pulse_index, offset_time in enumerate(pulse_offset_errors):
    if offset_time != 0.0
        phase_rotation = offset_time * 2 * cmath.pi * 
        phase_offsets[pulse_index] = phase_offsets[pulse_index] * cmath.exp()

ax1.plot(np.arange(len(ant_samples[0])),np.abs(ant_samples[0]))
ax2.plot(np.arange(len(aligned_ant_samples)),np.abs(aligned_ant_samples))
ax3.plot(np.arange(len(main_beams[0])),np.abs(main_beams[0]))
ax4.plot(np.arange(len(corr)),np.abs(corr))

plt.show()




