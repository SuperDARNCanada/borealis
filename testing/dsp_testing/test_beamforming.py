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

fig, (ax1,ax2,ax3) = plt.subplots(3,1)


#print(combined_tx_samples.shape, main_beams.shape)

corr = np.correlate(ant_samples[0],main_beams[0],mode='same')

ax1.plot(np.arange(len(ant_samples[0])),np.abs(ant_samples[0]))
ax2.plot(np.arange(len(main_beams[0])),np.abs(main_beams[0]))
ax3.plot(np.arange(len(corr)),np.abs(corr))

plt.show()




