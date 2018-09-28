import json
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


phases = []
ant_samples = []
for samplek, samplev in tx_samples.iteritems():
    for antk, antv in tx_samples.iteritems():
        real = antv['real']
        imag = antv['imag']

        cmplx = np.array(real) + 1.0j*np.array(imag)
        ant_samples.append(cmplx)
        phases.append(antv['phase'])


phases = np.array(phases)

ant_samples = np.array(ant_samples)
combined_tx_samples = np.sum(ant_samples,axis=1)

correlated_samples = []
for i in range(main_beams.shape[0]):
    prev_corr = -1.0
    corr = 0.0
    for j in range(main_beams.shape[1]):
        tx_sample_len = len(combined_tx_samples)

        window = main_beams[i][j:j+tx_sample_len]

        corr = np.correlate(window, combined_tx_samples)

        if corr < prev_corr:
            correlated_samples.append(window)
            break
        else:
            prev_corr = corr









