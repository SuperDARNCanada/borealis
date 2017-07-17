# Get arrays of increasing indices
ramp_up = np.arange(0,rsampleslen)
pulse_middle = np.arange(rsampleslen,sampleslen-rsampleslen)

# Calculate amplitudes
ramp_up_amplitude = max_amplitude*ramp_up/rsampleslen
pulse_middle_amplitude = max_amplitude
ramp_down_amplitude = np.flip(ramp_up_amplitude)

# Calculate rads
ramp_up_rads = ramp_up*sampling_freq
pulse_middle_rads = pulse_middle*sampling_freq
ramp_down_rads = np.flip(ramp_up_rads)

# Calculate samples, note that the continuity should be checked between rampup/middle and middle/rampdown
ramp_up_samples = ramp_up_amplitude * np.exp(ramp_up_rads*1j)
pulse_middle_samples = pulse_middle_amplitude * np.exp(pulse_middle_rads*1j)
ramp_down_samples = np.flip(ramp_up_samples) # OR ramp_down_amplitude *np.exp(ramp_down_rads*1j) # CHECK CORRECTNESS/FOR CONTINUITY

# concat the samples into one array
samples = np.concatenate(ramp_up_samples, pulse_middle_samples, ramp_down_samples)
