#
#
#
# Functions to process and build samples,
# including getting phase shifts from beam directions,
# and functions for creating samples
# as well as plotting them and shifting them

import numpy as np
import math
import cmath

def get_phshift(beamdir,freq,chan,pulse_shift): # REVIEW #26 chan should be antenna?
    """Form the beam given the beam direction (degrees off boresite),
       and the tx frequency, and a specified extra phase shift if there
       is any. # REVIEW #5 units for freq, chan, pulse_shift
    """
    if type(beamdir) == list: # REVIEW #39 should use isinstance instead http://stackoverflow.com/questions/707674/how-to-compare-type-of-an-object-in-python
        # if we're imaging and have multiple beam directions per integration period # REVIEW #6 TODO handle imaging/multiple beam directions
        beamdir = 0 # directly ahead for now
    else: # is a float
        beamdir=float(beamdir)

    beamrad=abs(math.pi*float(beamdir)/180.0)
    # Pointing to right of boresight, use point in middle
    #   (hypothetically channel 7.5) as phshift=0 # REVIEW #1 update this comment for 'channel' - should be 'beam'?
    #   so all channels have a non-zero phase shift
    phshift=2*math.pi*freq*(7.5-chan)*15*math.cos(math.pi/2-beamrad)/299792458 # REVIEW #29 looks like some magic numbers in here (speed of light/boresight beam-can be calculated by [max beam-1]/2/whatever 15 is?)
    if beamdir<0:
        # Change sign if pointing to the left
        phshift=-phshift
    phshift=phshift+pulse_shift
    phshift=math.fmod(phshift,2*math.pi)
    # Add an extra phase shift if there is any specified # REVIEW #1 should this comment be tied to the line 'phshift=phshift+pulse_shift' ?
    return phshift


def get_wavetables(wavetype):
    #NOTE: will there be any other wavetypes.

    if wavetype=="SINE":
        iwave_table=None
        qwave_table=None
        #wave_table_len=8192
        #for i in range(0, wave_table_len):
        #    iwave_table.append(math.cos(i*2*math.pi/wave_table_len))
        #    qwave_table.append(math.sin(i*2*math.pi/wave_table_len))

    else:
        iwave_table=[]
        qwave_table=[]
        errmsg="Wavetype %s not defined" % (wavetype)
        sys.exit(errmsg)

    return iwave_table, qwave_table

def get_samples(rate, wave_freq, pulse_len, iwave_table=None,
                qwave_table=None):
    """Find the normalized sample array given the rate (Hz),
       frequency (Hz), pulse length (s), and wavetables (list
       containing single cycle of waveform). Will shift for beam later.
       No need to use wavetable if just as sine wave.
    """

    wave_freq=float(wave_freq)
    rate=float(rate)

    if iwave_table is None and qwave_table is None:
        sampling_freq=2*math.pi*wave_freq/rate
        rsampleslen=int(rate*0.00001) # REVIEW #3 #29 #26 what is the conversion? is this supposed to be ramp? magic number should be used in config file. Name could be better than 'r'
        sampleslen=int(rate*pulse_len+2*rsampleslen) # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value)
        samples=np.empty([sampleslen],dtype=complex) # REVIEW #39 Don't really need to pass first argument as list, can just pass as int so np.empty(sampleslen,dtype=complex). zeros is also a potential since empty will have all random values but is marginally faster - maybe it is safer to use zeros.
        for i in range(0,rsampleslen): # REVIEW #1 Explain how you're indexing for amplitude in the rampup, in this code it will start at non-zero and hit max amplitude at the last index in the rampup
            amp=0.7*float(i+1)/float(rsampleslen) # REVIEW #29 0.7 is a magic number - should be something like max_amplitude of usrps in config file?
            rads=math.fmod(sampling_freq*i,2*math.pi) # REVIEW #33 Don't think that sin/cos need to be bounded by 2pi, so this math.fmod is unnecessary we think
            samples[i]=amp*math.cos(rads)+amp*math.sin(rads)*1j
        for i in range(rsampleslen,sampleslen-rsampleslen):
            amp=0.7
            rads=math.fmod(sampling_freq*i,2*math.pi)
            samples[i]=amp*math.cos(rads)+amp*math.sin(rads)*1j
        for i in range(sampleslen-rsampleslen,sampleslen):
            amp=0.7*float(sampleslen-i)/float(rsampleslen)
            rads=math.fmod(sampling_freq*i,2*math.pi)
            samples[i]=amp*math.cos(rads)+amp*math.sin(rads)*1j
        # we are using a sine wave and will use the sampling freq.
# REVIEW #39 There is another possible way to build these samples without for loops, see samplebuild.py. It may or may not be faster
    elif iwave_table is not None and qwave_table is not None:
        wave_table_len=len(iwave_table)
        rsampleslen=int(rate*0.00001) # REVIEW #3 #29 #26 what is the conversion? is this supposed to be ramp? magic number should be used in config file. Name could be better than 'r'
        # Number of samples in ramp-up, ramp-down
        sampleslen=int(rate*pulse_len+2*rsampleslen)
        samples=np.empty([sampleslen],dtype=complex)# REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value)

        # sample at wave_freq with given phase shift
        f_norm=wave_freq/rate
        sample_skip=int(f_norm*wave_table_len)
        # This must be an int to create perfect sine, and
        #   this int defines the frequency resolution of our generated
        #   waveform

        ac_freq=(float(sample_skip)/float(wave_table_len))*rate # REVIEW #26 - maybe use 'actual_freq'
        # This is the actual frequency given the sample_skip
# REVIEW #0 We should include the max_usrp_amplitude that we found (~0.7 or so) in the 'amp' variables below as well as in the middle samples
        for i in range (0, rsampleslen):
            amp=float(i+1)/float(rsampleslen) # rampup is linear
            if sample_skip<0:
                ind=-1*((abs(sample_skip*i))%wave_table_len)
            else:
                ind=(sample_skip*i)%wave_table_len
            samples[i]=(amp*iwave_table[ind]+amp*qwave_table[ind]*1j)
            #qsamples[chi,i]=amp*qwave_table[ind]
        for i in range(rsampleslen, sampleslen-rsampleslen):
            if sample_skip<0:
                ind=-1*((abs(sample_skip*i))%wave_table_len)
            else:
                ind=(sample_skip*i)%wave_table_len
            samples[i]=(iwave_table[ind]+qwave_table[ind]*1j)
            #qsamples[chi,i]=qwave_table[ind]
        for i in range(sampleslen-rsampleslen, sampleslen):
            amp=float(sampleslen-i)/float(rsampleslen)
            if sample_skip<0:
                ind=-1*((abs(sample_skip*i))%wave_table_len)
            else:
                ind=(sample_skip*i)%wave_table_len
            samples[i]=(amp*iwave_table[ind]+amp*qwave_table[ind]*1j)
            #qsamples[chi,i]=amp*qwave_table[ind]

    else:
        errmsg="Error: only one wavetable passed"
        sys.exit(errmsg) # REVIEW #6 TODO Handle gracefully or something

    # Samples is an array of complex samples
    # NOTE: phasing will be done in shift_samples function
    return samples


def shift_samples(basic_samples, phshift):
    """Take the samples and shift by given phase shift in rads."""
    samples=np.empty([len(basic_samples)],dtype=complex) # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value)
    for i in range(len(basic_samples)): # REVIEW #39 this can be an numpy array multiply: 'return basic_samples*np.exp(phshift*1j)'
        samples[i]=basic_samples[i]*cmath.exp(1j*phshift)
    return samples


def plot_samples(filename, samplesa,
                 samplesb=np.empty([2],dtype=complex), # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value)
                 samplesc=np.empty([2],dtype=complex)): # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value)
    """For testing only, plots samples to filename"""
    fig, smpplot = plt.subplots(1, 1)
    smpplot.plot(range(0,samplesa.shape[0]), samplesa)
    smpplot.plot(range(0,samplesb.shape[0]), samplesb)
    smpplot.plot(range(0,samplesc.shape[0]), samplesc)
    plt.ylim([-1,1])
    plt.xlim([0,100])
    fig.savefig(filename)
    plt.close(fig)
    return None


def plot_fft(filename, samplesa, rate):
    fft_samps=fft(samplesa)
    T= 1.0 /float(rate)
    num_samps=len(samplesa)
    xf=np.linspace(-1.0/(2.0*T),1.0/(2.0*T),num_samps)
    #print len(xf), len(fft_samps)
    fig, smpplt = plt.subplots(1,1)
    fft_to_plot=np.empty([num_samps],dtype=complex) # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value)
    if num_samps%2==1:
        halfway=(num_samps+1)/2
        for sample in range(halfway,num_samps):
            fft_to_plot[sample-halfway]=fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0,halfway):
            fft_to_plot[sample+halfway-1]=fft_samps[sample]
            # Move positive samples at end
    else:
        halfway=num_samps/2
        for sample in range(halfway,num_samps):
            fft_to_plot[sample-halfway]=fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0,halfway):
            fft_to_plot[sample+halfway]=fft_samps[sample]
            # Move positive samples at end
    smpplt.plot(xf, 1.0/num_samps * np.abs(fft_to_plot))
#    plt.xlim([-2500000,-2000000])
    fig.savefig(filename)
    plt.close(fig)
    return None


def make_pulse_samples(pulse_list, cpos, beamdir, txctrfreq, txrate,
                       power_divider):
    """Make and phase shift samples, and combine them if there are
       multiple pulse types to send within this pulse.
    """

    txrate=float(txrate)
    txctrfreq=float(txctrfreq)
    samples_dict={}

    for pulse in pulse_list:
        wave_freq=float(cpos[pulse[1]]['txfreq'])-txctrfreq
        samples_dict[tuple(pulse)]=[]
        phase_array=[]
        for channel in range(0,16): # REVIEW #29 Magic number 16 - is this equal to main antennas? should be a config option now
            # Get phase shifts for all channels
            phase_array.append(get_phshift(
                                    beamdir[pulse[1]],
                                    cpos[pulse[1]]['txfreq'],channel, # REVIEW #26 channel should be antenna 
                                    cpos[pulse[1]]['pulse_shift'][pulse[2]]))
        basic_samples=get_samples(txrate, wave_freq*1000, # REVIEW #1 comment for conversion of freq and pulse_len units (looks like kHz to Hz and us to s)
                                  float(cpos[pulse[1]]['pulse_len'])/1000000, None, # REVIEW #39 don't really need to put in the None, None since you've got default args in the get_samples method
                                  None)
        for channel in range(0,16): # REVIEW #6 TODO: Handle different amplitudes necessary for imaging. Something like pulse_samples = shape_samples(basic_samples, amplitude_array[antenna]) and that function could just be a numpy array multiply
            if channel in cpos[pulse[1]]['txchannels']:
                pulse_samples=shift_samples(basic_samples,
                                            phase_array[channel])
                samples_dict[tuple(pulse)].append(pulse_samples)
                # Samples_dict[pulse] is a list of numpy arrays now.
            else:
                pulse_samples=np.empty([len(basic_samples)],dtype=complex) # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value) # REVIEW #0 We should make this an array of 0s if we're not transmitting on this antenna, easier to pick out if you're looking at the samples array by eye and also less likely to transmit weird things if something horribly wrong happens.
                samples_dict[tuple(pulse)].append(pulse_samples)
                # Will be an empty array for that channel.

    # Combine samples given pulse timing in 'pulse' list.
    # Find timing of where the pulses start in comparison to the first
    #   pulse, and find out the total number of samples for this
    #   combined pulse.
    samples_begin=[]
    total_length=len(samples_dict[tuple(pulse_list[0])][0])
    for pulse in pulse_list:
        # pulse_list is in order of timing
        start_samples=int(txrate*float(pulse[0]-pulse_list[0][0])*1e-6)
        # First value in samples_begin should be zero.
        samples_begin.append(start_samples)
        if start_samples+len(samples_dict[tuple(pulse)][0])>total_length:
            total_length=start_samples+len(samples_dict[tuple(pulse)])
            # Timing from first sample + length of this pulse is max
    #print "Total Length : {}".format(total_length)

    # Now we have total length so make all pulse samples same length
    #   before combining them sample by sample.
    for pulse in pulse_list:
        start_samples=samples_begin[pulse_list.index(pulse)]
        #print start_samples
        for channel in range(0,16):
            array=samples_dict[tuple(pulse)][channel]
            new_array=np.empty([total_length],dtype=np.complex_) # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value) Also - what is the reason for np.complex_ ?
            for i in range(0,total_length):
                if i<start_samples:
                    new_array[i]=0.0
                if i>=start_samples and i<(start_samples+len(array)):
                    new_array[i]=array[i-samples_begin[
                                            pulse_list.index(pulse)]]
                if i>start_samples+len(array):
                    new_array[i]=0.0
            samples_dict[tuple(pulse)][channel]=new_array
            # Sub in new array of right length for old array.

    total_samples=[]
    # This is a list of arrays (one for each channel) with the combined
    #   samples in it (which will be transmitted).
    for channel in range(0,16):
        total_samples.append(samples_dict[tuple(pulse_list[0])][channel])
        for samplen in range(0,total_length):
            try:
                total_samples[channel][samplen]=(total_samples[channel][samplen] / power_divider)
            except RuntimeWarning:
                print "RUNTIMEWARNING {} {}".format(total_samples[channel][samplen], powerdivider)
            for pulse in pulse_list:
                if pulse==pulse_list[0]:
                    continue
                total_samples[channel][samplen]+=(samples_dict[tuple(pulse)]
                                                    [channel][samplen]
                                                    /power_divider)

    # Now get what channels we need to transmit on for this combined
    #   pulse.
    # print("First cpo: {}".format(pulse_list[0][1]))
    pulse_channels=cpos[pulse_list[0][1]]['txchannels']
    for pulse in pulse_list:
        for chan in cpos[pulse[1]]['txchannels']:
            if chan not in pulse_channels:
                pulse_channels.append(chan)
    pulse_channels.sort()

    return total_samples,pulse_channels



