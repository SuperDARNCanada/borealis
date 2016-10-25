#!/usr/bin/python

# Take cp_objects and combine them to make one RCP. 
# Report Errors if cp_objects cannot be combined.
# Communicate with the driver to control the radar.

import sys
import os
import zmq
import json
import math
import cmath # for complex numbers
#from controlprog import controlprog # this is pointing to the current control program file.
import currentctrlprog # this brings in myprog.
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
sys.path.append('../utils/protobuf')
import driverpacket_pb2
import time

def setup_tx_socket(): # to send pulses to driver.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("tcp://10.65.0.25:33033")
    return cpsocket

def setup_rx_socket(): #to send data to be written to iqdat.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("tcp://10.65.0.25:33044")
    return cpsocket

def setup_cp_socket(): #to get refreshed control program updates.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("tcp://10.65.0.25:33555")
    return cpsocket

def get_prog(socket):
    update=json.dumps("UPDATE")
    socket.send(update)
    ack=socket.recv()
    reply=json.loads(ack)
    if reply=="YES":
        socket.send(json.dumps("READY"))
        new_prog=socket.recv()
        prog=json.loads(new_prog)
        return prog #TODO: serialize a control program (class not JSON serializable)
    elif reply=="NO":
        print "no update"
        return None

def get_phshift(beamdir,freq,chan,pulse_shift):
    """Form the beam given the beam direction (degrees off boresite), and the tx frequency,
    and a specified extra phase shift if there is any."""
    beamdir=float(beamdir)
    # create beam by creating a list of phshifts for ALL possible channels
    beamrad=abs(math.pi*float(beamdir)/180.0)
    #phshifts=[None]*16
    # pointing to right of boresight, use point in middle (hypothetically channel 7.5) as phshift=0 (all channels have a non-zero phase shift)
    phshift=2*math.pi*freq*(7.5-chan)*15*math.cos(math.pi/2-beamrad)/299792458
    if beamdir<0:
        phshift=-phshift # change sign if pointing to th/e left
    phshift=phshift+pulse_shift
    # add an extra phase shift if there is any specified (for phase shift keying, adding phases between pulses in sequence,etc.)
    # note phase shifts may be greater than 2 pi 	
    # phshifts is a list of phshifts for all channels 0 through 15 to enable receiving on all channels even if 
    # transmitting on only a few.

    return phshift

def get_wavetables(wavetype):
    #NOTE: will there be any other wavetypes.
    iwave_table=[]
    qwave_table=[]

    if wavetype=="SINE":
        wave_table_len=8192
        for i in range(0, wave_table_len):
            iwave_table.append(math.cos(i*2*math.pi/wave_table_len))
            qwave_table.append(math.sin(i*2*math.pi/wave_table_len))

    else:
        errmsg="Wavetype %s not defined" % (wavetype)
        sys.exit(errmsg)

    return iwave_table, qwave_table

def get_samples(rate, wave_freq, pullength, iwave_table, qwave_table):
    """Find the normalized sample array given the rate (Hz), frequency (Hz), pulse length (s), 
    and wavetables (list containing single cycle of waveform). Will shift for beam later."""

    wave_freq=float(wave_freq)
    rate=float(rate)
    wave_table_len=len(iwave_table) # both i and q wave tables should be the same length.

    # length of list determined by rate and length of pulse
    rsampleslen=int(rate*0.00001) # number of samples in ramp-up, ramp-down
    sampleslen=int(rate*pullength+2*rsampleslen) # 10 us ramp up and ramp down before/after pulse
    # samples is a numpy array of complex samples i+qj
    samples=np.empty([sampleslen],dtype=complex) #[[[] for i in range(sampleslen)] for i in range(len(channels))]
    #qsamples=np.empty([len(channels),sampleslen],dtype=float) #[[[] for i in range(sampleslen)] for i in range(len(channels))]
            
    # sample at wave_freq with given phase shift
    f_norm=wave_freq/rate # this is a float
    sample_skip=int(f_norm*wave_table_len) # this must be an int to create perfect sine - this int defines the frequency resolution of our generated waveform
    #print sample_skip, wave_table_len, rate
    ac_freq=(float(sample_skip)/float(wave_table_len))*rate
    #print ac_freq
    # TODO: add - what is the actual frequency of our waveform? Based on above integer
    # TODO: to get precise frequencies, we will need precise sample_skip which will mean a precise sample rate and wave_table length. May need to calculate sample rate to use based on required frequency.
    #for chi in range(len(channels)):
        #sample_shift=int(phshifts[channels[chi]]*wave_table_len/(2*math.pi)) # same integer sample shift on all samples.
    #want precise phasing - therefore do not use the above method which makes phase shift an integer and samples.


    # phasing will be done in shift_samples.
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
	
    # samples is an array of complex samples that needs to be phase shifted for all channels.
    return samples

def shift_samples(basic_samples, phshift):
    """Take the samples and shift by given phase shift in rads."""

    samples=np.empty([len(basic_samples)],dtype=complex) # is a numpy array of complex samples.

    for i in range(len(basic_samples)):
        samples[i]=basic_samples[i]*cmath.exp(1j*phshift)
    # samples is a numpy array of len(basic_samples)
    return samples

def plot_samples(samplesa, samplesb=np.empty([2],dtype=complex), samplesc=np.empty([2],dtype=complex)):
    """For testing only"""
    # plot samples to check
    fig, smpplot = plt.subplots(1, 1)
    #print ipulse_samples_dict[(0,12)].shape[1]
    #print phshifts_dict[(0,12)]
    # plot first three channels waveform for cp program 0 beamnum 12:
    smpplot.plot(range(0,samplesa.shape[0]), samplesa)
    smpplot.plot(range(0,samplesb.shape[0]), samplesb)
    smpplot.plot(range(0,samplesc.shape[0]), samplesc)
    plt.ylim([-1,1])
    plt.xlim([0,100])
    fig.savefig('./plot.png')
    plt.close(fig)
    return None


def data_to_driver(driverpacket, txsocket, pulse, isamples_list, qsamples_list, txctrfreq, rxctrfreq, txrate, numberofreceivesamples, repeat=False):
    #Send this data via zeromq to the driver. Receive acknowledgement.	
    if repeat==True:
        #print "EMPTY"
        driverpacket.Clear()
        # channels empty
        # samples empty
        # ctrfreq empty
        # rxrate and txrate empty
        driverpacket.timetosendsamples=pulse.timing
        driverpacket.SOB=pulse.SOB
        driverpacket.EOB=pulse.EOB
        # timetoio empty
    else:
    # SETUP data to send to driver for transmit.
        driverpacket.Clear() #clear message back to empty state.
        for chan in pulse.channels:
            chan_add=driverpacket.channels.append(chan)	
            #chan_add=chan
        for chi in range(len(isamples_list)):
            sample_add=driverpacket.samples.add() # add one Samples message for each channel.
            real_samp=driverpacket.samples[chi].real.extend(isamples_list[chi])
            imag_samp=driverpacket.samples[chi].imag.extend(qsamples_list[chi]) # add a list
        driverpacket.txcenterfreq=txctrfreq * 1000
        driverpacket.rxcenterfreq=rxctrfreq * 1000
        driverpacket.txrate=txrate
        driverpacket.numberofreceivesamples=numberofreceivesamples
        driverpacket.timetosendsamples=pulse.timing #past time zero, start of sequence.
        #print pulse.timing
        driverpacket.SOB=pulse.SOB
        driverpacket.EOB=pulse.EOB
    
    txsocket.send(driverpacket.SerializeToString())
    # get response:
    #tx_ack = socket.recv()
    tx_ack=1

    del driverpacket.samples[:]
            
    return tx_ack
	

def main():

    with open('../config.ini') as config_data:
        config=json.load(config_data)
        #print config
    #setup socket to send pulse samples over.
    txsocket=setup_tx_socket()

    #initialize driverpacket.
    driverpacket=driverpacket_pb2.DriverPacket()

	
    while True:
        # receive pulse data from run_RCP
        #cpsocket=setup_cp_socket()
        #scan=None
        #while scan==None:
        #    prog=get_prog(cp_socket)
        # build_RCP will reload after every scan.
        #print "got a prog"

        # decipher the scans and phasing data and iterate through.

        prog=currentctrlprog.build_RCP() # make myprog, defined in currentctrlprog

        # make wavetables, and dictionary of pulses so we aren't wasting time making them in actual scan.
        wavetable_dict={}
        for cpo in range(prog.cpo_num):
            wavetable_dict[cpo]=get_wavetables(prog.cpo_list[cpo].wavetype)

        #iterate through scans, AveragingPeriods, Sequences, and pulses.

        for scan in prog.scan_objects:
            beam_remaining=True
            scan_iter=0 #iterator for cycling through beam numbers
            scans_done=0
            while (beam_remaining):
                for aveperiod in scan.aveperiods: #go through each aveperiod once then increase the scan iterator to the next beam in each scan.
                    if scan_iter>=len(scan.scan_beams[aveperiod.keys[0]]): #all keys will have the same length scan_beams inside the aveperiod
                        scans_done=scans_done+1
                        if scans_done==len(scan.aveperiods):
                            beam_remaining=False
                            break
                        continue
                    print "New AveragingPeriod"
                    int_time=datetime.utcnow()
                    done_time=int_time+timedelta(0,float(aveperiod.intt)/1000)
                    nave=0
                    beamdir={} # create a dictionary of beam directions with the keys being the cpos in this averaging period.
                    for cpo in aveperiod.keys:
                        bmnum=scan.scan_beams[cpo][scan_iter]
                        #print bmnum
                        beamdir[cpo]=scan.beamdir[cpo][bmnum] #get the beamdir from the beamnumber for this CP-object at this iteration.
                        #print beamdir[cpo]          
                    #TODO:send RX data for this averaging period (each cpo will have own data file? (how stereo is implemented currently))
                    while (int_time < done_time):
                        for sequence in aveperiod.integrations: #just alternate sequences
                            if int_time>=done_time:
                                break
                            numberofreceivesamples=int(config[u'rx_sample_rate']*sequence.ssdelay*1e-6)
                            for pulse in sequence.pulses: #pulses are in order of timing so this works.
                                # TODO: check to see if pulse is any different from last pulse. If it isn't we can send blank fields in driverpacket.
                                isamples_list=[] #this will be a list of lists for all channels and their samples.
                                qsamples_list=[] 
                                if sequence.pulses.index(pulse)!=0:
                                    last_pulse=sequence.pulses[sequence.pulses.index(pulse)-1]
                                    if last_pulse.cpoid==pulse.cpoid:
                                        # same cp_object, meaning same channels, freq, pullen, wavetype
                                        if last_pulse.pulse_shift==pulse.pulse_shift:
                                        #same phase offset in addition to beam dir, so same within this sequence besides its timing.
                                            ack=data_to_driver(driverpacket,txsocket,pulse,isamples_list,qsamples_list,0,0,0,0,repeat=True)
                                        #if pulse is the same as last pulse, except pulsen, and timing won't be.
                                for channel in range(0,16):
                                    # get phase shifts for all channels even if not transmitting on all.
                                    phase_array=get_phshift(beamdir[pulse.cpoid],pulse.freq,channel,pulse.pulse_shift)
                                for channel in pulse.channels:
                                    basic_samples=get_samples(prog.txrate, (pulse.freq-prog.txctrfreq)*1000, float(pulse.pullen)/1000000, wavetable_dict[pulse.cpoid][0], wavetable_dict[pulse.cpoid][1]) #pulse.iwave_talbe, pulse.qwave_table)	
                                    pulse_samples=shift_samples(basic_samples, phase_array) # returns numpy array 
                                    # conver numpy array to lists.
                                    isamples_list.append((pulse_samples.real).tolist())
                                    qsamples_list.append((pulse_samples.imag).tolist())
                                ack=data_to_driver(driverpacket,txsocket,pulse,isamples_list,qsamples_list,prog.txctrfreq,prog.rxctrfreq,prog.txrate,numberofreceivesamples,repeat=False)
                                
                            nave=nave+1
                            int_time=datetime.utcnow()
                    print "Number of integrations: %d" % nave
                scan_iter=scan_iter+1
                

main()

