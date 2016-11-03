#!/usr/bin/python

# Test Program to take a single cp_object and run it. 
# Original Test program.
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
sys.path.append('../build/debug/utils/protobuf')
import driverpacket_pb2
import time

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
	print ac_freq
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


def data_to_driver(data,socket):
	#Send this data via zeromq to the driver. Receive acknowledgement.	

	#for request in myprog.cpo_list[0].channels:
		# send request:
	socket.send(data.SerializeToString())
		
		# get response:
	#tx_ack = socket.recv()

	tx_ack=1

	return tx_ack
	

def main():
	
	myprog=currentctrlprog.experiment() # make myprog, defined in currentctrlprog
	#create_cp(controlprog)

	# normalized wave tables of i and q. 
	iwave_table=[]
	qwave_table=[]
	wave_table_len=8192
	for i in range(0, wave_table_len):
		iwave_table.append(math.cos(i*2*math.pi/wave_table_len))
		qwave_table.append(math.sin(i*2*math.pi/wave_table_len))
	# no errors? continue
	
	# time to serialize the data. make a dictionary of what needs to be passed.
	# metadata: how many separate pulse sequences are there
	# how are they separated (separate scans, separate channels, etc.
	# right now there is only one cp_object so we only have one pulse sequence.

	# scan boundary? Wait to start

	ctrfreq=12000 # 12 MHz
	rate=5e6
	
	wave_freq=myprog.cpo_list[0].freq-ctrfreq
	
	# create dictionary of all possible pulses
	# pulse sample array varies by length (pullength), frequency, phasing (beamdir)
	# number of different pulses = len(cpo_list[0].beamdir) + len(cpo_list[1].beamdir) [all cp_objects only have one pullength
	# and one frequency]

	# create tuple of pulse keys ( defined by their cp_object number and their beam direction )
	pullist=[]
	for cpo in range(0,myprog.cpo_num):
		for bmnum in range(0,len(myprog.cpo_list[cpo].beamdir)):
			for chan in myprog.cpo_list[cpo].channels:
				#for intn in range(0, myprog.cpo_list[cpo].intn):
				for pulsen in range(0, len(myprog.cpo_list[cpo].sequence)):
					pullist.append((cpo,bmnum,chan,pulsen)) # all variables the pulse could change with.
	# intn will not be a variable in the pulse dictionary because you could implement this with pulsen in a new cpo that goes nave by nave.
	pulse_keys=tuple(pullist) # convert to tuple to use as dictionary keys.
	#print pulse_keys	

	# iterate through pulse_keys, creating pulse dictionary.
	phshifts_dict={}
	pulse_samples_dict={}
	for cpo,bmnum,chan,pulsen in pulse_keys:
		# get phase shifts for this beam direction
		phshifts_dict[(cpo,bmnum,chan,pulsen)]=get_phshift(myprog.cpo_list[cpo].beamdir[bmnum],myprog.cpo_list[cpo].freq*1000,chan,myprog.cpo_list[cpo].pulse_shift[pulsen]) 
		# returns a phshift in rads for this cp,bmnum,channel,and pulsen
		# get samples for one pulse, then phase it for this beam, cpo, channel, integration, pulsenumber
		basic_samples=get_samples(rate, wave_freq*1000, float(myprog.cpo_list[cpo].pullen)/1000000, iwave_table, qwave_table)	
		pulse_samples_dict[(cpo,bmnum,chan,pulsen)]=shift_samples(basic_samples, phshifts_dict[cpo,bmnum,chan,pulsen]) # returns numpy array 
	#print phshifts_dict

	#plot for testing
	#plot_samples(pulse_samples_dict[(0, 12, 7, 0)].real,pulse_samples_dict[(0,12,7,0)].imag)
	

	#initialize zmq socket.
	context=zmq.Context()
	socket=context.socket(zmq.PAIR)
	socket.connect("ipc:///tmp/feeds/0")

	#initialize driverpacket.
	driverpacket=driverpacket_pb2.DriverPacket()

	# start the scan with cpo_object[0].scan
	integ_n=len(myprog.cpo_list[0].scan) #integration number starts at 0; simulate end of scan to start new scan inside loop.
	while True: # TODO: change to while ("receiving a keep-going signal from scheduler/cp")
		if integ_n==len(myprog.cpo_list[0].scan):
			print "New Scan Starting"
			integ_n=0
		bmnum=myprog.cpo_list[0].scan[integ_n]
		int_time=datetime.utcnow()
		nave=0
		# get timing data
		time_table=[]
		for i in range(len(myprog.cpo_list[0].sequence)):
			time_table.append(myprog.cpo_list[0].sequence[i]*myprog.cpo_list[0].mpinc) # in us
		done_time=int_time+timedelta(0,float(myprog.cpo_list[0].intt)/1000)
		while int_time < done_time:
			# send pulses in accordance with pulse table/tau. iterate through pulse table.
			for seqn in range(len(myprog.cpo_list[0].sequence)):
				# send pulses
				# serialize the data for tx
				# create one data dictionary for current pulse to send to driver
				# 
				# pulse time = time_table[seqn]
				isamples_list=[] #this will be a list of lists for all channels and their samples.
				qsamples_list=[]
				for chan in myprog.cpo_list[0].channels:
					isamples_list.append((pulse_samples_dict[(0,bmnum,chan,seqn)].real).tolist())
					qsamples_list.append((pulse_samples_dict[(0,bmnum,chan,seqn)].imag).tolist())
				#print isamples_list[0]
				#print qsamples_list[0]
				if seqn==0 and len(myprog.cpo_list[0].sequence)==1:
					SOB=True
					EOB=True
				elif seqn==0:
					SOB=True
					EOB=False
				elif seqn==len(myprog.cpo_list[0].sequence)-1:
					SOB=False
					EOB=True
				else:
					SOB=False
					EOB=False
				
				# clear the message.	
				#for sam in range(len(driverpacket.samples)):
				#	driverpacket.samples[sam].Clear() #clear samples message back to empty state.
			#	driverpacket.ClearField(driversamples.real)
				#driversamples.Clear()
				#print "After Clearing samples: ", len(driverpacket.samples)
				driverpacket.Clear() #clear message back to empty state.
				for chan in myprog.cpo_list[0].channels:
					chan_add=driverpacket.channels.append(chan)	
					#chan_add=chan
				for chi in range(len(isamples_list)):
					sample_add=driverpacket.samples.add() # add one Samples message for each channel.
					real_samp=driverpacket.samples[chi].real.extend(isamples_list[chi])
					imag_samp=driverpacket.samples[chi].imag.extend(qsamples_list[chi]) # add a list
					#for si in range(len(isamples_list[0])):
						#real_samp=sample_add.real.append(isamples_list[chi][si]) #add real samples
						#imag_samp=sample_add.imag.append(qsamples_list[chi][si])
						#real_samp=isamples_list[chi][si]
						#imag_samp=qsamples_list[chi][si]
				#wfreq_add=driverpacket.waveformfreq.append(wave_freq)
				#print "Done getting samples in driver packet!"
				driverpacket.centerfreq=ctrfreq * 1000
				driverpacket.rxrate=rate
				driverpacket.txrate=rate
				driverpacket.timetosendsamples=time_table[seqn]
				driverpacket.SOB=SOB
				driverpacket.EOB=EOB
				driverpacket.timetoio=0
				print "OK!"
#				for fn in range(len(ctrfreq)):
#					freq_add=driverpacket.centrefreq.append(ctrfreq[fn])

				ack=data_to_driver(driverpacket,socket)
				del driverpacket.samples[:]

				#time.sleep(1)	
			time.sleep(33e-3)
			print "New Sequence!"
			nave=nave+1 # finished a sequence
			int_time=datetime.utcnow()
		print "Number of integrations: ", nave
		integ_n=integ_n+1



main()

