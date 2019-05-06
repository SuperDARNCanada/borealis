import numpy as np
import uhd.usrp as usrp
import math
import cmath

# Define constants
ADDR = "num_recv_frames=512,num_send_frames=256,send_buff_size=2304000"
CLKSRC = "internal"

RXSUBDEV = "A:A A:B"
TXSUBDEV = "A:A"

TXCHAN = [0]
RXCHAN = [0, 1]

TXRATE = 250.0E3
RXRATE = 250.0E3
TXFREQ = 11E6
RXFREQ = 11E6
DELAY = 10E-3

PULSETIMES = [0.0, 13500E-3, 18000E-6, 30000E-6, 33000E-6, 39000E-6, 40500E-6]

start_tx = False

# Make ramped pulses
def make_ramped_pulse(double: tx_rate):
	"""
	Purpose:
		Create a ramped pulse to be used during testing
	Pre-conditions:
		tx_rate: the data transfer rate of the device, double format
	Return:
		a list of complex numbers representing the generated pulse
	"""
	# define amplitude and pulse length
	amp = 1.0/math.sqrt(2.0)
	pulse_len = 300.0E-6

	tr_start_pad = 60
	tr_end_pad = 60
	num_samps_per_antenna = np.ceil(pulse_len * (tx_rate + tr_start_pad + tr_end_pad))
	tx_freqs = [1E6]

	# initialize samples list
	default_v = complex(0, 0)
	samples = [default_v] * num_samps_per_antenna

	# build the pulse one sample at a time
	for i in range(tr_start_pad, num_samps_per_antenna-tr_end_pad):
			nco_point = complex(0, 0)
		for freq in tx_freqs:
			sampling_freq = 2 * math.pi * freq / tx_rate

			radians = math.fmod(sampling_freq * i, 2 * math.pi)
			I = amp * math.cos(radians)
			Q = amp * math.sin(radians)

			nco_point += complex(I, Q)

		samples[i] = nco_point

	ramp_size = int(10E-6 * tx_rate)

	k = 0
	for i in range(tr_start_pad, tr_start_pad+ramp_size):
		a = k / ramp_size
		samples[j] *= complex(a, 0)
		k += 1

	k = 0
	for j in range(num_samps_per_antenna-tr_end_pad-1, num_samps_per_antenna-tr_end_pad-1-ramp_size, -1):
		a = k / ramp_size
		samples[j] *= complex(a, 0)

	return samples

# RX THREAD
def recv(usrp.MultiUSRP: usrp_d, list: rx_chans):
	"""
	Function defines the operation of the recieve thread for the USRP
	:param usrp_d: The MultiUSRP object
	:param rx_chans: A list specifying the reciever channels
	"""
	# create the stream args object
	rx_stream_args = usrp.StreamArgs("fc32", "sc16")
	# set channel list
	rx_stream_args.channels = rx_chans
	# set rx streamer
	rx_stream = usrp_d.get_rx_stream(rx_stream_args)

	usrp_buffer_size = 100 * rx_stream.get_max_num_samps()
