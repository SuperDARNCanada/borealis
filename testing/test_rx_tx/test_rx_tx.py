import numpy as np
import uhd.usrp as usrp
import uhd
import math
import cmath
import sys
import time

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
	samples = np.zeros(num_samps_per_antenna, dtype=np.complex64)

	# build the pulse one sample at a time
	for i in np.arange(tr_start_pad, num_samps_per_antenna-tr_end_pad):
			nco_point = np.complex64(0.0)
		for freq in tx_freqs:
			sampling_freq = 2 * math.pi * freq / tx_rate

			radians = math.fmod(sampling_freq * i, 2 * math.pi)
			I = amp * math.cos(radians)
			Q = amp * math.sin(radians)

			nco_point += complex(I, Q)

		samples[i] = nco_point

	ramp_size = int(10E-6 * tx_rate)

	k = 0
	for i in np.arange(tr_start_pad, tr_start_pad+ramp_size):
		a = k / ramp_size
		samples[j] *= complex(a, 0.0)
		k += 1

	k = 0
	for j in np.arange(num_samps_per_antenna-tr_end_pad-1, num_samps_per_antenna-tr_end_pad-1-ramp_size, -1):
		a = k / ramp_size
		samples[j] *= complex(a, 0.0)

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

	# create buffer
	usrp_buffer_size = 100 * rx_stream.get_max_num_samps()
	ringbuffer_size = (sys.getsizeof(500.0E6)/sys.getsizeof(complex(float))/usrp_buffer_size) * 
									usrp_buffer_size
	recv_buffer = np.empty((rx_chans.size(), ringbuffer_size))

	# make stream command
	rx_stream_cmd = uhd.types.StreamCMD(uhd.types.STREAMCMD.start_cont)
	rx_stream_cmd.stream_now = False
	rx_stream_cmd.num_samps = 0
	rx_stream_cmd.time_spec = usrp_d.get_time_now() + uhd.TimeSpec(DELAY)

	# timing stuff
	stream_start = time.time()
	rx_stream.issue_stream_cmd(rx_stream_cmd)
	stream_end = time.time()

	# set up metadata and statistics counters
	meta = uhd.types.RXMetadata()

	buffer_inc = 0
	timeout_count = 0
	overflow_count = 0
	overflow_oos_count = 0
	late_count = 0
	bchain_count = 0
	align_count = 0
	badp_count = 0
	first_time = True

	# execute testing
	test_trials = 0
	test_while = True
	while(test_while):
		num_rx_samples = rx_stream.recv(recv_buffer, usrp_buffer_size, meta, 3.0)
		print("Recv", num_rx_samples, "samples", "\n")
		print("On ringbuffer idx", usrp_buffer_size * buffer_inc, "\n")

		error_code = meta.error_code
		print("RX TIME:", meta.time_spec.get_real_secs(), "\n")
		if first_time:
			start_time = meta.time_spec
			start_tx = True
			first_time = False

		# handle errors
		if error_code == uhd.types.RXMetadataErrorCode.none:
			pass
		elif error_code == uhd.types.RXMetadataErrorCode.timeout:
			print("Timed out!\n")
			timeout_count += 1
		elif error_code == uhd.types.RXMetadataErrorCode.overflow:
			print("Overflow!\n")
			print("OOS:" meta.out_of_sequence, "\n")
			if meta.out_of_sequence:
				overflow_oos_count += 1
			overflow_count += 1
		elif error_code == uhd.types.RXMetadataErrorCode.late:
			print("Late!\n")
			late_count += 1
		elif error_code == uhd.types.RXMetadataErrorCode.broken_chain:
			print("Broken Chain!\n")
			bchain_count += 1
		elif error_code == uhd.types.RXMetadataErrorCode.alignment:
			print("Alignment!\n")
			align_count += 1
		elif error_code == uhd.types.RXMetadataErrorCode.bad_packet:
			print("Bad Packet!")
			badp_count += 1

		# Print results
		print("Timeout count:", timeout_count, "\n")
		print("Overflow count:", overflow_count, "\n")
		print("Overflow OOS count:", overflow_oos_countm "\n")
		print("Late count:", late_count, "\n")
		print("Broken chain count", bchain_count, "\n")
		print("Alignment count", align_count, "\n")
		print("Bad packet count", badp_count, "\n")

		if not test_mode == "full":
			test_trials += 1
		if test_trials == 10
			test_while = 0
			test_trials = 0