import setup_options.SetupOptions
import numpy as np
import uhd
import math
import os


class USRPSetup(object):
	"""
	Class for handling the setup of a USRP based on
	config file options
	"""
	def __init__(self, config_file, tx_freq, rx_freq, tx_chans, rx_chans):
		"""
		Initializes txio board setup

		:param config_file: path to config file with options
		:param tx_freq: transmission frequency
		:param rx_freq: recieving frequency
		:param tx_chans: transmission channels
		:param rx_chans: reciever channels
		"""
		options = SetupOptions(config_file)
		self._tx_freq = tx_freq
		self._rx_freq = tx_freq
		self._rx_chans = rx_chans
		self._tx_chans = tx_chans

		# create usrp device
		self.usrp = uhd.usrp.MultiUSRP(options.devices())

	def set_usrp_clock_source(source):
		"""
		Sets the clock source on the usrp
		:param source: string representing the clock source
		"""
		self.usrp.set_clock_source(source)

	def set_tx_subdev(tx_subdev_str):
		"""
		Sets the subdevice for handling transmissions
		:param tx_subdev_str: A string specifying the subdevice
		"""
		tx_subdev = uhd.SubdevSpec(tx_subdev_str)
		self.usrp.set_tx_subdev_spec(tx_subdev)

	def set_tx_rate(tx_rate):
		"""
		Sets the transmission rate for specified transmission channels
		:param tx_rate: The desired data rate for transmission
		"""
		self.usrp.set_tx_rate(tx_rate)

	def get_tx_rate(channel):
		"""
		Gets the actual tx rate being used on a transmission channel
		:param channel: an integer representing the channel number
		"""
		return np.uint32(self.usrp.get_tx_rate(channel))

	def set_tx_center_freq(freq, chans):
		"""
		Tunes the usrp to the desired center frequency
		:param freq: The desired frequency in Hz
		:param chans: The channels to be tuned
		"""
		tx_tune_request = uhd.types.TuneRequest(freq):
		for channel in chans:
			self.usrp.set_tx_freq(tx_tune_request, channel)
			actual_freq = self.usrp.get_tx_freq(channel)
			if not (actual_freq == freq):
				print("Requested tx center frequency:", freq, "actual frequency:", actual_freq, "\n")

	def setup_tx_stream(cpu, otw, chans):
		"""
		Sets up the tx stream object based on given streaming options
		:param cpu: The host cpu format as a string
		:param otw: The over the wire format as a string
		:param chans: Desired transmission channels
		"""
		tx_stream_args = usrp.StreamArgs(cpu, otw)
		tx_stream_args.channels = chans
		tx_stream = usrp.get_tx_stream
		return tx_stream

	
