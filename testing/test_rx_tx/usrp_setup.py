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
		:param tx_freq: transmission center frequency
		:param rx_freq: recieving center frequency
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
		tx_subdev = uhd.usrp.SubdevSpec(tx_subdev_str)
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
				print("Requested tx center frequency:", freq, "actual frequency:",
						actual_freq, "\n")

	def get_tx_center_freq(chan):
		"""
		Gets the center frequency for a specified channel
		:param chan: The channel at which to retrieve the center frequency
		"""
		return self.usrp.get_tx_freq(chan)


	def create_tx_stream(cpu, otw, chans):
		"""
		Sets up the tx streamer object based on given streaming options
		:param cpu: The host cpu format as a string
		:param otw: The over the wire format as a string
		:param chans: Desired transmission channels
		"""
		tx_stream_args = uhd.usrp.StreamArgs(cpu, otw)
		tx_stream_args.channels = chans
		tx_stream = self.usrp.get_tx_stream(tx_stream_args)
		return tx_stream

	def set_main_rx_subdev(main_subdev):
		"""
		Sets up the subdevice for the main reciever
		:param main_subdev: String representing the subdevice(s) for the main reciever
		"""
		rx_subdev = uhd.usrp.SubdevSpec(main_subdev)
		self.usrp.set_rx_subdev_spec(rx_subdev)

	def set_rx_rate(rx_rate):
		"""
		Sets the data rate for the reciever
		:param rx_rate: The reciever data rate
		"""
		self.usrp.set_rx_rate(rx_rate)

	def get_rx_rate(channel):
		"""
		Gets the reciever rate on a specified channel
		:param channel: The desired channel
		"""
		return np.uint32(self.usrp.get_rx_rate(channel))

	def set_rx_center_freq(freq, chans):
		"""
		Tunes the reciever to a desired frequency
		:param freq: The desired reciever center frequency
		:param chans: The channels to tune to freq
		"""
		rx_tune_request = uhd.types.TuneRequest(freq)
		for channel in chans:
			self.usrp.set_rx_freq(rx_tune_request, channel)
			actual_freq = self.usrp.get_rx_freq(channel)
			if not (actual_freq == freq):
				print("Requested rx center frequency:", freq, "actual frequency:",
						actual_freq, "\n")
	def get_rx_center_freq(chan):
		"""
		Gets the center frequency for a specified channel
		:param chan: The channel at which to retrieve the center frequency
		"""
		return self.usrp.get_rx_freq(chan)

	def create_rx_stream(cpu, otw, chans):
		"""
		Sets up an rx streaming object based on given options
		:par
		:param cpu: The host cpu format as a string
		:param otw: The over the wire format as a string
		:param chans: Desired receiving channels
		"""
		rx_stream_args = uhd.usrp.StreamArgs(cpu, otw)
		rx_stream_args.channel = chans
		rx_stream = self.usrp.get_rx_stream(rx_stream_args)
		return rx_stream

	def setup_gpio(gpio_bank):
		"""
		Configures the given gpio bank for the txio
		:param gpio_bank: String representing the gpio bank
		"""
		for i in np.arange(self.usrp.get_num_mboards()):
			self.usrp.set_gpio_attr(gpio_bank, "CTRL", 0xFFFF, 0b11111111, i)
			self.usrp.set_gpio_attr(gpio_bank, "DDR", 0xFFFF, 0b11111111, i)

			# Mirror pins along bank for easier scoping
			usrp_d.set_gpio_attr(gpio_bank, "ATR_RX", 0xFFFF, 0b000000010, i)
			usrp_d.set_gpio_attr(gpio_bank, "ATR_RX", 0xFFFF, 0b000000100, i)

			usrp_d.set_gpio_attr(gpio_bank, "ATR_TX", 0xFFFF, 0b000001000, i)
			usrp_d.set_gpio_attr(gpio_bank, "ATR_TX", 0xFFFF, 0b000010000, i)

			#XX is the actual TR signal
			usrp_d.set_gpio_attr(gpio_bank, "ATR_XX", 0xFFFF, 0b000100000, i)
			usrp_d.set_gpio_attr(gpio_bank, "ATR_XX", 0xFFFF, 0b001000000, i)

			#0X acts as 'scope sync'
			usrp_d.set_gpio_attr(gpio_bank, "ATR_0X", 0xFFFF, 0b010000000, i)
			usrp_d.set_gpio_attr(gpio_bank, "ATR_0X", 0xFFFF, 0b100000000, i)
