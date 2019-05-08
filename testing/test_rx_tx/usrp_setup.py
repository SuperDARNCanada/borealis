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
		self._options = SetupOptions(config_file)
		self._tx_freq = tx_freq
		self._rx_freq = tx_freq
		self._rx_chans = rx_chans
		self._tx_chans = tx_chans
