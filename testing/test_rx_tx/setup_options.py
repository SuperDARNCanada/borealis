"""
Class SetupOptions parses options from a configuration file and
provideds methods for setting and retrieving options critical
to txio board and USRP device configuration
"""
import json
import os


class SetupOptions(object):
	"""
	Parses options from the config file relevant to txio setup
	"""
	def __init__(self, filepath: str):
		"""
		Constructor for option parser for board setup
		:param filepath: Path to config file
		"""
		try:
			with open(filepath, 'r') as config_data:
				raw_config = json.load(config_data)
		except IOError:
			errmsg = 'Cannon open config file at {}'.format(config_path)
			raise IOError(errmsg)

		self._devices = raw_config['devices']
		self._tx_subdev = raw_config['tx_subdev']
		self._tx_sample_rate = float(raw_config['tx_sample_rate'])
		self._main_rx_subdev = raw_config['main_rx_subdev']
		self._interferometer_rx_subdev = raw_config['interferometer_rx_subdev']
		self._rx_sample_rate = float(raw_config['rx_sample_rate'])
		self._pps = raw_config['pps']
		self._ref = raw_config['ref']
		self._otw = raw_config['overthewire']
		self._cpu = raw_config['cpu']
		self._gpio_bank = raw_config['gpio_bank']

	@property
	def get_devices(self):
		"""
		Get the address of the devices to be used

		:returns: The device address
		:rtype: str
		"""
		return self._devices
	
	@property
	def get_tx_subdev(self):
		"""
		Gets the subdevice for the transmit function

		:returns: The subdevice for the transmit function
		:rtype: str
		"""
		return self._tx_subdev
	
	@property
	def get_tx_sample_rate(self):
		"""
		Gets the transmit sample rate

		:returns: The transmit sample rate
		:rtype: float
		"""
		return self._tx_sample_rate
	
	@property
	def get_main_rx_subdev(self):
		"""
		Gets the subdevice for the main reciever

		:returns: The subdevice for the main reciever
		:rtype: str
		"""
		return self._main_rx_subdev
	
	@property
	def get_interferometer_rx_subdev(self):
		"""
		Gets the subdevice for the interferometer reciever

		:returns: The subdevice for the interferometer reciever
		:rtype: str
		"""
		return self._interferometer_rx_subdev
	
	@property
	def get_rx_sample_rate(self):
		"""
		Gets the reciever sample rate

		:returns: The sample rate for the reciever
		:rtype: float
		"""
		return self._rx_sample_rate
	
	@property
	def get_pps(self):
		"""
		Gets the clock source

		:returns: The clock source
		:rtype: str
		"""
		return self._pps
	
	@property
	def get_ref(self):
		"""
		Gets the reference clock specification

		:returns: The reference clock
		:rtype: str
		"""
		return self._ref
	
	@property
	def get_otw(self):
		"""
		Gets the over the wire format for communication

		:returns: The over the wire format
		:rtype: str
		"""
		return self._otw
	
	@property
	def get_cpu(self):
		"""
		Gets the cpu format for communication

		:returns: The cpu format
		:rtype: str
		"""
		return self._cpu
	
	@property
	def get_gpio_bank(self):
		"""
		Gets the gpio bank for formatting

		:returns: The gpio bank
		:rtype: str
		"""
		return self._gpio_bank
	