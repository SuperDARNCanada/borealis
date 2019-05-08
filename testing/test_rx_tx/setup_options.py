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
	def tx_subdev(self):
		"""
		Gets the subdevice for the transmit function

		:returns: The subdevice for the transmit function
		:rtype: str
		"""
		return self._tx_subdev
	
	@property
	def tx_sample_rate(self):
		"""
		Gets the transmit sample rate

		:returns: The transmit sample rate
		:rtype: float
		"""
		return self._tx_sample_rate
	
	@property
	def main_rx_subdev(self):
		"""
		Gets the subdevice for the main reciever

		:returns: The subdevice for the main reciever
		:rtype: str
		"""
		return self._main_rx_subdev
	