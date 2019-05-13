import uhd
import sys
import time
from numpy import arange

class gpio_testing(object):
	"""
	Class for handling gpio tests
	"""

	def __init__(self, address, bank):
		"""
		Constructor for gpio testing class
		:param address: Device address for linked USRP
		"""

		# Create USRP object
		self._bank = bank
		self.PULSE_TIME = 2
		self._usrp = uhd.usrp.MultiUSRP(address)
		print("USRP has", self._usrp.get_num_mboards(), "motherboards on board")
		pinout = input("Set pinout: 'user' for user or enter for default ")
		if pinout == 'user':
			self._rxo_pin = int(input("Specify pin connected to the RXO signal "))
			self._txo_pin = int(input("Specify pin connected to the TXO signal "))
			self._tr_pin = int(input("Specify pin connected to the T/R signal "))
			self._idle_pin = int(input("Specify pin connected to the IDLE signal "))
			self._tm_pin = int(input("Specify pin connected to the TEST_MODE signal "))
			self._lp_pin = int(input("Specify pin connected to the LOW_PWR signal "))
			self._agc_pin = int(input("Specify pin connected to the AGC_STATUS signal "))
		else:
			self._rxo_pin = 1
			self._txo_pin = 3
			self._tr_pin = 5
			self._idle_pin = 7
			self._tm_pin = 9
			self._lp_pin = 11
			self._agc_pin =  13


	def set_all_low(self):
		"""
		Set all GPIO pins on gpio bank low
		:param bank: String identifying the GPIO bank
		"""
		# for i in arange(self._usrp.get_num_mboards()):
		print("Setting all pins on", self._bank, "as low outputs.")
		self._usrp.set_gpio_attr(self._bank, "CTRL", 0x0000, 0b11111111)
		self._usrp.set_gpio_attr(self._bank, "DDR", 0xffff, 0b11111111)
		self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, 0b11111111)

	def set_pulse_time(self, pt):
		"""
		Sets the length of the pulse for testing
		:param time: The pulse length in seconds
		"""
		self.PULSE_TIME = pt

	def get_pin_mask(self, pin):
		"""
		Converts a pin number to a pin mask usable by usrp methods
		:param pin: GPIO pin number
		"""
		mask = 2 ** pin
		return mask

	def query_user(self):
		"""
		Gets user input for continuing or aborting testing
		"""
		user = input("\nContinue? [y/n] ")
		if user == "y":
			print("Going to next test...\n")
			return "y"
		elif user == "n":
			print("Ending tests...\n")
			return "n"
		else:
			print("Invalid input\n")
			return self.query_user()

	def run_single_signals_test(self):
		"""
		Handles the single ended output signals test
		"""
		# Select test
		signals = [("RXO", self._rxo_pin), ("TXO", self._txo_pin), ("T/R", self._tr_pin), ("IDLE", self._idle_pin), ("TEST_MODE", self._tm_pin)]

		for signal, pin in signals:
			mask = self.get_pin_mask(pin)
			self.set_all_low()
			print("Testing", signal)
			try:
				while True:
					self._usrp.set_gpio_attr(self._bank, "OUT", 0xffff, mask)
					# print(self._usrp.get_gpio_attr(self._bank, "READBACK"))
					time.sleep(self.PULSE_TIME)
					self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, mask)
					# print(self._usrp.get_gpio_attr(self._bank, "READBACK"))
					time.sleep(self.PULSE_TIME)
			except KeyboardInterrupt:
				user = self.query_user()
				if user == "y":
					continue
				else:
					return

	def run_differential_signal_test(self):
		"""
		Handles the loop-back differential signals test
		"""
		lp_mask = self.get_pin_mask(self._lp_pin)
		agc_mask = self.get_pin_mask(self._agc_pin)

		def set_lp_agc_inputs():
			"""
			sets pins connected to low power and AGC status signals as inputs
			"""
			self._usrp.set_gpio_attr(self._bank, "DDR", 0x0000, lp_mask)
			self._usrp.set_gpio_attr(self._bank, "DDR", 0x0000, agc_mask)

		for pin in [self._tr_pin, self._tm_pin]:
			# configure GPIO for testing
			self.set_all_low()
			set_lp_agc_inputs()
			mask = self.get_pin_mask(pin)
			try:
				while True:
					self._usrp.set_gpio_attr(self._bank, "OUT", 0xffff, mask)
					time.sleep(self.PULSE_TIME)
					self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, mask)
					time.sleep(self.PULSE_TIME)
			except KeyboardInterrupt:
				user = self.query_user()
				if user == "y":
					continue
				else:
					return


if __name__ == "__main__":
	# Get command line arguments
	# script should be called as: test_txio_gpio.py <device_address>
	argv = sys.argv
	ADDR = argv[1]

	tests = gpio_testing("num_recv_frames=512,num_send_frames=256,send_buff_size=2304000,addr=" + ADDR, "RXA")

	print("Beginning tests")
	tests.run_single_signals_test()
	print("Done!")
