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
		self.PULSE_TIME = 0.5
		self._usrp = uhd.usrp.MultiUSRP(address)
		print("USRP has", self._usrp.get_num_mboards(), "motherboards on board")
		self._rxo_pin = int(input("Specify RXO GPIO pin "))
		self._txo_pin = int(input("Specify TXO GPIO pin "))
		self._tr_pin = int(input("Specify T/R GPIO pin "))
		self._idle_pin = int(input("Specify IDLE GPIO pin "))
		self._tm_pin = int(input("Specify TEST_MODE GPIO pin "))


	def set_all_low(self):
		"""
		Set all GPIO pins on gpio bank low
		:param bank: String identifying the GPIO bank
		"""
		# for i in arange(self._usrp.get_num_mboards()):
		print("Setting all pins on", self._bank, "as low outputs.")
		self._usrp.set_gpio_attr(self._bank, "CTRL", 0x0000, 0b1111111111111111)
		self._usrp.set_gpio_attr(self._bank, "DDR", 0xffff, 0b1111111111111111)
		self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, 0b1111111111111111)

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

	def run_single_signals_test(self):
		"""
		Handles testing of rxo pin
		:param function: String specifying the function you wish to test
						 Options: "RXO"
						 		  "TXO"
						 		  "TX/RX"
						 		  "IDLE"
						 		  "TEST_MODE"
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
					time.sleep(self.PULSE_TIME)
					self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, mask)
					time.sleep(self.PULSE_TIME)
			except KeyboardInterrupt:
				user = input("\nContinue? [y/n] ")
				if user == "y":
					print("Going to next test...")
					pass
				elif user == "n":
					print("Ending tests...")
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
