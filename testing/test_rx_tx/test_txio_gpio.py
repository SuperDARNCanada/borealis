import uhd
import sys
import time

class gpio_testing(object):
	"""
	Class for handling gpio tests
	"""

	def __init__(self, address):
		"""
		Constructor for gpio testing class
		:param address: Device address for linked USRP
		"""

		# Create USRP object
		self.usrp = uhd.usrp.MultiUSRP(address)
		print("USRP has", self.usrp.get_num_mboards(), "motherboards on board")
		self.set_all_low("RXA")

	def set_all_low(self, bank):
		"""
		Set all GPIO pins on gpio bank low
		:param bank: String identifying the GPIO bank
		"""
		print("Setting all pins on", bank, "as low outputs.")
		self.usrp.set_gpio_attr(bank, "CTRL", 0x0000, 0b1111111111111111)
		self.usrp.set_gpio_attr(bank, "DDR", 0xffff, 0b1111111111111111)
		self.usrp.set_gpio_attr(bank, "OUT", 0x0000, 0b1111111111111111)

	def set_pulse_time(self, pt=1):
		"""
		Sets the length of the pulse for testing
		:param time: The pulse length in seconds
		"""
		PULSE_TIME = pt


if __name__ == "__main__":
	# Get command line arguments
	# script should be called as: test_txio_gpio.py <device_address>
	argv = sys.argv
	ADDR = argv[1]

	tests = gpio_testing("num_recv_frames=512,num_send_frames=256,send_buff_size=2304000,addr=" + ADDR)
	print("GPIO bank is in", tests.usrp.get_gpio_attr("RXA", "CTRL"), "mode")
	print("GPIO bank set as", tests.usrp.get_gpio_attr("RXA", "DDR"))
	print("GPIO level is", tests.usrp.get_gpio_attr("RXA", "OUT"))

	print("Beginning tests")
	for j in range(10):
		tests.usrp.set_gpio_attr("RXA", "OUT", 0xffff, 0b00100000)
		time.sleep(1)
		tests.usrp.set_gpio_attr("RXA", "OUT", 0x0000, 0b00100000)
		time.sleep(1)

	print("Done!")
