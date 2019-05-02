import numpy as np
import uhd.usrp.MultiUSRP as USRP
import math as math

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
	amp = 1.0/math.sqrt(2.0)
	pulse_len = 300.0E-6
	
	tr_start_pad = 60
	tr_end_pad = 60
	