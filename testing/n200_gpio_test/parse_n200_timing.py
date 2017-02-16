#! /usr/bin/env python

# Nabbed this class from http://code.activestate.com/recipes/410692/
# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.
class switch(object):
  def __init__(self, value):
    self.value = value
    self.fall = False
  
  def __iter__(self):
    """Return the match method once, then stop"""
    yield self.match
    raise StopIteration
                                  
  def match(self, *args):
    """Indicate whether or not to enter a case suite"""
    if self.fall or not args:
      return True
    elif self.value in args: # changed for v1.5, see below
      self.fall = True
      return True
    else:
      return False


# This script takes in a csv exported file from the Saleae Logic software
# and finds out if the timing passes or fails 
import csv

# Columns in the csv file correspond to certain information. 
# There is a timestamp, a scope sync bit, an atten bit, a tr bit and an rf bit.
time_index = 0
ss_index = 1
a_index = 2
tr_index = 3
rf_index = 4

# The time delays between each of the various events can be programmed, so here we can change them
start_delay_us = 49500            # Delay between scope sync low and scope sync high
scope_sync_pre_delay_us = 0  # Delay between scope sync high and ATTEN high
atten_pre_delay_us = 10       # Delay between ATTEN high and TR high
tr_pre_delay_us = 60           # Delay between TR high and RF pulse high
rf_pulse_time_us = 300        # How long is the RF pulse?
tr_post_delay_us = 40         # Delay between RF low and TR low
atten_post_delay_us = 10       # Delay between TR low and ATTEN low
THRESHOLD_US = 1

# Various items related to scope sync delay
num_ranges = 75
rsep_km = 45
first_range_km = 180
speed_of_light = 299792458.0
scope_sync_post_delay_us = (num_ranges*rsep_km + first_range_km)*1000.0*1000000*2.0/speed_of_light
print("Scope sync post delay: " + str(scope_sync_post_delay_us) + " us")

# Items related to the pulse sequence
tau_us = 1500
#pulse_sequence = [0,14,22,24,27,31,42,43]
pulse_sequence = [0,5,8,12,18,30,32,33]

# State machine
states = ("unknown", "between_sequences", "start_of_sequence", "atten_pre_pulse", "tr_pre_pulse", "pulse", "tr_post_pulse", "atten_post_pulse", "between_pulses", "end_of_sequence")
state = states[0]
print("state is : "+state)

# Get the file and read into memory
test_file = csv.reader(open('logic_export2.csv','r'),delimiter=',')

max_time_error_us = 0.0
timestamp = 0.0
pulse_num = 0
num_pulses = len(pulse_sequence)
for row in test_file:
  timestamp_previous = timestamp
  timestamp = float(row[0])
  # There are 16 possible cases for the 4 bits, so let's just handle them all
  bits_value = (int(row[ss_index]) << ss_index-1)
  bits_value += (int(row[a_index]) << a_index-1)
  bits_value += (int(row[tr_index]) << tr_index-1)
  bits_value += (int(row[rf_index]) << rf_index-1)
  print("state: "+state+" pulse number: " + str(pulse_num))
  print("Bits value : "+str(bits_value))
  for case in switch(bits_value):
    if case(0):
      if state == states[0]:
        state = states[1]
        pulse_num = 0
        #print("Starting sequence")
      elif state == states[9]:
        state = states[1]
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff-scope_sync_post_delay_us/1000000.0)*1000000.0 
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR: " + str(time_error_us) + " us")
        print("Starting downtime between sequences")
        pulse_num = 0
      else:
        print ("ERROR WITH PULSE SEQUENCE, state was "+state+" but we now see between pulse sequences. Resetting")
        state == states[0]
      break

    if case(1):
      # Start of pulse, SS high, either beginning of sequence, between pulses or end of sequence
      if state == states[0]:
        continue
      elif state == states[1]:
        # Starting pulse sequence
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff-start_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR1: " + str(time_error_us) + " us")
        state = states[2]
      elif state == states[7]:
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff-atten_post_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR2: " + str(time_error_us) + " us")
          print("Time diff: " + str(timediff*1000000.0) + " us")
        if pulse_num == num_pulses-1:
          state = states[9]
        else:
          pulse_num += 1
          # Between pulses
          state = states[8]
      else:
        print ("ERROR WITH PULSE SEQUENCE, state was "+state+" but we now see only ss high. Resetting")
        state == states[0]
      break

    if case(2):
      # Error, atten high but nothing else is
      print("Error, only atten is high")
      break
    
    if case(3):
      # Atten and ss high, either before or after TR
      if state == states[0]:
        continue
      elif state == states[2]:
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff-scope_sync_pre_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR1: " + str(time_error_us) + " us")
        # Pre pulse
        state = states[3]
      elif state == states[8]:
        timediff = timestamp - timestamp_previous
        correct_delay_us = tau_us*(pulse_sequence[pulse_num]-pulse_sequence[pulse_num-1]) - tr_post_delay_us - atten_post_delay_us - atten_pre_delay_us - tr_pre_delay_us - rf_pulse_time_us 
        time_error_us = (timediff - correct_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR2: " + str(time_error_us) + " us")
        # Pre pulse
        state = states[3]
      elif state == states[6]:
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff-tr_post_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR3: " + str(time_error_us) + " us")
          print("Time diff: " + str(timediff*1000000.0) + " us")
        # Post pulse
        state = states[7]
      else:
        print ("ERROR WITH PULSE SEQUENCE, state was "+state+" but we now see atten and ss high. Resetting")
        state == states[0]
      break

    if case(4):
      # Error, only tr high
      print "Error, only TR high"
      break
    if case(5):
      # Error, only ss and tr high"
      print "Error, only ss and TR high"
      break
    if case(6):
      # Error
      print "Error, only atten and TR high"
      break
    
    if case(7):
      # Scope sync, atten and tr high - either before or right after a pulse
      if state == states[0]:
        continue 
      elif state == states[3]:
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff - atten_pre_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR1: " + str(time_error_us) + " us")
        state = states[4]
      elif state == states[5]:
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff - rf_pulse_time_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR2: " + str(time_error_us) + " us")
        state = states[6]
      else:
        print ("ERROR WITH PULSE SEQUENCE, state was "+state+" but we now see ss atten and tr high. Resetting")
        state == states[0]
      break
    
    if case(8):
      # Error - only RF signal high
      print "Error, only RF high"
      break
    if case(9):
      # Error, only RF and scope sync high
      print "Error, only RF and scope sync high"
      break
    if case(10):
      # Error, RF and atten high
      print "Error, only RF and atten high"
      break
    if case(11):
      # Error, TR is low when it should be high
      print "Error, TR is low when it should be high"
      break
    if case(12):
      # Error, RF and TR high, but scope sync and atten low
      print "Error, RF and TR high, but scope sync and atten low"
      break
    if case(13):
      # Error, Atten low when it should be high
      print "Error, Atten is low when it should be high"
      break
    if case(14):
      # Error, Scope sync is low when it should be high"
      print "Error, scope sync low when it should be high"
      break
    if case(15):
      if state == states[0]:
        continue
      elif state == states[4]:
        timediff = timestamp - timestamp_previous
        time_error_us = (timediff - tr_pre_delay_us/1000000.0)*1000000.0
        if(abs(time_error_us) > THRESHOLD_US):
          if max_time_error_us < time_error_us:
            max_time_error_us = time_error_us
          print("TIMING ERROR: " + str(time_error_us) + " us")
        # We now go from tr high to pulse 
        state = states[5]
      else:
        # Error
        print("ERROR WITH PULSE SEQUENCE, state was " + state + " but now we see all high. Resetting")
        state = states[0]
      break
    
print("Max timing error: " + str(max_time_error_us) + " us")
