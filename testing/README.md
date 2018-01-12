# Testing #
This directory contains various programs, scripts, and test data used during the development of borealis.

## n200_gpio_test ##
This directory contains a c++ program to run gpio tests on the n200's as well as python code to use a saleae logic analyzer's output file to test for proper gpio signal timing.

## n200_test_packet ##	
This directory contains a c++ program to act like an experiment and talk to the n200 driver code in borealis. It sends pulse sequences but does not include acks.

## octoclock_test	## 
This directory contains a c++ program to explore all the functionality of the ettus octoclocks, with or without internal GPSDOs in them.

## parallel_reduce ##	
Test parallel reduce algorithm in cuda, as well as implementations of the algorithm in python.

## rx_tx_delay_test	##
There is an offset between the rx and tx chains in the USRPs. It is on the order of microseconds and is related to frequency, sampling rate and bandwidth. One person (Max Scharrenbroich) on the USRP-users mailing list has implemented a leading edge detect code to find these offsets.
See: http://lists.ettus.com/pipermail/usrp-users_lists.ettus.com/2013-September/007500.html
This is due to the reference points for time being different in TX and RX. In TX, it is before the DSP section, in RX it is after the DSP section.
This direcotry has code written by Max to calculate the offsets for combinations of frequency and sampling rates.

## sig_proc_test ##

This directory contains test code for decimations using cuda.
