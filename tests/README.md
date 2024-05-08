# Tools #
This directory contains various tools: programs, scripts, and test data used during the development of borealis.

# SIMULATORS #

## n200_test_packet ##
This directory contains a c++ program to act like an experiment and talk to the n200 driver code in borealis. It sends pulse sequences but does not include acks.

## radar_control_input_sim ##
A python program to act like the driver and dsp modules for radar_control.

## dsp_input_sim ##
TODO: Keith

## brian_testing ##
A communication model for brian. The system without the data processing and data packets.

# BOREALIS_TESTS #

## testing_utils ##
Contains functions and utilities for testing and plotting Borealis data and building decimation schemes for Borealis.

## data_test_scripts ##
Full scripts dedicated to testing loopback and other data.

## filter_testing ##
Scripts for building and plotting and testing filters.

# NTP #
Contains test NTP statistics files for plotting, as well as a python NTP plotting utility.

# OTHER #

## n200_gpio_test ##
This directory contains a c++ program to run gpio tests on the n200's as well as python code to use a saleae logic analyzer's output file to test for proper gpio signal timing.

## octoclock_test	##
This directory contains a c++ program to explore all the functionality of the ettus octoclocks, with or without internal GPSDOs in them.

## parallel_reduce ##
Test parallel reduce algorithm in cuda, as well as implementations of the algorithm in python.

## rx_tx_delay_test	##
There is an offset between the rx and tx chains in the USRPs. It is on the order of microseconds and is related to frequency, sampling rate and bandwidth. One person (Max Scharrenbroich) on the USRP-users mailing list has implemented a leading edge detect code to find these offsets.
See: http://lists.ettus.com/pipermail/usrp-users_lists.ettus.com/2013-September/007500.html
This is due to the reference points for time being different in TX and RX. In TX, it is before the DSP section, in RX it is after the DSP section.
This directory has code written by Max to calculate the offsets for combinations of frequency and sampling rates.

## dsp_testing ##

This directory houses some tests which can be used as a testbench for DSP development.

### decimate.cu ###

Several implementations of CUDA kernels for simultaneous filtering and downsampling.

### decimate_single_core.cpp ###

Standalone C++ script to test simultaneous filtering and downsampling on the CPU.

### dsp_analyze.py ###

This script contains functionality for plotting rf data written to file in an ascii format.

### rx_signal_processing_tests ###

This directory houses several C++ and CUDA files which mimic the rx_signal_processing C++ and CUDA files
as closely as possible, without any dependency on other modules. These files depend on the
core DSP files (borealis/rx_signal_processing/decimate.cu and .../filtering.cu), and as such
provide a test bench for any development on these files. Simulated data is generated with
rx_dsp_chain.cu, then passed into dsp_testing.cu which operates as closely to
borealis/rx_signal_processing/dsp.cu as possible, without any protobufs and without doing any
beamforming or correlating. The filter taps and data after each stage of filtering/downsampling
are saved to csv files for analysis.
