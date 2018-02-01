USRP N200 Driver
****************

The N200 driver is a C++ application that controls the operation of the USRP N200 transceivers. The driver is responsible for using Ettus' UHD software to configure a `multi-USRP device <https://files.ettus.com/manual/classuhd_1_1usrp_1_1multi__usrp.html>`_ and configure the device for SuperDARN operation.

As part of the driver, a C++ class was written to abstract the configuration of the N200s. The driver configures the N200s using certain options from the config file as well as options related to the experiment. All runtime options and control are defined by the Radar Control module.

The driver consists of the main function and three worker threads. The main function is responsible for instantiating a USRP object, and configuring some initial runtime options such as which physical devices to use, the GPIO bank, the timing signal masks, the clock source, the subdevs for TX and RX, and the time source. These options are configured once at runtime and then not changed during operation. The main function then starts the control, transmit, and receive worker threads.

==============
Control Thread
==============

The control thread accepts driver packets from Radar Control. Each packet contains information on how to control the N200s. All pulse driver packets for a sequence are sent at once, but processed in the order they come in. The start and end packets of a sequence are defined by a start of burst(SOB) and end of burst(EOB) flag to match the similar UHD nomenclature when beginning and ending transmission in burst mode(the mode used for pulse transmission). Each packet is deserialized to check for some basic errors. Provided there are no errors, the packets are then forwarded to the transmitting and receive worker threads.

When the driver finishes processing a sequence the control thread first forwards the metadata for the collected samples to Rx Signal Processing followed by an acknowledgment sent to Radar Control.

===============
Transmit Thread
===============

On a driver packet indicating the start of a new sequence(SOB is true), the transmit thread will configure some multi-USRP parameters such as what TX channels(antennas) to use, the TX center frequency, and the buffer of samples to send as a pulse. The driver requires these all be set once but can be omitted in future sequences if they are repeated. No need to continually serialize and deserialize duplicated information. Each driver packet in the sequence contains a relative time from the start of the sequence to when the pulse should be transmitted. If SOB is true, then a time zero is created by using the UHD current time as a reference to when pulses should start. A slight delay is added to allow for some CPU time to finish configuring the pulse. Again, if SOB is true, then the time zero value is sent to the receive thread as the time to begin receiving. Once the pulse time relative to time zero is calculated, the multi-USRP object is configured to send the pulse samples at that time.

TR switching signals are generated using the `USRP ATR <https://files.ettus.com/manual/classuhd_1_1usrp_1_1multi__usrp.html#a57f25d118d20311aca261e6dd252625e>`_ functionality. The ATR pins are only triggered exactly when the USRP is sending or receiving, so in order to properly window the RF signal, zeros are padded to the start and end of the signal. From testing, the zeros do not create any issues such as higher noise, etc. They purely allow us to create a window for TR signals. The actual TR signal is ATR_XX. We are receiving during the whole sequence, so the full-duplex pin is the pin that goes high when we are transmitting while receiving. Scope sync is mimicked by looking at the inverted ATR_0X signal since we are never idle during a sequence in normal operation. If in a receive only state, scope sync can be mimicked by using ATR_RX. The current version of borealis does not allow for transmitting only.

When final pulse in the sequence(EOB is true) is sent, an acknowledgment of the sequence is generated to be sent back to Radar Control via the control thread.

==============
Receive Thread
==============

Unlike the transmit thread, the receive thread only does work when the SOB flag is true. The multi-USRP RX center frequency is set if it exists. The driver requires this be set at least once, but then can be ignored if duplicated in successive runs. The receiver is configured to receive on all channels for which there are antennas. If any channels are to be unused, it is easier to ignore them in the Rx Signal Processing module.

Once the multi-USRP is configured, a shared memory handler is created. Received samples are put directly into shared memory that can be accessed by both the driver and Rx Signal Processing. This minimizes the amount of interprocess copying needed. Once the shared memory is created, pointer offsets into the shared memory are calculated for where each channel buffer begins.

The receive thread grabs the sequence start time(time zero) from the transmit time and then configures the N200s to begin receiving a number of samples defined by Radar Control into the buffers calculated earlier. Once all samples are collected, the receive thread sends the metadata for the collected samples to Rx Signal Processing via the control thread. This metadata includes the shared memory region name, the sequence number, and the number of receive samples per buffer.





.. toctree::
   :glob:

   file/n200__driver_8cpp.rst
   file/usrp_8hpp.rst



