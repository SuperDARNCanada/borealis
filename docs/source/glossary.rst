========
Glossary
========

array
	In SuperDARN data, the array data refers to the data after it has been beamformed and all
	antennas are combined into one array dataset. Typically the SuperDARN antennas are divided into
	the main antenna array and one interferometer antenna array.

averaging period
	A time during which sequences are transmitted repeatedly with the intent to average the received
	samples together. Averaging period is often used interchangeably with integration time.

channel
	This term is often used to denote frequency channels, but in USRPs it is also often used to
	denote the different transmit and receive physical ports, in which case for SuperDARN the
	different USRP channels would denote different antennas. We have tried to avoid the use of this
	term due to the ambiguity.

device
	When using Ettus UHD API this refers to the radio devices, or the N200s in the case of Borealis.
	When in the context of CUDA, this refers to the GPU.


integration time
	The time allocated for an averaging period. An averaging period can be defined by the
	integration time (during which as many sequences as possible are transmitted); or simply by the
	number of sequences to transmit for the averaging period. Integration time is often used
	interchangeably with averaging period.

host
	A local machine; for Borealis this is the Borealis computer.

nave
	number of averages; equivalent to number of sequences transmitted or number of sampling periods
	received.

record
	A recorded subset of data. In SuperDARN data, a record contains all data for an integration
	time, and in the rawacf data the data is already averaged from the integration time.

sampling period
	The receive sampling time allocated to a transmitted sequence.

sequence
	A pulse sequence to be transmitted. Each sequence has a sampling period, which extends past the
	length of the pulse sequence for some time dependent on the number of ranges to be sampled.

slice
	A slice, in the context of experiments run by Borealis, refers to an individual component of an
	experiment. An experiment can have one or many slices. See the documentation on Building an
	Experiment for more information on slices, as they are an integral part of the Borealis system.
