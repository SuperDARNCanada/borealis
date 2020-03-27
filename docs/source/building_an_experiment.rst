**********************
Building an Experiment
**********************

Borealis has an extensive set of features and this means that experiments can end up having lots of functionality and complicated functions. To help organize writing of experiments, we've designed the system so that experiments can be broken into smaller components, called slices, that interface together with other components to perform desired functionality. An experiment can have a single slice or several working together, depending on the complexity.

Each slice contains the information needed about a specific pulse sequence to run. The parameters of a slice contain features such as pulse sequence, frequency, fundamental time lag spacing, etc. These are the parameters that researchers will be familiar with. Each slice can be an experiment on its
own, or can be just a piece of a larger experiment. 

What are slices? 
----------------

Slices are software objects made for the Borealis system that allow easy integration of 
multiple modes into a single experiment. Each slice could be an experiment on its own, and 
averaged products are produced from each slice individually. Slices can be used to create 
separate frequency channels, separate pulse sequences, separate beam scanning order, 
etc. that can run simultaneously. Slices can be interfaced in four different ways. 
 
The following parameters are unique to a slice:  

* tx or rx frequency
* pulse sequence
* tau spacing (mpinc)
* pulse length
* number of range gates
* first range gate
* beam directions
* beam order

The other necessary part of an experiment is specifying how slices will interface with each other. Interfacing in this case refers to how these two components are meant to be run. To understand the interfacing, lets first understand the basic building blocks of a SuperDARN experiment. These are:

**Sequence (integration)**  

Made up of pulses with a specific spacing, at a specific frequency, and with a specified receive time 
following the transmission (to gather information from the number of ranges specified).

**Averaging period (integration time)**  

A time where the sequences are repeated to gather enough information to average and reduce the effect of 
spurious emissions on the data. These are defined by either number of sequences, or a length of time during 
which as many sequences as possible are transmitted.

**Scan**  

A time where the averaging periods are repeated, often with the pulses mixed to look in different beam 
directions with each averaging period. A scan is defined by the number of beams or integration times.

Interfacing types
-----------------

Knowing the basic building blocks of a SuperDARN-style experiment, the following types of interfacing are possible:

**1. SCAN**   

The scan by scan interfacing allows for slices to run a scan of one slice, followed by a scan of the second. The scan mode of interfacing typically means that the slice will cycle through all of its beams before switching to another slice.

**2. INTTIME**   

This type of interfacing allows for an integration period to run for one slice, before switching to another. This type of interface effectively creates an interleaving scan where the scans for different slices are run simultaneously, however the pulse sequences are alternated integration time by 
integration time rather than run concurrently.

**3. INTEGRATION**   

Integration interfacing allows for pulse sequences defined in the slices to alternate sequence by sequence each other within an integration period. Slices which are interfaced in this manner must share the same INTT and INTN values for this to work. It's important to remember that each sequence 
only averages with sequences from the same slice. 

**4. PULSE**   

Pulse interfacing allows for pulse sequences to be run together concurrently. Slices will have their pulse sequences mixed and layered together so that the data transmits at the same time. Slices of different frequencies can be 
mixed simultaneously and slices of different pulse sequences can also run together at the cost of having more blanked samples.

Let's look at some examples of common experiments that can easily be separated into multiple slices. 

In a CUTLASS-style experiment, the pulse in the sequence is actually two pulses of differing transmit frequency. This is a 'quasi'-simultaneous multi-frequency experiment. To build this experiment, two slices can be PULSE interfaced. The pulses from both slices are combined into a single sequence and data from those integrations are used for both slices (filtering the raw data separates the frequencies). 

.. image:: img/cutlass.png
   :width: 800px
   :alt: CUTLASS-style experiment slice interfacing 
   :align: center

In a themisscan experiment, a single beam is interleaved with a full scan. The beam_order can be unique to different slices, and these slices could be INTTIME interfaced to separate the camping beam data from the full scan,
if desired. With INTTIME interfacing, one averaging period of one slice will be followed by an averaging period of another, and so on. The averaging periods are interleaved. The resulting experiment runs beams 0, 7, 1, 7, etc. 

.. image:: img/themisscan.png
   :width: 800px
   :alt: THEMISSCAN slice interfacing 
   :align: center

In a twofsound experiment, a full scan of one frequency is followed by a full scan of another frequency. The txfreq are unique between the slices. In this experiment, the slices are SCAN interfaced. A full scan of slice 0 runs 
followed by a full scan of slice 1, and then the process repeats. 

.. image:: img/twofsound.png
   :width: 800px
   :alt: TWOFSOUND slice interfacing 
   :align: center


Here's a theoretical example showing all types of interfacing. In this example, slices 0 and 1 are PULSE interfaced. Slices 0 and 2 are INTEGRATION interfaced. Slices 0 and 3 are INTTIME interfaced. Slices 0 and 4 are SCAN interfaced.

.. image:: img/one-experiment-all-interfacing-types.png
   :width: 800px
   :alt: An example showing all types of slice interfacing 
   :align: center


Writing an Experiment
---------------------

All experiments must be written as their own class and must be built off of the built-in ExperimentPrototype class.  This means the ExperimentPrototype class must be imported
at the start of the experiment file::

    from experiments.experiment_prototype import ExperimentPrototype

You must also build your class off of the ExperimentPrototype class, which involves inheritance. To do this, define your class
like so::

    class MyClass(ExperimentPrototype):

        def __init__(self):
            cpid = 123123  # this must be a unique id for your control program.
            super(MyClass, self).__init__(cpid)

The experiment handler will create an instance of your experiment when your experiment is scheduled to start running. Your class is a child class of ExperimentPrototype and because of this, the parent class needs to be instantiated when the experiment is instantiated. This is important because the experiment_handler will build the scans required by your class in a way that is easily readable and iterable by the radar control program. This is done by methods that are set up in the ExperimentPrototype parent class.

The next step is to add slices to your experiment. An experiment is defined by the slices in the class, and how the slices interface. Slices are just dictionaries, with a preset list of keys available to define your experiment. ::



TODO

..  TODO outline ways to interface

..  TODO determine where users should write their experiments
    because that will affect the import statement - putting them
    directly in experiments?

Checking your Experiment for Errors
-----------------------------------

..  TODO how to check your experiment for errors

