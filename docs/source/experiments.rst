experiments package
===================

This is where you would create your experiment that you would like to run on the
radar. The following are a couple of examples of current SuperDARN experiments, and a
brief discussion of the update() method which will be implemented at a later date.

experiments.normalscan module
-----------------------------

Normalscan is a very common experiment for SuperDARN. It does not update itself, so
no update() method is necessary. It only has a single slice, as there is only one
frequency, pulse_len, beam_order, etc. Since there is only one slice there is no need
for an interface dictionary.

..  literalinclude:: ../../experiments/normalscan.py
    :linenos:

experiments.twofsound module
----------------------------

Twofsound is a common variant of the normalscan experiment for SuperDARN. It does not
update itself, so no update() method is necessary. It has two frequencies so will
require two slices. The frequencies switch after a full scan (full cycle through the
beams), therefore the interfacing between slices 0 and 1 should be 'SCAN'.

..  literalinclude:: ../../experiments/twofsound.py
    :linenos:

Example of experiment update() method
-------------------------------------

Experiments are a class of their own, built off the base class of ExperimentPrototype,
with an initializer and optionally an update() method. Update is currently not
implemented as datawrite is not currently passing the data back to experiment_handler.
The update() method will be called from within experiment_handler. A blank update
method would look like the following.

.. literalinclude:: ../../experiments/blank_experiment.py
    :pyobject: Blank.update