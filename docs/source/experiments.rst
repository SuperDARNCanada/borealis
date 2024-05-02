.. _experiments:

===========
Experiments
===========

This is where you would create your experiment that you would like to run on the
radar. The following are a couple of examples of current SuperDARN experiments, and a
brief discussion of the ``update()`` method which will be implemented at a later date.

normalscan
----------

Normalscan is a very common experiment for SuperDARN. It does not update itself, so
no ``update()`` method is necessary. It only has a single slice, as there is only one
frequency, pulse_len, beam_order, etc. Since there is only one slice there is no need
for an interface dictionary.

..  literalinclude:: ../../src/borealis_experiments/normalscan.py
    :linenos:
    :language: python
    :caption: normalsound.py

twofsound
---------

Twofsound is a common variant of the normalscan experiment for SuperDARN. It does not
update itself, so no ``update()`` method is necessary. It has two frequencies so will
require two slices. The frequencies switch after a full scan (full cycle through the
beams), therefore the interfacing between slices 0 and 1 should be 'SCAN'.

..  literalinclude:: ../../src/borealis_experiments/twofsound.py
    :linenos:
    :language: python
    :caption: twofsound.py

full_fov
--------

See :ref:`Full FOV Imaging<full fov imaging>` for more information.

..  literalinclude:: ../../src/borealis_experiments/full_fov.py
    :linenos:
    :language: python
    :caption: full_fov.py

bistatic_test
-------------

See :ref:`Bistatic Experiments<bistatic experiments>` for more information.

..  literalinclude:: ../../src/borealis_experiments/bistatic_test.py
    :linenos:
    :language: python
    :caption: bistatic_test.py
