====================
Experiment Unittests
====================

The ``experiment_unittests.py`` script tests both runnable experiments (those in borealis/src/borealis_experiments/) and a set of unit tests
(those in ``src/borealis_experiments/tests/``). Some unit tests are meant to raise an exception;
these tests have an extra method defined which returns the expected exception and a regex of the expected error message.
An example unit test is shown below.

The ``experiment_unittests.py`` script uses the ``RADAR_ID`` environment variable, typically defined in ``.profile`` or ``.bashrc``.
To test a different ``RADAR_ID``, the script ``test_as_site.py`` can be run, passing in a site ID such as ``rkn``. You can also
use this script to test a specific experiment or set of experiments as an alternate site.

..  literalinclude:: ../../src/borealis_experiments/tests/avg_method.py
    :language: python
    :linenos:
    :caption: Example Unit Test file
    :name: test_avg_method-py
