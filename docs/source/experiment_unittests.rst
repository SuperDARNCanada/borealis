====================
Experiment Unittests
====================

..  literalinclude:: ../../tests/experiments/experiment_unittests.py
    :language: python
    :linenos:
    :caption: experiment_unittests.py
    :name: experiment_unittests-py


This script tests both runnable experiments (those in borealis/src/borealis_experiments/) and a set of unit tests
(those in borealis/src/borealis_experiments/testing_archive/). Some unit tests are meant to raise an exception;
these tests have an extra method defined which returns the expected exception and a regex of the expected error message.
An example unit test is shown below.

..  literalinclude:: ../../src/borealis_experiments/testing_archive/test_rxonly_dne.py
    :language: python
    :linenos:
    :caption: Example Unit Test file
    :name: test_rxonly_dne-py
