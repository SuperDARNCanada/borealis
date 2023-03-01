====================
Experiment Unittests
====================

..  literalinclude:: ../../tests/testing_utils/experiments/experiment_unittests.py
    :language: python
    :linenos:
    :caption: experiment_unittests.py
    :name: experiment_unittests-py


Experiments must be listed in ``experiment_tests.csv`` in the following format: ::

    [#][experiment file module import name]::[regex error message]

An excerpt of ``experiment_tests.csv`` is shown below for referece.


..  literalinclude:: ../../tests/testing_utils/experiments/experiment_tests.csv
    :lineno-start: 128
    :lines: 128-133
    :caption: experiment_tests.csv
    :name: experiment_tests.csv