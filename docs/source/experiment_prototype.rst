====================
Experiment Prototype
====================

The experiment_prototype package contains the building blocks of experiments, which includes the
ExperimentPrototype base class, the scan_classes subpackage including the ScanClassBase classes,
and the ExperimentException. There is also a list_tests module which is used by the
ExperimentPrototype class.

----------------------------
Experiment Prototype Package
----------------------------

.. automodule:: src.experiment_prototype.experiment_prototype
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: src.experiment_prototype.experiment_slice.ExperimentSlice()
    :members: check_slice

.. automodule:: src.experiment_prototype.experiment_exception
    :members:
    :undoc-members:
    :show-inheritance:

------------------
Scan Class Package
------------------

.. automodule:: src.experiment_prototype.scan_classes.scan_class_base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: src.experiment_prototype.scan_classes.scans
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: src.experiment_prototype.scan_classes.averaging_periods
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: src.experiment_prototype.scan_classes.sequences
    :members:
    :undoc-members:
    :show-inheritance:

-----------------
Decimation Scheme
-----------------

.. automodule:: src.experiment_prototype.experiment_utils.decimation_scheme
    :members:
    :undoc-members:
    :show-inheritance:
