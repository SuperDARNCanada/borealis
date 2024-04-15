====================
Experiment Prototype
====================

The experiment_prototype package contains the building blocks of experiments, which includes the
ExperimentPrototype base class, the interface_classes subpackage including the InterfaceClassBase classes,
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

-----------------------
Interface Class Package
-----------------------

.. automodule:: src.experiment_prototype.interface_classes.interface_class_base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: src.experiment_prototype.interface_classes.scans
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: src.experiment_prototype.interface_classes.averaging_periods
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: src.experiment_prototype.interface_classes.sequences
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
