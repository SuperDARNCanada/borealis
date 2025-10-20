==================
Experiment Handler
==================

The experiment_handler package contains a single module, experiment_handler, that is a
standalone program.

.. automodule:: src.experiment_handler
    :members:
    :undoc-members:
    :show-inheritance:

-----
Usage
-----

Starting the ``experiment_handler`` module::

    experiment_handler.py [-h] experiment_module scheduling_mode_type

        Pass the module containing the experiment to the experiment handler as a required
        argument. It will search for the module in the BOREALISPATH/experiment_prototype
        package. It will retrieve the class from within the module (your experiment).

        It will use the experiment's build_scans method to create the iterable InterfaceClassBase
        objects that will be used by the radar_control block, then it will pass the
        experiment to the radar_control block to run.

        It will be passed some data to use in its .update() method at the end of every
        integration time. This has yet to be implemented but will allow experiments to
        modify themselves based on received data as feedback. This is not a necessary method
        for all experiments and if there is no update method experiment updates will not
        occur.

    positional arguments:
      experiment_module     The name of the module in the experiment_prototype package that contains your Experiment class, e.g. normalscan
      scheduling_mode_type  The type of scheduling time for this experiment run, e.g. common, special, or discretionary.

    options:
      -h, --help            show this help message and exit
      --embargo             Embargo the file (makes the CPID negative)
      --kwargs KWARGS [KWARGS ...]
                            Keyword arguments for the experiment. Each must be formatted as kw=val

.. autoprogram:: src.experiment_handler:experiment_parser()
    :prog: experiment_handler.py
