==================
Starting the radar
==================

Writing an Experiment
---------------------

All experiments must be written as their own class and
must be built off of the built-in ExperimentPrototype
class.  This means the ExperimentPrototype class must be imported
at the start of the experiment file::

    from experiments.experiment_prototype import ExperimentPrototype

You must also build your class off of the ExperimentPrototype
class, which involves inheritance. To do this, define your class
like so::

    class MyClass(ExperimentPrototype):

        def __init__(self):
            cpid = 123123  # this must be a unique id for your control program.
            super(MyClass, self).__init__(cpid)

The experiment handler will create an instance of your
experiment when your experiment is scheduled to start running.
Your class is a child class of ExperimentPrototype and because of this,
the parent class needs to be instantiated when the experiment is
instantiated. This is important because the experiment_handler will build the scans
required by your class in a way that is easily readable and iterable
by the radarcontrol program. This is done by methods that are set up
in the ExperimentPrototype parent class.

The next step is to add slices to your experiment. An experiment is
defined by the slices in the class, and how the slices interface. You
can think of a slice as an experiment of its own, and your experiment
may only require one slice. However, more complicated functionality
will require multiple slices, interfaced in one of three ways:



..  TODO determine where users should write their experiments
    because that will affect the import statement - putting them
    directly in experiments?

Checking your Experiment for Errors
-----------------------------------



