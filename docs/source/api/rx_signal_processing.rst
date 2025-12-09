====================
RX Signal Processing
====================

The Borealis radar receive side signal processing is mostly moved into software using the digital
radar design. The RX DSP block is designed to utilize a GPU to maximize
parallelism and be able to process as much data as possible in real-time.

Borealis experiments give lots of flexibility for filtering. Filter coefficients are generated as
part of decimation schemes at the experiment level. The DSP block receives the coefficients for
each stage of decimation from Radar Control. The DSP block has been designed to be able to run
as many decimation stages as are configured in the decimation scheme. This allows SuperDARN users
to have as much control as they want in designing filter characteristics.

Sampled data stored in shared memory is then opened, and operation of the GPU is configured. The GPU
does not have enough computation resources to be able to process data from more than one sequence,
but it can copy data from one sequence to the GPU memory while another
sequence is being processed.

.. figure:: /source/img/dsp_data_flow.jpg
   :width: 80 %
   :alt: Diagram of Rx DSP data flow during decimation
   :align: center

   Diagram of Rx DSP data flow during decimation

The ``rx_signal_processing`` module doesn't take any arguments, so it can be started
like any python script.

DSP Class
---------

See the :ref:`source code <Signals>`

See also
--------

:ref:`Borealis DSP`

:ref:`frerking`
