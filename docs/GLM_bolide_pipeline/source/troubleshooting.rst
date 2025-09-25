.. _troubleshooting:

.. sectionauthor:: Robert L. Morris <robert.l.morris@nasa.gov>

Troubleshooting
===============================
This page contains notes on issues that have arisen in configuring and running the software and what we've done to resolve them.


.. _installing_pytorch:

Installing PyTorch
------------------
When installing PyTorch with conda, it may be necessary to update your conda installation to the most recent version. This can be accomplished with the following command: ::

    $ conda update -n base -c defaults conda

If you still see package conflicts after updating, it may be necessary to relax constraints on package versions. The recommended pytorch installation (see :ref:`Setting up a Python Environment for the GLM Bolide Pipeline<python_environment>`) specifies versions for each package. Removing some or all of those version specifications may help: ::

    $ conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c Nvidia

Note that, by allowing alternate package versions to be installed, you may end up with an installation having limited functionality and may see warnings to that effect. Depending on what you intend to do with the software (i.e., what you really need from PyTorch), this may or may not be acceptable to you.

