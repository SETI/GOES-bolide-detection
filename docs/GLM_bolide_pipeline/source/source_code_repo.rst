.. _source_code_repo:

Getting the GLM Pipeline Source Code
====================================

There are two GitHub repositories: A public and a private one. The public is for public code releases.

Public repo: `https://github.com/SETI/GOES-bolide-detection<https://github.com/SETI/GOES-bolide-detection>`_
Private repo: `https://github.com/SETI/atap<https://github.com/SETI/atap>`_

Neither is Pip installable and must be installed manually from source. If you desire access the private repo then please
contact the developers.

Decide where you want to put the repository (e.g., under $HOME/atap_dev), then change your current working directory to
that location, and clone the repository as follows:::

    cd $HOME/<git-path>

    git clone https://<username>@github.com/SETI/GOES-bolide-detection.git **(public)**

    or 

    git clone https://<username>@github.com/SETI/atap.git **(private)**

You will either be asked for a password or you use your set up GitHub security passthrough. 
You can stay on the main branch for now, unless you have a specific branch to checkout.

Generating This Documentation
-----------------------------

You apparently already have this documentation. You can compile it if you have Sphinx installed. 
To generate this documentation, go to this directory in the source code: `docs/GLM_bolide_pipeline` and then type::

    make html

You can then go to your web browser and navigate to::

    file:///<$ATAP_REPO_ROOT>/docs/GLM_bolide_pipeline/build/html/index.html

This assumes your already have a python environment with the Sphinx document generation software installed. Instructions
for setting up a Python environment is in the next section: :doc:`python_environment`.
    
