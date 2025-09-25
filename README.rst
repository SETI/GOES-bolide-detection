=====================
GOES-bolide-detection
=====================

**A pipeline to search GOES weather satellite ABI and GLM data for bolides (exploding meteors).**


*GOES-bolide-detection* is a python-based automated pipeline to search for bolides in GOES weather satellite data.  It
detects bolides principally using the Geostationary Lightning Mapper instruments onboard the GOES-16, 17, 18 and 19
satellites, https://www.goes-r.gov/spacesegment/glm.html.  Detected bolides are posted at `neo-bolide.ndc.nasa.gov
<https://neo-bolide.ndc.nasa.gov>`_. We now have over 9500 bolides posted on our website as illustrated with the
following graphic, created using our bolides user package: https://github.com/SETI/bolides.  This image shows over 9500
detected bolides and highlights which GOES satellite the detection was made with by color. In the stereo region, where
GOES-East and GOES-West fields of view overlap, the detections can be made by two satellites, hence the numerous
different colors.

.. image:: docs/GLM_bolide_pipeline/media/all_detections_globe_cropped.png


Details about the pipeline can be found in these papers:

.. [1] Jeffrey C. Smith, et al., *An automated bolide detection pipeline for GOES GLM*, Icarus, Volume 368, 2021, 114576, ISSN 0019-1035, https://doi.org/10.1016/j.icarus.2021.114576. https://www.sciencedirect.com/science/article/pii/S0019103521002451.

.. [2] Jeffrey C. Smith, et al., *Finding Fireballs in Lightning: A Daily Pipeline to Find Meteors in Weather Satellite Data*, In prep. for JGR: Machine Learning and Computation

Motivation
==========

Large, life threatening, asteroid impacts are fortunately very rare. A 10 meter object capable of causing a
100 kiloton high altitude explosion, a sonic boom, broken windows and some damage to property and risk to
human health should occur on average about 1 per decade. A larger 50 meter object capable of a 10 megaton
explosion, an impact crater, local devastation and potential loss of life should occur on average about 1 per
1000 years. Extremely larger, 1 km sized objects could cause global devastation and possibly even the collapse
of civilization, but are unlikely to occur more often then 1 per 700,000 years.  The risk of devastation on a
short time period is therefore low, but the consequences are very high. The U.S. Government has therefore
released the most recent National Preparedness Strategy and Action Plan for Near-Earth Object Hazards and
Planetary Defense in 2023 [3] to address the hazard of NEO impacts, and focuses future
work on planetary defense across the U.S.  Government.  This ongoing effort has resulted the creation of the
NASA Planetary Defense Coordination Office to facilitate the study of this risk. One goal is to "improve Near
Earth Object modeling, prediction, and information integration and to coordinate the development of validated
modeling tools and simulation capabilities that aid in characterizing and mitigating NEO impact risks while
integrating and streamlining data flows to support effective decision-making".
To achieve this goal, NASA's Asteroid Threat Assessment Project (ATAP), a NASA Ames Research Center activity
in support of NASA’s Planetary Defense Coordination Office (PDCO), is generating a database of bolide light curves.
The goal of the work to develop this codebase is the detection,
characterization and generation of a large dataset of calibrated entry light curves of bolides in order to calibrate our
entry models to assess the risk of larger impacts. But the publicly available data set is useful for numerous
other science studies, including asteroid population studies to better understand the evolution of our Solar System.


.. [3] U.S. National Science and Technology Council. (2023). National preparedness strategy & action plan for potentially hazardous near-Earth objects and planetary defense (Tech. Rep.). : White House Office of Science and Technology Policy. https://assets.science.nasa.gov/content/dam/science/psd/planetary-science-division/2025/2023-NSTC-National-Preparedness-Strategy-and-Action-Plan-for-Near-Earth-Object-Hazards-and-Planetary-Defense.pdf



.. end-before-here


Documentation
=============

The GOES bolide pipeline documentation uses the Sphinx Python documentation generator https://www.sphinx-doc.org.
You can compile the documentation if you have Sphinx installed. 
To generate the documentation, go to the following directory in the source code: 

    `docs/GLM_bolide_pipeline` 

and then type::

    make html

You can then go to your web browser and navigate to::

    file:///<$REPO_ROOT>/docs/GLM_bolide_pipeline/build/html/index.html

This assumes your already have a python environment with the Sphinx document generation software installed. Instructions
for setting up a Python environment for the pipeline (which includes Sphinx) is in the documentation.

Installation
============

.. installation-start

This is not pip installable code. It has it's own custom installation and instructions are in the Sphinx documentation.
We'd greatly appreciate learning if anybody installs and runs this software and how it works for you. Please let us
know!

Acknowledgments
===============

This development is supported through NASA's Asteroid Threat Assessment Project (ATAP), which is funded through NASA's Planetary Defense Coordination Office (PDCO).
Jeffrey Smith and Robert Morris are supported through NASA Cooperative Agreement 80NSSC25M7108.
