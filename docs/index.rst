ARKHE Framework Documentation
==============================

Welcome to the ARKHE (ARKHĒ) Framework documentation!

ARKHE is an enterprise-grade Python framework for mathematical sequence research and machine learning experimentation. It provides tools for exploring mathematical sequences (such as Collatz), performing statistical analysis, and training transformer models to understand sequence patterns.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/getting_started
   guides/faq
   guides/training_guide
   guides/usage_examples
   guides/streamlit_setup
   guides/docker_setup
   api/index

Quick Start
-----------

Install the framework:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Generate a Collatz sequence:

.. code-block:: python

   from math_research.sequences import CollatzSequence

   seq = CollatzSequence(start=27)
   sequence = seq.generate()
   print(sequence)

Features
--------

* **Sequence Generation**: Generate and analyze mathematical sequences
* **Machine Learning**: Train transformer models on sequence data
* **Statistical Analysis**: Comprehensive analysis tools
* **Visualization**: Built-in plotting and visualization
* **CLI Interface**: Command-line tools for common tasks
* **Web Interface**: Streamlit-based interactive dashboard

Documentation Structure
------------------------

* :doc:`guides/getting_started` - Get started with ARKHE
* :doc:`guides/faq` - Frequently asked questions
* :doc:`guides/training_guide` - ML training guide
* :doc:`guides/usage_examples` - Usage examples
* :doc:`api/index` - Complete API reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

