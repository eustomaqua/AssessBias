.. quickstart.rst


Requirements
================
.. toctree::
   :maxdepth: 1


We developed `AssessBias <https://github.com/eustomaqua/AssessBias>`_ primarily with ``Python 3.8``, but plotted figures with ``Python 3.11``. Remember to install anaconda/miniconda first, and then choose the corresponding environment.

.. code-block:: console
  :linenos:

  $ # To obtain the empirical data
  $ conda create -n py38 python=3.8
  $ source activate py38
  $ pip install --upgrade pip
  $ pip install -r requirements.txt
  $ # conda deactivate
  $ # conda remove -n py38 --all

.. code-block:: console
  :linenos:

  $ # To obtain the plotted figures
  $ conda create -n py311 python=3.11
  $ source activate py311
  $ pip install -U pip
  $ pip install -r reqs_dev.txt
  $ # conda deactivate
  $ # conda remove -n py311 --all


We borrow some auxiliary functions from `PyFairness <https://github.com/eustomaqua/PyFairness>`_, and to use it, please do the following.

.. code-block:: console
  :linenos:

  $ # Two ways to install (& uninstall) PyFairness
  $ git clone git@github.com:eustomaqua/PyFairness.git
  $
  $ pip install -r PyFairness/reqs_py311.txt
  $ pip install -e ./PyFairness
  $ # pip uninstall pyfair
  $
  $ cp -r ./PyFairness/pyfair ./
  $ cp -r ./PyFairness/data ./
  $ # rm -r pyfair data
  $ yes | rm -r PyFairness

