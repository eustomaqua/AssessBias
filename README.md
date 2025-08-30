# AssessBias

Welcome to the AssessBias's documentation. This is to help you reproduce our work from

1. Algorithmic fairness: Not a purely technical but socio-technical property
2. Do existing fairness measures suffice? Assessing discrimination in algorithmic decision-making


## Requirements

We developed [AssessBias](https://github.com/eustomaqua/AssessBias) primarily with ``Python 3.8``, but plotted figures with ``Python 3.11``. Remember to install anaconda/miniconda first, and then choose the corresponding environment.

```shell
$ # To obtain the empirical data
$ conda create -n py38 python=3.8
$ source activate py38
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ # conda deactivate
$ # conda remove -n py38 --all
```

```shell
$ # To obtain the plotted figures
$ conda create -n py311 python=3.11
$ source activate py311
$ pip install -U pip
$ pip install -r reqs_dev.txt
$ # conda deactivate
$ # conda remove -n py311 --all
```

We borrow some auxiliary functions from [PyFairness](https://github.com/eustomaqua/PyFairness), and to use it, please do the following.

```shell
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
```

## Reproduction

To reproduce our empirical results, you may use the [released data](https://github.com/eustomaqua/AssessBias/tree/master/findings) and do as follows.

```shell
$ # source activate py311
$ python fair_int_draw.py -exp KF_exp1b
$ # conda deactivate
```


## Additional information

### Licence

*AssessBias* is released under the [MIT Licence](./LICENSE).

