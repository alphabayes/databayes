# Data Bayes

Machine learning driven by Bayesian decision making theory

## Installation for development

```bash
git clone git@gitlab.com:alphabayes/databayes.git
```

Notes :
- It is recommanded to create a Python virtual environment to start developping the library.
- You can install some useful Python development library with suivante :
```bash
pip install --use-feature=2020-resolver -r databayes/python-devtools-requirements.txt
```

Finally, use the following command to install the library on your system/virtual env.:
```bash
pip install --use-feature=2020-resolver -e databayes
```

Check if everythings ok by launching the tests :
```bash
cd databayes/databayes/tests
pytest -v --runslow
```


## Install the library for using in a project

Just run the following command to install the library on your system :
```bash
pip install --use-feature=2020-resolver git+ssh://git@gitlab.com/alphabayes/databayes.git
```

