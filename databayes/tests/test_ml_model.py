# -*- coding: utf-8 -*-
from import_pkg import MLModel, RandomUniformModel, discretize, Discretizer
import yaml
import numpy as np
import pandas as pd
import logging
import pytest
import os

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


logger = logging.getLogger()

DATA_DIR = "data"
EXPECTED_DIR = "expected"


@pytest.fixture
def data_5_df():
    df = pd.DataFrame(dict(
        V1=[0, 2, 50.56, 30, 100],
        V2=["A", "A", "A", "B", "B"],
        V3=[-0.01, 0.023, 0.115, 0.751, -0.56]))

    return df


@pytest.fixture(scope="module")
def gmaobus_models_specs():
    filename = os.path.join(DATA_DIR, "gmaobus_mlp_models.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)


@pytest.fixture
def gmaobus_om_ot_100_df():
    filename = os.path.join(DATA_DIR, "gmaobus_om_ot_100.csv")
    df = pd.read_csv(filename, sep=";")

    return df


def test_RandomUniformModel_001(gmaobus_om_ot_100_df):

    mlmodel = \
        RandomUniformModel(var_targets=["ODM_LIBELLE", "SIG_INCIDENT"],
                           var_features=["SIG_LIEU", "SIG_ORGANE"])

    data = discretize(gmaobus_om_ot_100_df)
    data_train_df = data.loc[:75]
    data_test_df = data.loc[75:]

    mlmodel.fit(data_train_df)
    pred_prob = mlmodel.predict(data_test_df)

    for tv in mlmodel.var_targets:
        tv_dom_size = len(data[tv].cat.categories)
        assert np.allclose(pred_prob[tv]["scores"].values, 1/tv_dom_size)


def test_RandomUniformModel_002(gmaobus_om_ot_100_df):

    mlmodel = \
        RandomUniformModel(var_targets=["ODM_LIBELLE"])

    data = discretize(gmaobus_om_ot_100_df)
    data_train_df = data.loc[:75]
    data_test_df = data.loc[75:]

    mlmodel.fit(data_train_df)
    pred_prob = mlmodel.predict(data_test_df)

    for tv in mlmodel.var_targets:
        tv_dom_size = len(data[tv].cat.categories)
        assert np.allclose(pred_prob[tv]["scores"].values, 1/tv_dom_size)


def test_MLModel_001(data_5_df):

    mlmodel_specs = {
        "var_discretizer": {
            "process_all_variables": True,
            "variables": {
                "V1": {"params": {"bins": 50}}
            }
        }
    }

    mlmodel = MLModel(**mlmodel_specs)

    data_ddf = mlmodel.prepare_fit_data(data_5_df)

    data_ddf_exp = \
        {'V1': {0: '(-0.1, 2.0]',
                1: '(-0.1, 2.0]',
                2: '(50.0, 52.0]',
                3: '(28.0, 30.0]',
                4: '(98.0, 100.0]'},
         'V2': {0: 'A',
                1: 'A',
                2: 'A',
                3: 'B',
                4: 'B'},
         'V3': {0: '(-0.0356, 0.0955]',
                1: '(-0.0356, 0.0955]',
                   2: '(0.0955, 0.227]',
                3: '(0.62, 0.751]',
                4: '(-0.561, -0.429]'}}

    assert data_ddf.to_dict() == data_ddf_exp
    assert len(data_ddf["V1"].cat.categories) == 50
    assert len(data_ddf["V2"].cat.categories) == 2
    assert len(data_ddf["V3"].cat.categories) == 10
