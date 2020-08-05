# -*- coding: utf-8 -*-

import databayes.modelling.core as dcmc
import databayes.utils.etl as etl
import yaml
import numpy as np
import pandas as pd
import logging
import pytest
import os
import json
from copy import deepcopy

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


logger = logging.getLogger()

DATA_DIR = "data"
EXPECTED_DIR = "expected"


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
        dcmc.RandomUniformModel(var_targets=["ODM_LIBELLE", "SIG_INCIDENT"],
                                var_features=["SIG_LIEU", "SIG_ORGANE"])

    data = etl.discretize(gmaobus_om_ot_100_df)
    data_train_df = data.loc[:75]
    data_test_df = data.loc[75:]

    mlmodel.fit(data_train_df)
    pred_prob = mlmodel.predict(data_test_df)

    for tv in mlmodel.var_targets:
        tv_dom_size = len(data[tv].cat.categories)
        assert np.allclose(pred_prob[tv]["scores"].values, 1/tv_dom_size)


def test_RandomUniformModel_002(gmaobus_om_ot_100_df):

    mlmodel = \
        dcmc.RandomUniformModel(var_targets=["ODM_LIBELLE"])

    data = etl.discretize(gmaobus_om_ot_100_df)
    data_train_df = data.loc[:75]
    data_test_df = data.loc[75:]

    mlmodel.fit(data_train_df)
    pred_prob = mlmodel.predict(data_test_df)

    for tv in mlmodel.var_targets:
        tv_dom_size = len(data[tv].cat.categories)
        assert np.allclose(pred_prob[tv]["scores"].values, 1/tv_dom_size)
