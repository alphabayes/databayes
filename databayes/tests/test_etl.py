# -*- coding: utf-8 -*-
from import_pkg import Discretizer
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


def test_Discretizer_001(data_5_df):

    disc = Discretizer(variables=dict(
        V1=dict(params=dict(
            bins=50)
        )
    ))

    disc_var = disc.variables['V1']

    series_V1_d = disc_var.discretize(data_5_df["V1"])
    series_V1_d_exp = ['(-0.1, 2.0]', '(-0.1, 2.0]',
                       '(50.0, 52.0]', '(28.0, 30.0]', '(98.0, 100.0]']
    assert (series_V1_d == series_V1_d_exp).all()
    assert len(series_V1_d.cat.categories) == 50

    series_V2_d = disc_var.discretize(data_5_df["V2"])
    series_V2_d_exp = ['A', 'A', 'A', 'B', 'B']
    assert (series_V2_d == series_V2_d_exp).all()
    assert len(series_V2_d.cat.categories) == 2

    series_V3_d = disc_var.discretize(data_5_df["V3"])
    series_V3_d_exp = ['(-0.0356, -0.00938]', '(0.0168, 0.0431]',
                       '(0.0955, 0.122]', '(0.725, 0.751]', '(-0.561, -0.534]']
    assert (series_V3_d == series_V3_d_exp).all()
    assert len(series_V3_d.cat.categories) == 50

    data_disc_df = disc.discretize(data_5_df)
    data_disc_exp = \
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
         'V3': {0: -0.01,
                1: 0.023,
                2: 0.115,
                3: 0.751,
                4: -0.56}}
    assert data_disc_df.to_dict() == data_disc_exp


def test_Discretizer_001(data_5_df):

    # TODO: Make a test with bins specification
    # + process_all_variables option
    disc = Discretizer(variables=dict(
        V1=dict(params=dict(
            bins=50)
        )
    ))
