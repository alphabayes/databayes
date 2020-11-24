# -*- coding: utf-8 -*-
from import_pkg import DurationModelSingleStateBase, Weibull
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
def data_8_df():
    df = pd.DataFrame(dict(
        V1=[12.5, 0, 25, 10, 112.75, 50, 154.5, 0],
        V2=["A", "A", "B", "B", "C", "C", "D", "D"],
        V3=[True, False, True, False, True, False, True, False]))

    return df


def test_DurationModelSingleStateBase_001(data_8_df):

    model_specs = {
        "filter": {
            "V3": True
        }
    }

    model = DurationModelSingleStateBase(**model_specs)
    data_filtered_df = model.get_event_data(data_8_df)

    data_filtered_exp = \
        {'V1': {0: 12.5, 2: 25.0, 4: 112.75, 6: 154.5},
         'V2': {0: 'A', 2: 'B', 4: 'C', 6: 'D'},
         'V3': {0: True, 2: True, 4: True, 6: True}}

    assert data_filtered_df.to_dict() == data_filtered_exp


def test_Weibull_001(data_8_df):

    model_specs = {
        "var_targets": ["V1"],
        "filter": {
            "V3": True
        },
        "predict_parameters": {
            "var_targets_discrete_domain": {
                "V1": list(range(0, 161, 10)) + [np.inf]
            }
        },
    }

    model = Weibull(**model_specs)

    model.fit(data_8_df)

    model.predict(data_8_df)

    # ADD TEST HERE !!!!
    ipdb.set_trace()
