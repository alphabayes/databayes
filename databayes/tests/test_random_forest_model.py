from databayes.modelling.DiscreteDistribution import DiscreteDistribution
from databayes.modelling.SKlearnClassifiers import RandomForestModel
from databayes.utils.ml_performance import MLPerformance
import yaml
import pprint
import numpy as np
import pandas as pd
import logging
import pytest
import os
import sys
import json
from copy import deepcopy
import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


logger = logging.getLogger()

DATA_DIR = "data"
EXPECTED_DIR = "expected"

def discretize_data(data_df):

    data_ddf = data_df.copy(deep=True)

    for var in data_df.columns:
        if isinstance(data_df[var].dtypes, pd.CategoricalDtype):
            continue

        if data_df[var].dtypes == "float":
            data_disc = pd.cut(data_df.loc[:, var], 3)
            cats_str = data_disc.cat.categories.astype(str)
            cat_type = \
                pd.api.types.CategoricalDtype(categories=cats_str,
                                              ordered=True)
            data_ddf.loc[:, var] = data_disc.astype(str).astype(cat_type)

        else:
            data_ddf.loc[:, var] = data_df.loc[:, var].astype("category")

    return data_ddf

@pytest.fixture(scope="module")
def gmaobus_ml_performance_specs():
    filename = os.path.join(DATA_DIR, "gmaobus_ml_performance.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)

@pytest.fixture(scope="module")
def gmaobus_models_specs():
    filename = os.path.join(DATA_DIR, "gmaobus_rf_models.yaml")
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

@pytest.fixture
def gmaobus_om_ot_10000_df():
    filename = os.path.join(DATA_DIR, "gmaobus_om_ot_10000.csv")
    df = pd.read_csv(filename, sep=";")

    return df

def test_RandomForestModel_001(gmaobus_models_specs, gmaobus_om_ot_100_df):

    rf_ml = RandomForestModel(**gmaobus_models_specs["rf_02"])

    gmaobus_om_ot_100_df[rf_ml.var_targets] = gmaobus_om_ot_100_df[rf_ml.var_targets].astype('category')

    for feature in rf_ml.var_features:
        if not gmaobus_om_ot_100_df[feature].dtypes in ['int', 'float']:
            gmaobus_om_ot_100_df[feature] = gmaobus_om_ot_100_df[feature].astype('category')

    data_train_df = gmaobus_om_ot_100_df[:50]
    data_test_df = gmaobus_om_ot_100_df.loc[25:]

    rf_ml.fit(data_train_df)
    pred_test = rf_ml.predict(data_test_df)
    
    for var in pred_test.keys():
        for pred_key in pred_test.get(var, {}).keys():
            pred_res = pred_test.get(var, {}).get(pred_key, {})

    assert isinstance(pred_test["ODM_LIBELLE"]
                      ["scores"], DiscreteDistribution)

@pytest.mark.slow
def test_RandomForestModel_002(gmaobus_models_specs, gmaobus_om_ot_10000_df, gmaobus_ml_performance_specs):

    model_param = deepcopy(gmaobus_models_specs["rf_01"])

    data = gmaobus_om_ot_10000_df.copy().iloc[50:100]
    data['KILOMETRAGE'] = pd.qcut(data['KILOMETRAGE'], 3)

    ml_perf_param = deepcopy(
        gmaobus_ml_performance_specs["ml_performance_analysis_02"])

    data['GARANTIE'] = \
        data['GARANTIE'].astype('str')

    rf_ml_str = RandomForestModel(**model_param)

    # Note: deepcopy is necessary since MLPerformance Validator modifies
    # input specs dictionnary
    ml_perf_gar_str = MLPerformance(
        model=rf_ml_str,
        **deepcopy(ml_perf_param))
    
    ml_perf_gar_str.run(data,
                        logger=logger,
                        progress_mode=True)

    data['GARANTIE'] = data['GARANTIE'].astype('int')

    rf_ml_int = RandomForestModel(**model_param)
    ml_perf_gar_int = MLPerformance(
        model=rf_ml_int,
        **deepcopy(ml_perf_param))

    ml_perf_gar_int.run(data,
                        logger=logger,
                        progress_mode=True)

    assert ml_perf_gar_int.measures_approx_equal(ml_perf_gar_str)



def test_RandomForestModel_003_01(gmaobus_models_specs, gmaobus_om_ot_100_df):

    rf_ml = RandomForestModel(**gmaobus_models_specs["rf_02"])

    data = discretize_data(gmaobus_om_ot_100_df)
    data_train_df = gmaobus_om_ot_100_df[:75]
    data_test_df = gmaobus_om_ot_100_df.loc[75:]

    rf_ml.fit(data_train_df)
    pred_prob = rf_ml.predict(data_test_df)

    for tv in rf_ml.var_targets:
        expected_filename = os.path.join(EXPECTED_DIR,
                                         "test_RandomForestModel_003_01_" + tv + ".csv")
        #pred_prob[tv]["scores"].to_csv(expected_filename, index=True)
        expected_prob = DiscreteDistribution.read_csv(
            expected_filename, index_col=0)

        assert np.allclose(pred_prob[tv]["scores"], expected_prob)

@pytest.mark.slow
def test_RandomForestModel_003_02(gmaobus_models_specs, gmaobus_om_ot_10000_df):

    rf_ml = RandomForestModel(**gmaobus_models_specs["rf_01"])

    data = discretize_data(gmaobus_om_ot_10000_df)
    data_train_df = data[:9000]
    data_test_df = data[9000:]

    rf_ml.fit(data_train_df)
    pred_prob = rf_ml.predict(data_test_df)

    for tv in rf_ml.var_targets:
        expected_filename = os.path.join(EXPECTED_DIR,
                                         "test_RandomForestModel_003_02_" + tv + ".csv")
        #pred_prob[tv]["scores"].to_csv(expected_filename, index=True)
        expected_prob = DiscreteDistribution.read_csv(
            expected_filename, index_col=0)

        assert np.allclose(pred_prob[tv]["scores"], expected_prob)

def test_RandomForestModel_004_01(gmaobus_models_specs, gmaobus_om_ot_100_df, gmaobus_ml_performance_specs):

    model_param = deepcopy(gmaobus_models_specs["rf_03"])

    rf_ml = RandomForestModel(**model_param)

    ml_perf_param = deepcopy(
        gmaobus_ml_performance_specs["ml_performance_analysis_02"])

    ml_perf = MLPerformance(
        model=rf_ml,
        **ml_perf_param)

    data = discretize_data(gmaobus_om_ot_100_df)

    ml_perf.run(data,
                logger=logger,
                progress_mode=True)

    ml_perf_expected_filename = os.path.join(EXPECTED_DIR,
                                             "test_RandomForestModel_004_01_perf.json")

    # with open(ml_perf_expected_filename, 'w') as json_file:
    #     json.dump(ml_perf.dict(), json_file)

    with open(ml_perf_expected_filename, 'r') as json_file:
        ml_perf_expected_specs = json.load(json_file)

    ml_perf_expected = MLPerformance(**ml_perf_expected_specs)

    assert ml_perf.measures_approx_equal(ml_perf_expected)

@pytest.mark.slow
def test_RandomForestModel_004_02(gmaobus_models_specs, gmaobus_om_ot_10000_df, gmaobus_ml_performance_specs):

    model_param = deepcopy(gmaobus_models_specs["rf_01"])

    rf_ml = RandomForestModel(**model_param)

    ml_perf_param = deepcopy(
        gmaobus_ml_performance_specs["ml_performance_analysis_02"])

    ml_perf = MLPerformance(
        model=rf_ml,
        **ml_perf_param)

    data = discretize_data(gmaobus_om_ot_10000_df)

    ml_perf.run(data,
                logger=logger,
                progress_mode=True)

    ml_perf_expected_filename = os.path.join(EXPECTED_DIR,
                                             "test_RandomForestModel_004_02_perf.json")

    # with open(ml_perf_expected_filename, 'w') as json_file:
    #     json.dump(ml_perf.dict(), json_file)

    with open(ml_perf_expected_filename, 'r') as json_file:
        ml_perf_expected_specs = json.load(json_file)

    ml_perf_expected = MLPerformance(**ml_perf_expected_specs)

    assert ml_perf.measures_approx_equal(ml_perf_expected)

def test_RandomForestModel_005(gmaobus_models_specs, gmaobus_om_ot_10000_df, gmaobus_ml_performance_specs):

    model_param = deepcopy(gmaobus_models_specs["rf_01"])

    ml_perf_param = deepcopy(
        gmaobus_ml_performance_specs["ml_performance_analysis_02"])

    rf_ml = RandomForestModel(**model_param)  

    ml_perf = MLPerformance(
        model=rf_ml,
        **ml_perf_param)

    data = discretize_data(gmaobus_om_ot_10000_df).iloc[:20]

    ml_perf.run(data,
                logger=logger,
                progress_mode=True)

    ml_perf_expected_filename = os.path.join(EXPECTED_DIR,
                                             "test_MLPClassifierModel_005_01_perf.json")
    
    # with open(ml_perf_expected_filename, 'w') as json_file:
    #     json.dump(ml_perf.dict(), json_file)
    
    with open(ml_perf_expected_filename, 'r') as json_file:
        ml_perf_expected_specs = json.load(json_file)

    ml_perf_expected = MLPerformance(**ml_perf_expected_specs)

    assert ml_perf.measures_approx_equal(ml_perf_expected)


    