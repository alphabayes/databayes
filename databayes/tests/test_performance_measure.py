import databayes.utils.performance_measure as pfm
from databayes.modelling.DiscreteDistribution \
    import DiscreteDistribution

from databayes.utils.ml_performance import MLPerformance
import yaml
from databayes.modelling.SKlearnClassifiers import RandomForestModel
from databayes.utils import pdInterval_from_string, pdInterval_series_from_string

import numpy as np
import pandas as pd
import logging
import pytest
import os
import json
import pkg_resources


installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


logger = logging.getLogger()

DATA_DIR = "data"
EXPECTED_DIR = "expected"


@pytest.fixture(scope="module")
def cmapss_30_interval_df():
    filename = os.path.join(DATA_DIR, "cmapss_data_30_interval.csv")
    df = pd.read_csv(filename, sep=";")

    # Convert string interval to pandas interval
    df["RUL"] = pdInterval_series_from_string(df["RUL"])

    return df


@pytest.fixture(scope="module")
def cmapss_30_numeric_df():
    filename = os.path.join(DATA_DIR, "cmapss_data_30_numeric.csv")
    df = pd.read_csv(filename, sep=";")

    return df

# @pytest.fixture(scope="module")
# def cmapss_models_specs():
#     filename = os.path.join(DATA_DIR, "cmapss_models.yaml")
#     with open(filename, 'r', encoding="utf-8") as yaml_file:
#         try:
#             return yaml.load(yaml_file, Loader=yaml.SafeLoader)
#         except yaml.YAMLError as exc:
#             if not (logger is None):
#                 logger.error(exc)


@pytest.fixture(scope="module")
def cmapss_30_numeric_pred_prob():
    filename = os.path.join(DATA_DIR, "cmapss_30_numeric_dd.csv")
    dd_RUL = DiscreteDistribution.read_csv(filename, index_col=0)
    dd_RUL.columns = dd_RUL.columns.astype('int')
    dd_RUL.variable.domain = dd_RUL.columns

    pred_prob = {"RUL": {"scores": dd_RUL}}

    return pred_prob


@pytest.fixture(scope="module")
def cmapss_30_interval_pred_prob():
    filename = os.path.join(DATA_DIR, "cmapss_30_interval_dd.csv")
    dd_RUL = DiscreteDistribution.read_csv(filename, index_col=0)

    pred_prob = {"RUL": {"scores": dd_RUL}}

    return pred_prob


@pytest.fixture(scope="module")
def gmaobus_50_test_df():
    filename = os.path.join(DATA_DIR, "gmaobus_2500_test.csv")
    df = pd.read_csv(filename, sep=";", index_col=0, nrows=50)

    return df


@pytest.fixture(scope="module")
def gmaobus_50_pred_prob():
    filename = os.path.join(DATA_DIR, "gmaobus_2500_dd_ODM_LIBELLE.csv")
    dd_ODM_LIBELEE = DiscreteDistribution.read_csv(
        filename, sep=";", index_col=0, nrows=50)

    filename = os.path.join(DATA_DIR, "gmaobus_2500_dd_FONCTION_NIV2.csv")
    dd_FONCTION_NIV2 = DiscreteDistribution.read_csv(
        filename, sep=";", index_col=0, nrows=50)

    pred_prob = {"ODM_LIBELLE": {"scores": dd_ODM_LIBELEE},
                 "FONCTION_NIV2": {"scores": dd_FONCTION_NIV2}}

    return pred_prob


@pytest.fixture(scope="module")
def gmaobus_2500_test_df():
    filename = os.path.join(DATA_DIR, "gmaobus_2500_test.csv")
    df = pd.read_csv(filename, sep=";", index_col=0)

    return df


@pytest.fixture(scope="module")
def gmaobus_2500_pred_prob():
    filename = os.path.join(DATA_DIR, "gmaobus_2500_dd_ODM_LIBELLE.csv")
    dd_ODM_LIBELEE = DiscreteDistribution.read_csv(
        filename, sep=";", index_col=0)

    filename = os.path.join(DATA_DIR, "gmaobus_2500_dd_FONCTION_NIV2.csv")
    dd_FONCTION_NIV2 = DiscreteDistribution.read_csv(
        filename, sep=";", index_col=0)

    pred_prob = {"ODM_LIBELLE": {"scores": dd_ODM_LIBELEE},
                 "FONCTION_NIV2": {"scores": dd_FONCTION_NIV2}}

    return pred_prob


def test_success_001(gmaobus_2500_test_df, gmaobus_2500_pred_prob):

    success_measure = pfm.SuccessMeasure(map_k=[1, 2, 3, 4, 5])

    assert success_measure.spread_threshold == 1

    success_measure.evaluate(gmaobus_2500_test_df, gmaobus_2500_pred_prob)
    success_dfd = success_measure.result_to_frame()

    for res, success_df in success_dfd.items():
        filename = os.path.join(
            EXPECTED_DIR, f"gmaobus_2500_success_{res}.csv")

        # success_df.to_csv(filename, sep=";")  # if expected changes
        index_col = list(range(len(success_df.index.names)))
        header_lines = list(range(len(success_df.columns.names)))

        success_df_expected = pd.read_csv(filename, sep=";",
                                          header=header_lines,
                                          index_col=index_col)

        np.testing.assert_allclose(success_df.to_numpy(),
                                   success_df_expected.to_numpy())

    success_measure_expected_filename = \
        os.path.join(EXPECTED_DIR,
                     "test_success_001.json")

    # with open(success_measure_expected_filename, 'w') as json_file:
    #     json.dump(success_measure.dict(), json_file)

    with open(success_measure_expected_filename, 'r') as json_file:
        success_measure_expected = pfm.SuccessMeasure(**json.load(json_file))

    assert success_measure.approx_equal(success_measure_expected)

# TODO: Check BUG: Add a test for SuccessMeasure with map_k = [1]
    # ipdb.set_trace()


def test_confusion_matrix_count(gmaobus_2500_test_df, gmaobus_2500_pred_prob):

    confusion_matrix_measure = pfm.ConfusionMatrixMeasure()
    result = confusion_matrix_measure.evaluate(
        gmaobus_2500_test_df, gmaobus_2500_pred_prob)
    for tv in gmaobus_2500_pred_prob.keys():
        assert len(gmaobus_2500_test_df) == sum(
            [sum(d.values()) for d in result.get(tv)])


@pytest.mark.parametrize("thresh,expected",
                         [
                             (0.25, [(24, 41), (17, 2),
                                     (16, 0), (15, 0), (8, 0)]),
                             (0.5, [(24, 41), (27, 3), (26, 2), (21, 2), (20, 2)]),
                             (0.75, [(24, 41), (30, 6),
                                     (30, 3), (29, 3), (27, 3)]),
                             (1, [(24, 41), (30, 47), (30, 47), (30, 48), (30, 49)])
                         ])
def test_spread_threshold_001(gmaobus_50_test_df, gmaobus_50_pred_prob, thresh, expected):

    success_measure = pfm.SuccessMeasure(
        map_k=[1, 2, 3, 4, 5], spread_threshold=thresh)
    success_measure.pred_prob = gmaobus_50_pred_prob
    success_measure.data_test = gmaobus_50_test_df
    result = success_measure.evaluate_pred_success()
    for k, exp in zip(success_measure.map_k, expected):
        assert result[k]['ODM_LIBELLE'].sum() == exp[0]
        assert result[k]['FONCTION_NIV2'].sum() == exp[1]


@pytest.mark.parametrize("thresh,expected",
                         [
                             (0.25, [(728, 2116), (527, 119),
                                     (528, 77), (532, 81), (401, 85)]),
                             (0.5, [(728, 2116), (836, 187),
                                    (888, 117), (932, 131), (955, 136)]),
                             (0.75, [(728, 2116), (1005, 256),
                                     (1164, 267), (1274, 288), (1299, 304)]),
                             (1, [(728, 2116), (1068, 2259),
                                  (1263, 2342), (1401, 2399), (1497, 2427)])
                         ])
def test_spread_threshold_002(gmaobus_2500_test_df, gmaobus_2500_pred_prob, thresh, expected):

    success_measure = pfm.SuccessMeasure(
        map_k=[1, 2, 3, 4, 5], spread_threshold=thresh)
    success_measure.pred_prob = gmaobus_2500_pred_prob
    success_measure.data_test = gmaobus_2500_test_df
    result = success_measure.evaluate_pred_success()
    for k, exp in zip(success_measure.map_k, expected):
        assert result[k]['ODM_LIBELLE'].sum() == exp[0]
        assert result[k]['FONCTION_NIV2'].sum() == exp[1]


def test_map_k(gmaobus_2500_test_df, gmaobus_2500_pred_prob):

    success_measure = pfm.SuccessMeasure(map_k=[1])
    success_measure.pred_prob = gmaobus_2500_pred_prob
    success_measure.data_test = gmaobus_2500_test_df
    result = success_measure.evaluate_pred_success()
    assert result[1]['ODM_LIBELLE'].sum() == 728
    assert result[1]['FONCTION_NIV2'].sum() == 2116


@pytest.mark.parametrize("calc_method,expected",
                         [
                             ('eap', [9.5,
                                      7.159999999999997,
                                      2.75,
                                      1.6200000000000045,
                                      3.125,
                                      0.024999999999977263,
                                      0.5500000000000114,
                                      8.340000000000003,
                                      7.564999999999998,
                                      8.670000000000016,
                                      9.605000000000018,
                                      5.969999999999999]),
                             ('map', [15.5, 0.0, 8.0, 7.5, 8.0,
                                      7.5, 0.0, 7.5, 7.5, 15.5, 15.5, 0.0])
                         ])
def test_absolute_error_001(cmapss_30_interval_df,
                            cmapss_30_interval_pred_prob,
                            calc_method, expected):

    absolute_error_measure = pfm.AbsoluteErrorMeasure(
        calculation_method=calc_method)
    absolute_error_measure.evaluate(
        cmapss_30_interval_df.loc[16:], cmapss_30_interval_pred_prob)
    assert absolute_error_measure.result["ae"]['RUL'].to_list() == expected


@pytest.mark.parametrize("calc_method,expected",
                         [
                             ('eap', [4.560000000000002, 2.259999999999991, 0.21000000000000796, 1.4099999999999966, 1.5500000000000114,
                                      1.0400000000000205, 2.319999999999993, 3.9399999999999977, 4.0, 5.319999999999993, 6.980000000000018, 5.509999999999991]),
                             ('map', [9.0, 2.0, 3.0, 0.0, 2.0,
                                      6.0, 3.0, 4.0, 4.0, 11.0, 12.0, 5.0])
                         ])
def test_absolute_error_002(cmapss_30_numeric_df, cmapss_30_numeric_pred_prob, calc_method, expected):

    absolute_error_measure = pfm.AbsoluteErrorMeasure(
        calculation_method=calc_method)
    absolute_error_measure.evaluate(
        cmapss_30_numeric_df.loc[16:], cmapss_30_numeric_pred_prob)

    assert absolute_error_measure.result["ae"]['RUL'].to_list() == expected
