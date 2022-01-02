import databayes.ohlcv as ohlcv
import yaml
import json
import numpy as np
import pandas as pd
import logging
import pytest
import os
import pkg_resources
import copy

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


logger = logging.getLogger()

# DATA_DIR = "data"

# Util function


# def ppcpt(c): print(gum_pp.cpt2txt(c))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
EXPECTED_PATH = os.path.join(os.path.dirname(__file__), "expected")


@pytest.fixture(scope="module")
def data_btc_usdc_1m_100():

    data_filename = os.path.join(DATA_PATH, "data_btc_usdc_1m_100.csv")
    data_ohlcv_df = pd.read_csv(data_filename, sep=";",
                                index_col="timestamp",
                                parse_dates=["datetime"])
    return data_ohlcv_df


@pytest.fixture(scope="module")
def data_edf_1d_1mo():

    data_filename = os.path.join(DATA_PATH, "data_edf_1d_1mo.csv")
    data_ohlcv_df = pd.read_csv(data_filename, sep=",",
                                index_col="timestamp",
                                parse_dates=["timestamp"])
    return data_ohlcv_df


def test_ohlcv_add_data_001(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    analyser_ref = ohlcv.ohlcvDataAnalyser()
    analyser_ref.add_ohlcv_data(data_ohlcv_df)

    data_ohlcv_p1_df = data_ohlcv_df.iloc[:50]
    data_ohlcv_p2_df = data_ohlcv_df.iloc[50:]

    analyser_1 = ohlcv.ohlcvDataAnalyser()
    analyser_1.add_ohlcv_data(data_ohlcv_p1_df)
    analyser_1.add_ohlcv_data(data_ohlcv_p2_df)

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)


def test_ohlcv_target_001(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    target_time_horizon: [1, 10]
    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_targets()

    # TODO: Check targets computations and test values

    # data_ohlcv_p1_df = data_ohlcv_df.iloc[:50]
    # data_ohlcv_p2_df = data_ohlcv_df.iloc[50:]

    # analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    # analyser_1.add_ohlcv_data(data_ohlcv_p1_df)
    # analyser_1.compute_targets()
    # analyser_1.add_ohlcv_data(data_ohlcv_p2_df)
    # analyser_1.compute_targets()

    # pd.testing.assert_frame_equal(
    #     analyser_ref.ohlcv_df,
    #     analyser_1.ohlcv_df)

    # pd.testing.assert_frame_equal(
    #     analyser_ref.target_df,
    #     analyser_1.target_df)


def test_ohlcv_target_002(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    target_time_horizon: [1, 10]
    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_targets()

    data_ohlcv_p1_df = data_ohlcv_df.iloc[:50]
    data_ohlcv_p2_df = data_ohlcv_df.iloc[50:]

    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_1.add_ohlcv_data(data_ohlcv_p1_df)
    analyser_1.compute_targets()
    analyser_1.add_ohlcv_data(data_ohlcv_p2_df)
    analyser_1.compute_targets()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.target_df,
        analyser_1.target_df)


def test_ohlcv_target_003(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    target_time_horizon: [1, 10]
    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_targets()

    data_ohlcv_p1_df = data_ohlcv_df.iloc[:50]
    data_ohlcv_p2_df = data_ohlcv_df.iloc[50:]

    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_1.add_ohlcv_data(data_ohlcv_p1_df)
    analyser_1.compute_targets()
    analyser_1.add_ohlcv_data(data_ohlcv_p2_df)
    analyser_1.update_targets()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.target_df,
        analyser_1.target_df)


def test_ohlcv_target_004(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    target_time_horizon: [1, 10]
    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_targets()

    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_1.add_ohlcv_data(data_ohlcv_df)
    analyser_1.update_targets()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.target_df,
        analyser_1.target_df)


def test_ohlcv_target_005(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    target_time_horizon: [1, 10]
    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_targets()

    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    for ts in data_ohlcv_df.index:
        analyser_1.add_ohlcv_data(data_ohlcv_df.loc[ts:ts])
        analyser_1.update_targets()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.target_df,
        analyser_1.target_df)


def test_ohlcv_perf_001(data_edf_1d_1mo):

    data_ohlcv_df = data_edf_1d_1mo.copy()

    ohlcv_specs = dict(
        perf_time_horizon=[0, 1, 5],
        control_regular_ts_delta=False,
        ohlcv_names={"open": "Open",
                     "close": "Close",
                     "low": "Low",
                     "high": "High",
                     "volume": "Volume"},
    )

    analyser_ref = ohlcv.ohlcvDataAnalyser(**ohlcv_specs)
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_perf()

    data_expected_filename = os.path.join(EXPECTED_PATH, "ohlcv_perf_001.csv")
    #analyser_ref.perf_df.to_csv(data_expected_filename, sep=",")
    ohlcv_perf_expected_df = \
        pd.read_csv(data_expected_filename,
                    sep=",",
                    index_col="timestamp",
                    parse_dates=["timestamp"])

    pd.testing.assert_frame_equal(
        analyser_ref.perf_df,
        ohlcv_perf_expected_df)

#    ipdb.set_trace()

    analyser_1 = ohlcv.ohlcvDataAnalyser(**ohlcv_specs)
    for ts in data_ohlcv_df.index:
        analyser_1.add_ohlcv_data(data_ohlcv_df.loc[ts:ts])
        analyser_1.update_perf()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.perf_df,
        analyser_1.perf_df)


def test_ohlcv_indic_001(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    indicators:
        hammer_t:
            cls: GeneralizedHammer
            lag: 0
        hammer_tm1:
            cls: GeneralizedHammer
            lag: 1
        hammer_tm5:
            cls: GeneralizedHammer
            lag: 5
        mvq:
          cls: MovingVolumeQuantile
          window_size: 10
        range:
          cls: RangeIndex
          window_size: 20

    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_indicators()


def test_ohlcv_indic_002(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    indicators:
        hammer_t:
            cls: GeneralizedHammer
            lag: 0
        hammer_tm1:
            cls: GeneralizedHammer
            lag: 1
        hammer_tm5:
            cls: GeneralizedHammer
            lag: 5
        mvq:
          cls: MovingVolumeQuantile
          window_size: 10
        range:
          cls: RangeIndex
          window_size: 20


    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_indicators()

    data_ohlcv_p1_df = data_ohlcv_df.iloc[:50]
    data_ohlcv_p2_df = data_ohlcv_df.iloc[50:]

    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_1.add_ohlcv_data(data_ohlcv_p1_df)
    analyser_1.compute_indicators()
    analyser_1.add_ohlcv_data(data_ohlcv_p2_df)
    analyser_1.compute_indicators()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.indic_df,
        analyser_1.indic_df)


def test_ohlcv_indic_003(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    indicators:
        hammer_t:
            cls: GeneralizedHammer
            lag: 0
        hammer_tm1:
            cls: GeneralizedHammer
            lag: 1
        hammer_tm5:
            cls: GeneralizedHammer
            lag: 5
        mvq:
          cls: MovingVolumeQuantile
          window_size: 10
        range:
          cls: RangeIndex
          window_size: 20


    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_indicators()

    data_ohlcv_p1_df = data_ohlcv_df.iloc[:50]
    data_ohlcv_p2_df = data_ohlcv_df.iloc[50:]

    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_1.add_ohlcv_data(data_ohlcv_p1_df)
    analyser_1.update_indicators()
    analyser_1.add_ohlcv_data(data_ohlcv_p2_df)
    analyser_1.update_indicators()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.indic_df,
        analyser_1.indic_df)


def test_ohlcv_indic_004(data_btc_usdc_1m_100):

    data_ohlcv_df = data_btc_usdc_1m_100.copy()

    ohlcv_specs = yaml.load("""
    indicators:
        hammer_t:
            cls: GeneralizedHammer
            lag: 0
        hammer_tm1:
            cls: GeneralizedHammer
            lag: 1
        hammer_tm5:
            cls: GeneralizedHammer
            lag: 5
        mvq:
          cls: MovingVolumeQuantile
          window_size: 10
        range:
          cls: RangeIndex
          window_size: 20


    """, Loader=yaml.SafeLoader)

    analyser_ref = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    analyser_ref.add_ohlcv_data(data_ohlcv_df)
    analyser_ref.compute_indicators()
    analyser_1 = ohlcv.ohlcvDataAnalyser(**copy.deepcopy(ohlcv_specs))
    for ts in data_ohlcv_df.index:
        analyser_1.add_ohlcv_data(data_ohlcv_df.loc[ts:ts])
        analyser_1.update_indicators()

    pd.testing.assert_frame_equal(
        analyser_ref.ohlcv_df,
        analyser_1.ohlcv_df)

    pd.testing.assert_frame_equal(
        analyser_ref.indic_df,
        analyser_1.indic_df)

    # data_train_df = data_df.iloc[:900]
    # data_test_df = data_df.iloc[900:]

    # model.fit(data_train_df)

    # for var in model.model.variables.keys():
    #     var_cct_expected_filename = \
    #         os.path.join(EXPECTED_PATH,
    #                      f"pure_bayesian_model_001_cct_{var}.csv")
    #     var_cct_df = model.model.get_cct(var, transpose=True)
    #     # Save expected data (once validated of course !)
    #     # var_cct_df.to_csv(var_cct_expected_filename,
    #     #                   index=len(var_cct_df.index) > 1)

    #     cct_expected_df = pd.read_csv(var_cct_expected_filename)

    #     if len(var_cct_df.index) > 1:
    #         cct_expected_df.set_index(var_cct_df.index.names,
    #                                   inplace=True)

    #     np.testing.assert_allclose(var_cct_df, cct_expected_df)

    # # Ensure data_test_0 is directly a DataFrame.
    # # Do not use intermediate Series to avoid dtypes problems
    # data_test_0 = \
    #     data_test_df.loc[data_test_df.index[0:1], model.var_features]

    # data_0_pred = model.predict(data_test_0)

    # assert data_0_pred["ret_close_t2"]["scores"].values.tolist() == \
    #     [[0.493368700265252, 0.506631299734748]]
