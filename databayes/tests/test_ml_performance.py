from import_pkg import MLPerformance, MLModel, BayesianNetworkModel
import yaml
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

# Util function


def ppcpt(c): print(gum_pp.cpt2txt(c))


def check_cpt_backend(bnet):
    if bnet.backend == "pyagrum":
        # Test CPT backend parameters
        for var_name in bnet.variables.keys():
            cpt_cur = bnet.get_cpt(var_name, flatten=True)
            bn_cpt_cur = bnet.bn.cpt(var_name)
            bn_cpt_I_cur = gum.Instantiation(bn_cpt_cur)
            for idx, val in enumerate(cpt_cur.index.to_frame().to_dict("records")):
                bn_cpt_I_cur.fromdict(val)
                assert bn_cpt_cur.get(bn_cpt_I_cur) == cpt_cur.iloc[idx]
    else:
        raise ValueError(f"Backend {bnet.backend} not supported for this test")


@pytest.fixture
def cmapss_models_specs():
    filename = os.path.join(DATA_DIR, "cmapss_bn_models.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)


@pytest.fixture
def gmaobus_models_specs():
    filename = os.path.join(DATA_DIR, "gmaobus_bn_models.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)


@pytest.fixture
def gmaobus_ml_performance_specs():
    filename = os.path.join(DATA_DIR, "gmaobus_ml_performance.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)


@pytest.fixture
def cmapss_data_100_df():
    filename = os.path.join(DATA_DIR, "cmapss_data_100.csv")
    df = pd.read_csv(filename)

    return df


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


@pytest.fixture
def cmapss_data_100_discrete_df():
    filename = os.path.join(DATA_DIR, "cmapss_data_100.csv")
    df = pd.read_csv(filename)

    df["system_id"] = ("s" + df["system_id"].astype(str)).astype("category")
    df["cycle_id_d"] = pd.cut(
        df["cycle_id"], bins=list(np.arange(0, 4, 1)) + [np.inf])

    bins_os_1 = [-np.inf] + list(np.linspace(df["os_1"].min(),
                                             df["os_1"].max(),
                                             2)) + [np.inf]
    df["os_1_d"] = pd.cut(df["os_1"], bins=bins_os_1)

    bins_sm_2 = [-np.inf] + list(np.linspace(df["sm_2"].min(),
                                             df["sm_2"].max(),
                                             2)) + [np.inf]
    df["sm_2_d"] = pd.cut(df["sm_2"], bins=bins_sm_2)

    df = df[["system_id", "cycle_id_d", "os_1_d", "sm_2_d"]]

    return df


@pytest.fixture
def gmaobus_om_ot_10000_discrete_df(gmaobus_om_ot_10000_df):

    df = gmaobus_om_ot_10000_df.copy()

    df['KILOMETRAGE'] = pd.qcut(df['KILOMETRAGE'], 4).astype('str')\
                                                     .astype('category')

    df["GARANTIE"] = df["GARANTIE"].astype(str)

    df["ODM_LIBELLE"] = df["ODM_LIBELLE"].str.replace("  ", " ")

    return df


@pytest.mark.slow
def test_MLPerformance_001(gmaobus_models_specs,
                           gmaobus_ml_performance_specs,
                           gmaobus_om_ot_10000_discrete_df):

    data = gmaobus_om_ot_10000_discrete_df.copy()

    ml_perf_specs = gmaobus_ml_performance_specs.get(
        "ml_performance_analysis_01")

    # Use BayesNetModel class
    bnet_ml = BayesianNetworkModel(**gmaobus_models_specs["bn_naive_bayes_04"])

    bnet_ml.init_from_dataframe(data)

    ml_perf = MLPerformance(model=bnet_ml, **ml_perf_specs)

    ml_perf.run(
        data, logger=logger, progress_mode=True)

    ml_perf_expected_filename = \
        os.path.join(EXPECTED_DIR,
                     "test_MLPerformance_001.json")

    # with open(ml_perf_expected_filename, 'w') as json_file:
    #     json.dump(ml_perf.dict(), json_file)

    with open(ml_perf_expected_filename, 'r') as json_file:
        ml_perf_expected_specs = json.load(json_file)

    ml_perf_expected = MLPerformance(**ml_perf_expected_specs)

    assert ml_perf.measures_approx_equal(ml_perf_expected)


@pytest.mark.parametrize("test_input,expected,ml_perf_specs",
                         [
                             (pd.DataFrame(range(1)), [],
                              "ml_performance_analysis_03"),
                             (pd.DataFrame(range(1)), [],
                              "ml_performance_analysis_04"),
                             (pd.DataFrame(range(1)), [],
                              "ml_performance_analysis_05"),
                             (pd.DataFrame(range(1)), [],
                              "ml_performance_analysis_06"),
                             (pd.DataFrame(range(1)), [],
                              "ml_performance_analysis_07"),
                             (pd.DataFrame(range(1)), [],
                              "ml_performance_analysis_08"),
                             (pd.DataFrame(range(2)), [([0], [1])],
                              "ml_performance_analysis_03"),
                             (pd.DataFrame(range(2)), [([0], [1])],
                              "ml_performance_analysis_04"),
                             (pd.DataFrame(range(2)), [],
                              "ml_performance_analysis_05"),
                             (pd.DataFrame(range(2)), [([0], [1])],
                              "ml_performance_analysis_06"),
                             (pd.DataFrame(range(2)), [([0], [1])],
                              "ml_performance_analysis_07"),
                             (pd.DataFrame(range(2)), [([0], [1])],
                              "ml_performance_analysis_08"),
                             (pd.DataFrame(range(5)), [([0, 1], [2, 3, 4])],
                              "ml_performance_analysis_03"),
                             (pd.DataFrame(range(5)), [([1], [2, 3, 4])],
                              "ml_performance_analysis_04"),
                             (pd.DataFrame(range(5)), [],
                              "ml_performance_analysis_05"),
                             (pd.DataFrame(range(5)), [([0, 1], [2, 3, 4])],
                              "ml_performance_analysis_06"),
                             (pd.DataFrame(range(5)), [([0, 1], [2]), ([1, 2], [3]), ([2, 3], [4])],
                              "ml_performance_analysis_07"),
                             (pd.DataFrame(range(5)), [([0, 1], [2, 3, 4])],
                              "ml_performance_analysis_08"),
                             (pd.DataFrame(range(10)), [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])],
                              "ml_performance_analysis_03"),
                             (pd.DataFrame(range(10)), [([2, 3, 4], [5, 6, 7, 8, 9])],
                              "ml_performance_analysis_04"),
                             (pd.DataFrame(range(10)), [([2, 3, 4], [5]), ([3, 4, 5], [6]), ([4, 5, 6], [7]), ([5, 6, 7], [8]), ([6, 7, 8], [9])],
                              "ml_performance_analysis_05"),
                             (pd.DataFrame(range(10)), [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])],
                              "ml_performance_analysis_06"),
                             (pd.DataFrame(range(10)), [([0, 1, 2, 3, 4], [5]), ([1, 2, 3, 4, 5], [6]), ([2, 3, 4, 5, 6], [7]), ([3, 4, 5, 6, 7], [8]),
                                                        ([4, 5, 6, 7, 8], [9])],
                              "ml_performance_analysis_07"),
                             (pd.DataFrame(range(10)), [([1, 2, 3, 4], [5, 6, 7, 8, 9])],
                              "ml_performance_analysis_08"),
                             (pd.DataFrame(range(20)), [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])],
                              "ml_performance_analysis_03"),
                             (pd.DataFrame(range(20)), [([5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])],
                              "ml_performance_analysis_04"),
                             (pd.DataFrame(range(20)), [([5, 6, 7, 8, 9], [10, 11]), ([7, 8, 9, 10, 11], [12, 13]), ([9, 10, 11, 12, 13], [14, 15]),
                                                        ([11, 12, 13, 14, 15], [16, 17]), ([13, 14, 15, 16, 17], [18, 19])],
                              "ml_performance_analysis_05"),
                             (pd.DataFrame(range(20)), [([3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])],
                              "ml_performance_analysis_06"),
                             (pd.DataFrame(range(20)), [([5, 6, 7, 8, 9], [10]), ([6, 7, 8, 9, 10], [11]), ([7, 8, 9, 10, 11], [12]), ([8, 9, 10, 11, 12], [13]),
                                                        ([9, 10, 11, 12, 13], [14]), ([10, 11, 12, 13, 14], [
                                                            15]), ([11, 12, 13, 14, 15], [16]),
                                                        ([12, 13, 14, 15, 16], [17]), ([13, 14, 15, 16, 17], [18]), ([14, 15, 16, 17, 18], [19])],
                              "ml_performance_analysis_07"),
                             (pd.DataFrame(range(20)), [([2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16]),
                                                        ([9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19])],
                              "ml_performance_analysis_08"),
                         ])
def test_MLPerformance_sliding_split(gmaobus_ml_performance_specs,
                                     test_input,
                                     expected,
                                     ml_perf_specs):

    ml_perf = MLPerformance(
        model=MLModel(), **gmaobus_ml_performance_specs[ml_perf_specs])
    data_train_idx, data_test_idx = ml_perf.split_data(test_input)
    slide = ml_perf.sliding_split(data_train_idx, data_test_idx)

    assert [(d_train_idx, d_test_idx)
            for d_train_idx, d_test_idx in slide] == expected
