from databayes.modelling.DiscreteDistribution import DiscreteDistribution
from databayes.modelling.BayesNet import BayesianNetwork, BayesianNetworkModel
import yaml
import pprint
import numpy as np
import pandas as pd
import pyAgrum.lib.pretty_print as gum_pp
import pyAgrum as gum
import logging
import pytest
import os
import sys
import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


logger = logging.getLogger()

DATA_DIR = "data"

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


@pytest.fixture(scope="module")
def cmapss_models_specs():
    filename = os.path.join(DATA_DIR, "cmapss_bn_models.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)


@pytest.fixture(scope="module")
def gmaobus_models_specs():
    filename = os.path.join(DATA_DIR, "gmaobus_bn_models.yaml")
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            if not (logger is None):
                logger.error(exc)


@pytest.fixture(scope="module")
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


def test_BayesianNetwork_001(cmapss_models_specs):

    bnet = BayesianNetwork(**cmapss_models_specs["cmapss_bn_model_01"])

    assert bnet.bn.cpt("cycle_id").nbrDim() == 1
    assert bnet.bn.cpt("cycle_id").var_dims == [2]

    assert bnet.bn.cpt("os_1").nbrDim() == 2
    assert bnet.bn.cpt("os_1").var_dims == [2, 3]


def test_BayesianNetwork_002(cmapss_models_specs, cmapss_data_100_discrete_df):

    bnet = BayesianNetwork(**cmapss_models_specs["cmapss_bn_model_03"])

    bnet.init_from_dataframe(cmapss_data_100_discrete_df)

    # Check BN backend domain
    assert bnet.bn.cpt("cycle_id_d").var_dims == [3, 3, 4]
    assert bnet.bn.cpt("os_1_d").var_dims == [3]
    assert bnet.bn.cpt("sm_2_d").var_dims == [3]

    assert bnet.variables["cycle_id_d"].domain == [
        '(0.0, 1.0]', '(1.0, 2.0]', '(2.0, 3.0]', '(3.0, inf]']
    assert bnet.variables["os_1_d"].domain == [
        '(-inf, -0.0043]', '(-0.0043, 0.004]', '(0.004, inf]']
    assert bnet.variables["sm_2_d"].domain == [
        '(-inf, 641.71]', '(641.71, 643.07]', '(643.07, inf]']

    bnet.fit(data=cmapss_data_100_discrete_df,
             logger=logger)

    # Test CPT backend parameters
    check_cpt_backend(bnet)


def test_BayesianNetwork_003(cmapss_models_specs, cmapss_data_100_discrete_df):

    bnet = BayesianNetwork(**cmapss_models_specs["cmapss_bn_model_02"])

    # Check BN backend domain
    assert bnet.bn.cpt("cycle_id_d").nbrDim() == 1
    assert bnet.bn.cpt("cycle_id_d").var_dims == [4]

    assert bnet.bn.cpt("os_1_d").nbrDim() == 2
    assert bnet.bn.cpt("os_1_d").var_dims == [4, 3]

    assert bnet.bn.cpt("sm_2_d").nbrDim() == 2
    assert bnet.bn.cpt("sm_2_d").var_dims == [4, 3]

    # Basic CPT fit tests
    bnet.fit_cpt(data=cmapss_data_100_discrete_df,
                 var_name="os_1_d",
                 logger=logger)

    cct_os_1_d_df = bnet.get_cct("os_1_d")
    cpt_os_1_d_df = bnet.get_cpt("os_1_d")

    assert cct_os_1_d_df.index.to_list() == bnet.variables["os_1_d"].domain
    assert cpt_os_1_d_df.index.to_list() == bnet.variables["os_1_d"].domain

    assert cct_os_1_d_df.columns.to_list(
    ) == bnet.variables["cycle_id_d"].domain
    assert cpt_os_1_d_df.columns.to_list(
    ) == bnet.variables["cycle_id_d"].domain

    assert cct_os_1_d_df.loc["(-0.0043, 0.004]", "(2.0, 3.0]"] == 0
    assert cpt_os_1_d_df.loc["(-0.0043, 0.004]", "(2.0, 3.0]"] == 0

    assert cct_os_1_d_df.loc["(-inf, -0.0043]", "(3.0, inf]"] == 1
    assert cpt_os_1_d_df.loc["(-inf, -0.0043]",
                             "(3.0, inf]"] == 0.010309278350515464

    bnet.fit_cpt(data=cmapss_data_100_discrete_df,
                 var_name="cycle_id_d",
                 logger=logger)

    cct_cycle_id_d_df = bnet.get_cct("cycle_id_d")
    cpt_cycle_id_d_df = bnet.get_cpt("cycle_id_d")

    assert cct_cycle_id_d_df.index.to_list(
    ) == bnet.variables["cycle_id_d"].domain
    assert cpt_cycle_id_d_df.index.to_list(
    ) == bnet.variables["cycle_id_d"].domain

    assert cct_cycle_id_d_df.to_numpy().flatten().tolist() == [1, 1, 1, 97]
    assert cpt_cycle_id_d_df.to_numpy().flatten().tolist() == [
        0.01, 0.01, 0.01, 0.97]

    # CPT update fit tests
    bnet.fit_cpt(data=cmapss_data_100_discrete_df,
                 var_name="os_1_d",
                 update_fit=True,
                 logger=logger)

    cct_os_1_d_df = bnet.get_cct("os_1_d", flatten=True)

    assert cct_os_1_d_df.loc[("(-0.0043, 0.004]", "(0.0, 1.0]")] == 2
    assert cct_os_1_d_df.loc[("(-0.0043, 0.004]", "(3.0, inf]")] == 192
    assert cct_os_1_d_df.to_numpy().flatten().tolist() == [
        0, 0, 2, 2, 2, 2, 0, 192, 0, 0, 0, 0]

    cpt_os_1_d_df = bnet.get_cpt("os_1_d", flatten=True)
    assert cpt_os_1_d_df.to_list() == [
        0.0, 0.0, 1.0, 0.010309278350515464, 1.0, 1.0, 0.0, 0.9896907216494846, 0.0, 0.0, 0.0, 0.0]

    # test cycle_id_d variables
    bnet.fit_cpt(data=cmapss_data_100_discrete_df,
                 var_name="cycle_id_d",
                 update_fit=True,
                 logger=logger)

    cct_cycle_id_d_df = bnet.get_cct("cycle_id_d", flatten=True)
    assert cct_cycle_id_d_df.to_list() == [2, 2, 2, 194]

    cpt_cycle_id_d_df = bnet.get_cpt("cycle_id_d", flatten=True)
    assert cpt_cycle_id_d_df.to_list() == [0.01, 0.01, 0.01, 0.97]

    # CPT update fit tests with decay
    bnet.fit_cpt(data=cmapss_data_100_discrete_df,
                 var_name="os_1_d",
                 update_fit=True,
                 update_decay=0.75,
                 logger=logger)

    cct_os_1_d_df = bnet.get_cct("os_1_d", flatten=True)
    assert cct_os_1_d_df.to_list() == [
        0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 0.0, 144.0, 0.0, 0.0, 0.0, 0.0]
    cpt_os_1_d_df = bnet.get_cpt("os_1_d", flatten=True)
    assert cpt_os_1_d_df.to_list() == [
        0.0, 0.0, 1.0, 0.010309278350515464, 1.0, 1.0, 0.0, 0.9896907216494846, 0.0, 0.0, 0.0, 0.0]

    bnet.fit_cpt(data=cmapss_data_100_discrete_df,
                 var_name="cycle_id_d",
                 update_fit=True,
                 update_decay=0.75,
                 logger=logger)

    cct_cycle_id_d_df = bnet.get_cct("cycle_id_d", flatten=True)
    assert cct_cycle_id_d_df.to_list() == [1.5, 1.5, 1.5, 145.5]

    cpt_cycle_id_d_df = bnet.get_cpt("cycle_id_d", flatten=True)
    assert cpt_cycle_id_d_df.to_list() == [0.01, 0.01, 0.01, 0.97]

    cmapss_data_100_discrete_bis_df = \
        cmapss_data_100_discrete_df.iloc[:25]

    bnet.fit(data=cmapss_data_100_discrete_bis_df,
             update_fit=True,
             update_decay=0.5,
             logger=logger)

    cct_os_1_d_df = bnet.get_cct("os_1_d", flatten=True)
    cct_sm_2_d_df = bnet.get_cct("sm_2_d", flatten=True)
    cct_cycle_id_d_df = bnet.get_cct("cycle_id_d", flatten=True)

    assert cct_os_1_d_df.to_list() == [
        0.0, 0.0, 1.75, 1.75, 1.75, 1.75, 0.0, 93.0, 0.0, 0.0, 0.0, 0.0]
    assert cct_sm_2_d_df.to_list() == [0, 0, 0, 1, 1, 1, 1, 21, 0, 0, 0, 0]
    assert cct_cycle_id_d_df.to_list() == [1.75, 1.75, 1.75, 94.75]

    cpt_os_1_d_df = bnet.get_cpt("os_1_d", flatten=True)
    cpt_sm_2_d_df = bnet.get_cpt("sm_2_d", flatten=True)
    cpt_cycle_id_d_df = bnet.get_cpt("cycle_id_d", flatten=True)

    assert cpt_os_1_d_df.to_list() == [
        0.0, 0.0, 1.0, 0.018469656992084433, 1.0, 1.0, 0.0, 0.9815303430079155, 0.0, 0.0, 0.0, 0.0]
    assert cpt_sm_2_d_df.to_list() == [
        0.0, 0.0, 0.0, 0.045454545454545456, 1.0, 1.0, 1.0, 0.9545454545454546, 0.0, 0.0, 0.0, 0.0]
    assert cpt_cycle_id_d_df.to_list() == [0.0175, 0.0175, 0.0175, 0.9475]


def test_BayesianNetwork_004(cmapss_models_specs, cmapss_data_100_discrete_df):

    bnet = BayesianNetwork(**cmapss_models_specs["cmapss_bn_model_02"])

    bnet.fit(data=cmapss_data_100_discrete_df,
             logger=logger)

    # Test CPT backend parameters
    check_cpt_backend(bnet)

    data_test = cmapss_data_100_discrete_df.loc[:0, ["os_1_d", "cycle_id_d"]]

    # ipdb.set_trace()

    pred_test = bnet.predict(data=data_test,
                             var_targets=["sm_2_d"],
                             map_k=1)

    assert bnet.get_cpt("sm_2_d").loc[:, data_test["cycle_id_d"].loc[0]].idxmax() == \
        pred_test["sm_2_d"]["map"]["map_1"].iloc[0]
    assert bnet.get_cpt("sm_2_d").loc[:, data_test["cycle_id_d"].loc[0]].to_list() == \
        pred_test["sm_2_d"]["scores"].iloc[0].to_list()

    data_test = cmapss_data_100_discrete_df[["sm_2_d"]]

    pred_test = bnet.predict(data=data_test,
                             var_targets=["os_1_d", "cycle_id_d"])

    assert len(pred_test) == 2
    assert "cycle_id_d" in pred_test.keys()
    pred_cycle_id_d = pred_test["cycle_id_d"]["scores"].drop_duplicates()
    assert len(pred_cycle_id_d) == 2
    assert pred_cycle_id_d.loc[0].to_list() == [
        0.010101010101010102, 0.010101010101010102, 0.010101010101010102, 0.9696969696969697]
    assert pred_cycle_id_d.loc[9].to_list() == [0.0, 0.0, 0.0, 1.0]
    pred_os_1_d = pred_test["os_1_d"]["scores"].drop_duplicates()
    assert len(pred_os_1_d) == 2
    assert pred_os_1_d.loc[0].to_list(
    ) == [0.02009788607726752, 0.9799021139227325, 0.0]
    assert pred_os_1_d.loc[9].to_list(
    ) == [0.010309278350515464, 0.9896907216494846, 0.0]


def test_BayesianNetwork_005(cmapss_data_100_discrete_df):

    bnet = BayesianNetwork(name="cmapss")

    bnet.add_variable(name="os_1_d",
                      domain=['(-inf, -0.0043]', '(-0.0043, 0.004]', '(0.004, inf]'])

    bnet.add_variable(name="system_id",
                      domain=['s1'])

    bnet.add_variable(name="sm_2_d",
                      bins=[-np.inf, 641.71, 643.07, np.inf])

    bnet.add_variable(name="cycle_id_d",
                      domain=['(0.0, 1.0]', '(1.0, 2.0]', '(2.0, 3.0]', '(3.0, inf]'])

    # Test CCT initialization
    for var_name in bnet.variables.keys():
        var_domain_size = len(bnet.variables[var_name].domain)
        assert bnet.get_cct(var_name, flatten=True).to_list() == [
            0]*var_domain_size

    bnet.add_parents(var_name="cycle_id_d",
                     parents=["os_1_d", "sm_2_d"])

    data_train_1_df = cmapss_data_100_discrete_df.loc[:50]
    bnet.fit(data=data_train_1_df,
             update_fit=True,
             logger=logger)

    check_cpt_backend(bnet)

    assert bnet.get_cct("cycle_id_d", flatten=True).to_list() == [
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 46, 0, 0, 0, 0]
    assert bnet.get_cpt("cycle_id_d", flatten=True).to_list() == [0.25, 0.0, 0.25, 0.0, 0.020833333333333332, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.25, 0.0,
                                                                  0.020833333333333332, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 1.0, 0.9583333333333334, 0.25, 0.25, 0.25, 0.25]

    assert bnet.get_cct("os_1_d", flatten=True).to_list() == [2, 49, 0]
    assert bnet.get_cpt("os_1_d", flatten=True).to_list() == [
        0.0392156862745098, 0.9607843137254902, 0.0]

    assert bnet.get_cct("sm_2_d", flatten=True).to_list() == [1, 50, 0]
    assert bnet.get_cpt("sm_2_d", flatten=True).to_list() == [
        0.0196078431372549, 0.9803921568627451, 0.0]

    data_train_2_df = cmapss_data_100_discrete_df.loc[51:]
    bnet.fit(data=data_train_2_df,
             update_fit=True,
             logger=logger)

    # Test CPT backend consistency
    check_cpt_backend(bnet)

    assert bnet.get_cct("cycle_id_d", flatten=True).to_list() == [
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 95, 0, 0, 0, 0]
    assert bnet.get_cpt("cycle_id_d", flatten=True).to_list() == [0.25, 0.0, 0.25, 0.0, 0.010309278350515464, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.25, 0.0,
                                                                  0.010309278350515464, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 1.0, 0.979381443298969, 0.25, 0.25, 0.25, 0.25]

    assert bnet.get_cct("os_1_d", flatten=True).to_list() == [2, 98, 0]
    assert bnet.get_cpt("os_1_d", flatten=True).to_list() == [0.02, 0.98, 0.0]

    assert bnet.get_cct("sm_2_d", flatten=True).to_list() == [1, 99, 0]
    assert bnet.get_cpt("sm_2_d", flatten=True).to_list() == [0.01, 0.99, 0.0]

    data_test = cmapss_data_100_discrete_df.loc[:0, ["os_1_d", "sm_2_d"]]

    pred_test = bnet.predict(data=data_test,
                             var_targets=["cycle_id_d"])

    assert bnet.get_cpt("cycle_id_d", transpose=True).loc[tuple(data_test.loc[0].to_list(
    ))].to_list() == pred_test["cycle_id_d"]["scores"].iloc[0].to_list()

    data_test = cmapss_data_100_discrete_df[["os_1_d", "sm_2_d"]]

    pred_test = bnet.predict(data=data_test,
                             var_targets=["cycle_id_d"])

    for idx, data in data_test.iterrows():

        assert bnet.get_cpt("cycle_id_d", transpose=True).loc[tuple(
            data.to_list())].to_list() == pred_test["cycle_id_d"]["scores"].iloc[idx].to_list()


def test_BayesianNetwork_006(gmaobus_models_specs, gmaobus_om_ot_100_df):

    bnet = BayesianNetwork(name="gmaobus_nb",
                           **gmaobus_models_specs["bn_naive_bayes_01"]["model"])

    bnet.init_from_dataframe(gmaobus_om_ot_100_df)

    data_train_df = gmaobus_om_ot_100_df[:50]
    data_test_df = gmaobus_om_ot_100_df[25:]

    bnet.fit(data_train_df)

    cpt_sig_organe_html_filename = "gmaobus_cpt_sig_organe.html"
    try:
        os.remove(cpt_sig_organe_html_filename)
    except OSError:
        pass

    bnet.cpt_to_html("SIG_ORGANE", filename=cpt_sig_organe_html_filename)

    assert os.path.exists(cpt_sig_organe_html_filename)

    bnet_html_filename = "gmaobus_bn.html"
    try:
        os.remove(bnet_html_filename)
    except OSError:
        pass

    bnet.to_html(filename=bnet_html_filename)

    assert os.path.exists(bnet_html_filename)


def test_BayesianNetwork_007(gmaobus_models_specs, gmaobus_om_ot_100_df):

    bnet_ml_specs = gmaobus_models_specs["bn_naive_bayes_01"]

    bnet = BayesianNetwork(name="gmaobus_nb",
                           **bnet_ml_specs["model"])

    bnet.init_from_dataframe(gmaobus_om_ot_100_df)

    data_train_df = gmaobus_om_ot_100_df[:50]
    data_test_df = gmaobus_om_ot_100_df.loc[25:,
                                            bnet_ml_specs.get("var_features")]

    bnet.fit(data_train_df)

    pred_test = bnet.predict(data_test_df, var_targets=["ODM_LIBELLE"])

    assert (np.ceil(pred_test.get("ODM_LIBELLE").get("scores").sum(axis=1)).astype(int) ==
            pred_test.get("ODM_LIBELLE").get("comp_ok").astype(int)).all(None)


def test_BayesianNetwork_008(gmaobus_models_specs, gmaobus_om_ot_100_df):

    bnet_ml_specs = gmaobus_models_specs["bn_naive_bayes_02"]

    bnet = BayesianNetwork(name="gmaobus_nb",
                           **bnet_ml_specs["model"])

    bnet.init_from_dataframe(gmaobus_om_ot_100_df)

    data_train_df = gmaobus_om_ot_100_df[:50]
    data_test_df = gmaobus_om_ot_100_df.loc[25:,
                                            bnet_ml_specs.get("var_features")]

    bnet.fit(data_train_df)

    pred_test = bnet.predict(data_test_df, var_targets=["ODM_LIBELLE"])

    assert (np.ceil(pred_test.get("ODM_LIBELLE").get("scores").sum(axis=1)).astype(int) ==
            pred_test.get("ODM_LIBELLE").get("comp_ok").astype(int)).all(None)


def test_BayesianNetwork_009(gmaobus_models_specs, gmaobus_om_ot_100_df):

    bnet_ml_specs = gmaobus_models_specs["bn_naive_bayes_03"]

    bnet = BayesianNetwork(name="gmaobus_nb",
                           **bnet_ml_specs["model"])

    bnet.init_from_dataframe(gmaobus_om_ot_100_df)

    data_train_df = gmaobus_om_ot_100_df[:50]
    data_test_df = gmaobus_om_ot_100_df.loc[25:,
                                            bnet_ml_specs.get("var_features")]

    bnet.fit(data_train_df)

    pred_test = bnet.predict(data_test_df, var_targets=["ODM_LIBELLE"])

    assert (np.ceil(pred_test.get("ODM_LIBELLE").get("scores").sum(axis=1)).astype(int) ==
            pred_test.get("ODM_LIBELLE").get("comp_ok").astype(int)).all(None)


def test_BayesianNetworkModel_001(cmapss_data_100_discrete_df):

    bnet = BayesianNetwork(name="BayesianNetworkModel_001")

    bnet.add_variable(name="os_1_d",
                      domain=['(-inf, -0.0043]', '(-0.0043, 0.004]', '(0.004, inf]'])

    bnet.add_variable(name="system_id",
                      domain=['s1'])

    bnet.add_variable(name="sm_2_d",
                      bins=[-np.inf, 641.71, 643.07, np.inf])

    bnet.add_variable(name="cycle_id_d",
                      domain=['(0.0, 1.0]', '(1.0, 2.0]', '(2.0, 3.0]', '(3.0, inf]'])

    bnet.add_parents(var_name="cycle_id_d",
                     parents=["os_1_d", "sm_2_d"])

    # Test copy constructor
    bnet_copy = BayesianNetwork(**bnet.dict())

    data_train_df = cmapss_data_100_discrete_df.loc[:50]

    # ipdb.set_trace()
    bnet.fit(data=data_train_df,
             update_fit=True,
             logger=logger)

    var_targets = ["cycle_id_d"]
    var_features = ["os_1_d", "sm_2_d"]

    bnet_ml = BayesianNetworkModel(
        var_targets=var_targets,
        var_features=var_features,
        model=bnet_copy.dict(),
        fit_parameters={
            "update_fit": True,
        }
    )

    bnet_ml.fit(data_train_df)

    data_test_df = cmapss_data_100_discrete_df.loc[50:]
    data_test_feat_df = data_test_df[var_features]

    pred_test = bnet.predict(data=data_test_feat_df,
                             var_targets=var_targets)

    pred_test_bis = bnet_ml.predict(data=data_test_feat_df)

    for var in pred_test.keys():
        for pred_key in pred_test.get(var, {}).keys():
            pred_res = pred_test.get(var, {}).get(pred_key, {})
            pred_res_bis = pred_test_bis.get(var, {}).get(pred_key, {})
            assert (pred_res.values == pred_res_bis.values).all(None)


def test_BayesianNetworkModel_002(gmaobus_models_specs, gmaobus_om_ot_100_df):

    # Use BayesNetModel class
    bnet_ml = BayesianNetworkModel(**gmaobus_models_specs["bn_naive_bayes_03"])

    bnet_ml.init_from_dataframe(gmaobus_om_ot_100_df)

    data_train_df = gmaobus_om_ot_100_df[:50]
    data_test_df = gmaobus_om_ot_100_df.loc[25:]

    data_train_adapt_df = bnet_ml.model.adapt_data(data_train_df)
    data_test_adapt_df = bnet_ml.model.adapt_data(data_test_df)

    for var in bnet_ml.model.variables.keys():
        assert data_train_adapt_df[var].cat.categories.dtype == data_test_adapt_df[var].cat.categories.dtype
        assert (data_train_adapt_df[var].cat.categories ==
                data_test_adapt_df[var].cat.categories).all()

    bnet_ml.fit(data_train_df)

    # Use BayesNet class directly
    bnet = BayesianNetwork(name="gmaobus_nb",
                           **gmaobus_models_specs["bn_naive_bayes_03"]["model"])

    bnet.init_from_dataframe(gmaobus_om_ot_100_df)

    bnet.fit(data_train_df)

    assert bnet_ml.model.is_num_equal(bnet)

    pred_test = bnet.predict(data_test_df[bnet_ml.var_features],
                             var_targets=["ODM_LIBELLE"])

    map_k = 4
    pred_test_bis = bnet_ml.predict(data_test_df, map_k=map_k)

    for var in pred_test.keys():
        for pred_key in pred_test.get(var, {}).keys():
            pred_res = pred_test.get(var, {}).get(pred_key, {})
            pred_res_bis = pred_test_bis.get(var, {}).get(pred_key, {})
            np.testing.assert_allclose(pred_res, pred_res_bis)

    assert isinstance(pred_test_bis["ODM_LIBELLE"]
                      ["scores"], DiscreteDistribution)

    assert "map" in pred_test_bis["ODM_LIBELLE"].keys()
    assert len(pred_test_bis["ODM_LIBELLE"]["map"].columns) == 4
    assert pred_test_bis["ODM_LIBELLE"]["map"].index.name == "ODM_LIBELLE"
