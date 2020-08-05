# ==========================================================
# Classe de test pour les Distributions Discrètes
# josquin.foulliaron@edgemind.net
# ==========================================================
import os
import logging

from databayes.modelling.DiscreteDistribution import DiscreteDistribution
import pytest
import numpy as np
import pandas as pd

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


logger = logging.getLogger()

DATA_DIR = "data"
EXPECTED_DIR = "expected"

INF = float("inf")


def assert_get_map(dd, nlargest=1):
    for k in range(1, nlargest + 1):
        map_names = [f"map_{i}" for i in range(1, k+1)]
        map_k_check_df = \
            dd.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                         .iloc[:k].index,
                                         index=map_names), axis=1)
        map_k_df = dd.get_map(nlargest=k)

        assert (map_k_check_df == map_k_df).all(None)

    return map_k_df


def test_dd_value_01():
    dd = DiscreteDistribution(domain=[1, 3, 4, 5, 7, 10],
                              index=["X1", "X2"],
                              probs=[
        [0.1, 0.2, 0.05, 0.3, 0.05, 0.3],
        [0.7, 0.05, 0.05, 0.1, 0, 0.1]
    ])

    test_io = [
        {"params": {"value": 5}, "expected": [0.3, 0.1]},
        {"params": {"value": 1}, "expected": [0.1, 0.7]},
        {"params": {"value": 10}, "expected": [0.3, 0.1]},
        {"params": {"value": 34}, "expected": [0, 0]},
        {"params": {"value": -6}, "expected": [0, 0]},
        {"params": {"value": 6}, "expected": [0, 0]}
    ]
    for io in test_io:
        assert dd.get_prob_from_value(
            **io["params"]).tolist() == io["expected"]

    test_io = [
        {"params": {"bmin": 3, "bmax": 7}, "expected": [0.6, 0.2]},
        {"params": {"bmin": 0, "bmax": 10}, "expected": [1, 1]},
        {"params": {"bmin": 3, "bmax": 14}, "expected": [0.9, 0.3]},
        {"params": {"bmin": -4, "bmax": 0}, "expected": [0, 0]},
        {"params": {"bmin": 2, "bmax": 6}, "expected": [0.55, 0.2]},
        {"params": {"bmin": 12, "bmax": 16}, "expected": [0, 0]},
        {"params": {"bmin": -12, "bmax": -6}, "expected": [0, 0]},
    ]
    for io in test_io:
        res = dd.get_prob_from_interval(**io["params"])
        np.testing.assert_allclose(res, io["expected"])

    # Test expectancy
    np.testing.assert_allclose(dd.E(),
                               [5.75, 2.55])


# Interval distribution tests
def test_dd_interval_01():
    dd = DiscreteDistribution(bins=[-INF, 0, 1, 3, 4, 7, 10],
                              index=["X1", "X2"],
                              probs=[
                                  [0.1, 0.2, 0.05, 0.3, 0.05, 0.3],
                                  [0.7, 0.05, 0.05, 0.1, 0, 0.1]
    ])

    test_io = [
        {"params": {"value": 5, "interval_zero_prob": False},
            "expected": [0.05, 0]},
        {"params": {"value": 1, "interval_zero_prob": False},
            "expected": [0.20, 0.05]},
        {"params": {"value": 10, "interval_zero_prob": False},
            "expected": [0.3, 0.1]},
        {"params": {"value": 34, "interval_zero_prob": False},
            "expected": [0, 0]},
        {"params": {"value": -6, "interval_zero_prob": False},
            "expected": [0.1, 0.7]},
        {"params": {"value": 6, "interval_zero_prob": False},
            "expected": [0.05, 0]},
        {"params": {"value": 6}, "expected": [0, 0]},
        {"params": {"value": -6}, "expected": [0, 0]},
        {"params": {"value": 34}, "expected": [0, 0]}
    ]
    for io in test_io:
        assert dd.get_prob_from_value(
            **io["params"]).tolist() == io["expected"]

    test_io = [
        {"params": {"bmin": 3, "bmax": 7}, "expected": [0.35, 0.1]},
        {"params": {"bmin": 0, "bmax": 10}, "expected": [0.9, 0.3]},
        {"params": {"bmin": 3, "bmax": 14}, "expected": [0.65, 0.2]},
        {"params": {"bmin": -4, "bmax": 0}, "expected": [0, 0]},
        {"params": {"bmin": 3.25, "bmax": 3.75}, "expected": [0.15, 0.05]},
        {"params": {"bmin": 3.5, "bmax": 4.5},
            "expected": [0.15 + 0.05*0.5/3, 0.05]},
        {"params": {"bmin": 2, "bmax": 6}, "expected": [
            0.5*0.05 + 0.3 + 2/3*0.05, 0.5*0.05 + 0.1]},
        {"params": {"bmin": 12, "bmax": 16}, "expected": [0, 0]},
        {"params": {"bmin": -12, "bmax": -6}, "expected": [0, 0]},
        {"params": {"bmin": -12, "bmax": -6, "lower_bound": -12},
            "expected": [0.05, 0.35]},
        {"params": {"bmin": 5.5, "bmax": 8, "upper_bound": 9},
            "expected": [0.025 + 0.15, 0.05]},
    ]
    for io in test_io:
        res = dd.get_prob_from_interval(**io["params"])
        np.testing.assert_allclose(res, io["expected"])

    # Test expectancy
    np.testing.assert_allclose(dd.E(),
                               [-float("inf"), -float("inf")])

    np.testing.assert_allclose(dd.E(lower_bound=-12),
                               [3.475, -2.875])


# Labelized distribution
def test_dd_label_01():
    dd = DiscreteDistribution(domain=["a", "b", "c", "d", "e", "f"],
                              index=["X1", "X2"],
                              probs=[
                                  [0.1, 0.2, 0.05, 0.3, 0.5, 0.3],
                                  [0.7, 0.05, 0.05, 0.1, 0, 0.1]
    ])

    test_io = [
        {"params": {"value": "a"}, "expected": [0.1, 0.7]},
        {"params": {"value": "b"}, "expected": [0.2, 0.05]},
        {"params": {"value": "c"}, "expected": [0.05, 0.05]},
        {"params": {"value": "d"}, "expected": [0.3, 0.1]},
        {"params": {"value": "e"}, "expected": [0.5, 0]},
        {"params": {"value": "f"}, "expected": [0.3, 0.1]},
        {"params": {"value": "g"}, "expected": [0, 0]},
        {"params": {"value": ["a", "c"]}, "expected": [0.15, 0.75]},
    ]
    for io in test_io:
        np.testing.assert_allclose(dd.get_prob_from_value(**io["params"]),
                                   io["expected"])


@pytest.fixture(scope="module")
def gmaobus_odm_libelle_01_dd():
    filename = os.path.join(DATA_DIR, "gmaobus_odm_libelle_01_dd.csv")
    dd = DiscreteDistribution.read_csv(filename, sep=";", index_col="name")
    dd.variable.name = "ODM_LIBELLE"
    return dd


def test_check_dd_prop(gmaobus_odm_libelle_01_dd):

    assert gmaobus_odm_libelle_01_dd.variable.domain == list(
        gmaobus_odm_libelle_01_dd.columns)
    assert gmaobus_odm_libelle_01_dd.variable.domain_type == "label"
    assert gmaobus_odm_libelle_01_dd.variable.name == "ODM_LIBELLE"
    assert gmaobus_odm_libelle_01_dd.index.name == None

    # For k == 3, gmaobus_odm_libelle_01_dd has columns with identical values
    # The checking method sorts identical valued columns in different order than
    # the method in get_map
    assert_get_map(gmaobus_odm_libelle_01_dd, 2)

    k = len(gmaobus_odm_libelle_01_dd.columns) + 10
    map_k = gmaobus_odm_libelle_01_dd.get_map(k)
    cats_ref = map_k["map_1"].cat.categories

    assert all([(cats_ref == s.cat.categories).all(None)
                for v, s in map_k.items()])

    dd_bis = DiscreteDistribution(domain=gmaobus_odm_libelle_01_dd.variable.domain,
                                  index=gmaobus_odm_libelle_01_dd.index)

    assert (dd_bis.values == 0).all()
    assert dd_bis.variable.domain == gmaobus_odm_libelle_01_dd.variable.domain
    assert (dd_bis.index == gmaobus_odm_libelle_01_dd.index).all(None)


# Labelized distribution
def test_dd_plotly_label_01():

    dd = DiscreteDistribution(domain=["a", "b", "c", "d", "e", "f"],
                              index=["X1", "X2", "X3"],
                              probs=[
                                  [0.1, 0.2, 0.05, 0.3, 0.5, 0.3],
                                  [0.7, 0.05, 0.05, 0.1, 0, 0.1],
                                  [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    ])

    # plot_filename = os.path.join(EXPECTED_DIR, "test_dd_plotly_label_01.html")
    # dd.plot(filename=plot_filename, auto_open=False)

    # ipdb.set_trace()
