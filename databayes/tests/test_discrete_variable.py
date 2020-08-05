# -*- coding: utf-8 -*-

# ==========================================================
# Classe de test pour les Distributions Discrètes
# josquin.foulliaron@edgemind.net
# ==========================================================
import os
import logging

from databayes.modelling.DiscreteVariable import DiscreteVariable
import pytest

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb

logger = logging.getLogger()


@pytest.mark.parametrize("inputs, expected", [
    ([], "label"),
    (["a", "b"], "label"),
    ([1], "numeric"),
    ([1.0, 3], "numeric"),
    (['(0.0, 1.0]', '(1.0, 2.0]', '(2.0, 3.0]', '(3.0, inf]'], "interval"),
])
def test_detect_domain_type(inputs, expected):
    dom = DiscreteVariable.detect_domain_type(inputs)
    assert dom == expected


@pytest.mark.parametrize("inputs, expected", [
    ([], "label"),
    (["a", "b"], "label"),
    ([1], "numeric"),
    ([1.0, 3], "numeric"),
    (['(0.0, 1.0]', '(1.0, 2.0]', '(2.0, 3.0]', '(3.0, inf]'], "interval"),
])
def test_dv_basic_0001(inputs, expected):

    dv = DiscreteVariable(domain=inputs)
    assert dv.domain_type == expected
