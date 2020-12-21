# -*- coding: utf-8 -*-
import pandas as pd
import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


def ddomain_equals(series, ddomain):
    """ Compare series domain with a discrete domain ddomain."""
    test = False
    if series.dtype.name == "category":
        test = list(series.cat.categories) == ddomain

    return test
