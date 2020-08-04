# -*- coding: utf-8 -*-
import pandas as pd
import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


def discretize(data_df, discretization_fun={},
               discretization_fun_default=pd.cut,
               discretization_params_default={"bins": 10}):
    """ Auto discretize data. """
    data_ddf = data_df.copy(deep=True)

    for var in data_df.columns:
        if isinstance(data_df[var].dtypes, pd.CategoricalDtype):
            continue

        if data_df[var].dtypes == "float":

            disc_specs = discretization_fun.get(var, {})
            disc_fun = disc_specs.get("fun", discretization_fun_default)
            disc_params = disc_specs.get(
                "params", discretization_params_default)

            data_disc = disc_fun(data_df.loc[:, var],
                                 **disc_params)

            cats_str = data_disc.cat.categories.astype(str)
            cat_type = \
                pd.api.types.CategoricalDtype(categories=cats_str,
                                              ordered=True)
            data_ddf.loc[:, var] = data_disc.astype(str).astype(cat_type)

        else:
            data_ddf.loc[:, var] = data_df.loc[:, var].astype("category")

    return data_ddf
