# -*- coding: utf-8 -*-
import pandas as pd
import re
import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


def discretize(data_df, var_specs={},
               prefix_default="",
               suffix_default="",
               fun_default=pd.cut,
               params_default={"bins": 10},
               var_specs_only=False):
    """ Auto discretize data. """
    data_ddf = data_df.copy(deep=True)

    for var in data_df.columns:
        if isinstance(data_df[var].dtypes, pd.CategoricalDtype):
            continue

        disc_specs = var_specs.get(var, {})
        if len(disc_specs) == 0:
            for pat, specs in var_specs.items():
                if re.search(pat, var):
                    disc_specs = specs
                    break

        if len(disc_specs) == 0 and var_specs_only:
            # Do not discretize if current var has no
            # discretization specs specified
            continue

        prefix_cur = disc_specs.get("prefix", prefix_default)
        suffix_cur = disc_specs.get("suffix", suffix_default)

        var_result = prefix_cur + var + suffix_cur

        if data_df[var].dtypes == "float":

            disc_fun = disc_specs.get("fun", fun_default)
            disc_params = disc_specs.get(
                "params", params_default)

            data_disc = disc_fun(data_df.loc[:, var],
                                 **disc_params)

            cats_str = data_disc.cat.categories.astype(str)
            cat_type = \
                pd.api.types.CategoricalDtype(categories=cats_str,
                                              ordered=True)
            data_ddf.loc[:, var_result] = \
                data_disc.astype(str).astype(cat_type)

        else:
            data_ddf.loc[:, var_result] = \
                data_df.loc[:, var].astype("category")

    return data_ddf
