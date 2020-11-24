# -*- coding: utf-8 -*-
import typing
import pydantic
import pandas as pd
import re
import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}

if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


class DiscretizationScheme(pydantic.BaseModel):
    """ Discretization specification for a Series."""

    fun: typing.Callable = \
        pydantic.Field(pd.cut,
                       description="Discretization function")
    params: dict = \
        pydantic.Field({"bins": 10},
                       description="Discretization function parameters")

    prefix: str = \
        pydantic.Field("",
                       description="Prefix of the discretized variable")
    suffix: str = \
        pydantic.Field("",
                       description="Suffix of the discretized variable")

    def discretize(self, series, logging=None):

        # Skip processing if the variable is already categorical
        if isinstance(series.dtypes, pd.CategoricalDtype):
            return series

        if series.dtypes in ["float", "int"]:

            series_d = self.fun(series, **self.params)
            series_d.name = self.prefix + series.name + self.suffix

            cats_str = series_d.cat.categories.astype(str)
            cat_type = \
                pd.api.types.CategoricalDtype(categories=cats_str,
                                              ordered=True)
            series_d = series_d.astype(str).astype(cat_type)

            if not(logging is None):
                logging.debug(
                    ">> Float variable detected: discretization done")

        else:

            series_d = series.astype("category")

            if not(logging is None):
                logging.debug(
                    ">> Object variable detected: discretization done")

        return series_d


class Discretizer(pydantic.BaseModel):

    variables: typing.Dict[str, DiscretizationScheme] = \
        pydantic.Field({},
                       description="Discretization specification"
                       " for each variables."
                       "Variable key can be given as regex")

    process_all_variables: bool = \
        pydantic.Field(False,
                       description="Try to discretize all variables."
                       " Event those without specs using default "
                       "discretization parameters")

    def discretize(self, data_df, logging=None, **kwargs):
        data_ddf = data_df.copy(deep=True)

        for var in data_df.columns:

            if not(logging is None):
                logging.debug(f"> Processing variable {var}")

            disc_specs = None
            for var_pat, var_disc_specs in self.variables.items():
                if re.search(var_pat, var):
                    disc_specs = var_disc_specs
                    break

            if disc_specs is None:
                if not(self.process_all_variables):
                    # Do not discretize if current var has no
                    # discretization specs specified
                    if not(logging is None):
                        logging.debug(
                            ">> Skip: no discrization specifications")
                    continue
                else:
                    disc_specs = DiscretizationScheme()

            series_var = disc_specs.discretize(data_df[var], logging=logging)

            data_ddf.loc[:, series_var.name] = series_var

        return data_ddf


# TODO: Deprecated
# Use Discretizer instead
def discretize(data_df, var_specs={},
               prefix_default="",
               suffix_default="",
               fun_default=pd.cut,
               params_default={"bins": 10},
               var_specs_only=False,
               logging=None):
    """ Auto discretize data. """
    data_ddf = data_df.copy(deep=True)

    for var in data_df.columns:

        if not(logging is None):
            logging.debug(f"> Scanning variable {var}")

        if isinstance(data_df[var].dtypes, pd.CategoricalDtype):
            if not(logging is None):
                logging.debug(">> Skip: already categorical")
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
            if not(logging is None):
                logging.debug(">> Skip: no discrization specifications")
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

            if not(logging is None):
                logging.debug(
                    ">> Float variable detected: discretization done")

        else:

            data_ddf.loc[:, var_result] = \
                data_df.loc[:, var].astype("category")

            if not(logging is None):
                logging.debug(
                    ">> Object variable detected: discretization done")

    return data_ddf
