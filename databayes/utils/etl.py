# -*- coding: utf-8 -*-
import typing
import pydantic
import pandas as pd
import re
import pkg_resources
import os
from intervals import FloatInterval

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}

if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


# ETL functions
def pdInterval_from_string(s):
    it = FloatInterval.from_string(s)

    if it.lower_inc and not(it.upper_inc):
        closed = 'left'
    elif not(it.lower_inc) and it.upper_inc:
        closed = 'right'
    elif not(it.lower_inc) and not(it.upper_inc):
        closed = 'both'
    else:
        closed = 'neither'

    pit = pd.Interval(left=it.lower,
                      right=it.upper,
                      closed=closed)

    return pit


def pdInterval_series_from_string(strlist):

    it_list = [pdInterval_from_string(s)
               for s in strlist]
    it_s = pd.Series(it_list)
    return it_s

# ETL Classes
# ===========


class DiscretizationScheme(pydantic.BaseModel):
    """ Discretization specification for a Series."""

    fun: typing.Callable = pydantic.Field(pd.cut,
                                          description="Discretization function")
    params: dict = pydantic.Field({"bins": 10},
                                  description="Discretization function parameters")

    prefix: str = pydantic.Field("",
                                 description="Prefix of the discretized variable")
    suffix: str = pydantic.Field("",
                                 description="Suffix of the discretized variable")

    force_str: bool = pydantic.Field(False,
                                     description="Force string conversion for categories")

    def discretize(self, series, logging=None):

        # Skip processing if the variable is already categorical
        if isinstance(series.dtypes, pd.CategoricalDtype):
            return series

        # Add 'int64' in the list for Windows compatibility
        # in Pandas 1.1.2
        if series.dtypes in ["float", "int", "int64"]:

            series_d = self.fun(series, **self.params)
            series_d.name = self.prefix + series.name + self.suffix

            if self.force_str:
                cats_str = series_d.cat.categories.astype(str)
                cat_type = pd.api.types.CategoricalDtype(categories=cats_str,
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

    variables: typing.Dict[str, DiscretizationScheme] = pydantic.Field({},
                                                                       description="Discretization specification"
                                                                       " for each variables."
                                                                       "Variable key can be given as regex")

    process_all_variables: bool = pydantic.Field(False,
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
            cat_type = pd.api.types.CategoricalDtype(categories=cats_str,
                                                     ordered=True)
            data_ddf.loc[:, var_result] = data_disc.astype(
                str).astype(cat_type)

            if not(logging is None):
                logging.debug(
                    ">> Float variable detected: discretization done")

        else:

            data_ddf.loc[:, var_result] = data_df.loc[:,
                                                      var].astype("category")

            if not(logging is None):
                logging.debug(
                    ">> Object variable detected: discretization done")

    return data_ddf


def split_csv_file(filename,
                   event_col_name,
                   separator,
                   date_min=None,
                   date_max=None,
                   time_range=None,
                   nb_parts=None):
    """" This function split a csv file  according to two possible rules :  If  "time range" is given in  input, the
    split will be done according to the specified time col  between the input date_min and date_max. If nb_parts is given
    the split will be done  by cutting the col_event_name into equal parts

    The csv ouputs are saved in a subfolder named : filename_split . If time_range is found, then it used for plit otherwise, nb of row should be
    defined in input.
        inputs:
            - filename: name of the file to split
            - time_range: Pandas frequency chosen for the chosen cut
            - nb_parts: Nunber of parts of the output split
            - date_min: lowest date of the data set to split
            - date_max: Highest date of the data set to split
            - event_col_name: Name of the reference column for splitting
            - separator: separator used by the csv file
        output:
            Set of csv files stored in : filename_split/filename_split_part_x
        Example:
            generic_lib.general_tools.split_csv_file(filename='data_raw/SIGNALEMENTS/SIGNALEMENT.csv',
                                                time_range='1Y',
                                                date_min='01/01/2012',
                                                date_max='01/01/2019',
                                                event_col_name='SIG_DATE',
                                                separator=";")

            generic_lib.general_tools.split_csv_file(filename=path,
                                                    nb_part=10,
                                                    event_col_name='OT_ID',
                                                    separator=";")"""

    # Create outputt directory if it does not exist
    output_dir_name = filename[:-4] + '_split'
    exact_name = filename[filename.rfind('/') + 1:len(filename) - 4]

    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    # Read the csv file
    input_df = pd.read_csv(filename, sep=separator)

    # Create the list of interval for the cut

    if not (time_range is None):

        # Read input dates

        date_min = pd.Timestamp(date_min)
        date_max = pd.Timestamp(date_max)
        print('Chosen dates for the intervall to split : ', date_min, date_max)

        # Datetype conversion
        input_df[event_col_name] = pd.to_datetime(input_df[event_col_name])

        #  Create time intervals for split

        range_interval = list(pd.date_range(
            start=date_min, end=date_max, freq=time_range))
        intervals_list = [date_min] + range_interval + [date_max]

        for i in range(0, len(intervals_list) - 1):
            low_bound = intervals_list[i]
            max_bound = intervals_list[i + 1]
            mask = (input_df[event_col_name] > low_bound) & (
                input_df[event_col_name] <= max_bound)

            df_temp = input_df.loc[mask]

            print('df number : ', i, 'between date ',
                  low_bound, ' and date', max_bound)

            filename_temp = output_dir_name + '/' + \
                exact_name + '_split_part_' + str(i) + '.csv'
            df_temp.to_csv(filename_temp, sep=separator, index=False)

    if not (nb_parts is None):
        #  Cuts the set of event_id

        intervals_list = pd.cut(
            list(input_df[event_col_name].unique()), bins=nb_parts).categories
        print('liste des intervalles', intervals_list)

        for i in range(0, len(intervals_list)):
            interval = intervals_list[i]
            mask = input_df[event_col_name].apply(lambda x: x in interval)
            df_temp = input_df.loc[mask]
            print('df number : ', i, 'sur lintervalle', interval)
            filename_temp = output_dir_name + '/' + \
                exact_name + '_split_part_' + str(i) + '.csv'
            df_temp.to_csv(filename_temp, sep=separator, index=False)
