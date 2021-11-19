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

PandasDataFrame = typing.TypeVar('pd.core.dataframe')


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


class DataImporter(pydantic.BaseModel):

    data_raw_dir: str = pydantic.Field(
        "./data_raw/", description="Raw data directory")

    data_raw_status_filename: str = pydantic.Field(
        "data_raw_status.csv", description="Raw data status filename")

    data_tidy_dir: str = pydantic.Field(
        "./data_tidy/", description="Tidy data directory")

    data_tidy_filename: str = pydantic.Field(
        "data_tidy.csv", description="Tidy data filename")

    reload_data: bool = pydantic.Field(
        False, description="Reset tidy data and reload all raw data")

    data_raw_pattern: str = pydantic.Field(
        "*", description="Data raw filename pattern to be processed")

    data_df: PandasDataFrame = pydantic.Field(
        pd.DataFrame(), description="Data tidy")

    def __init__(self, logger=None, **data: typing.Any):
        super().__init__(**data)

        self.init_env(logger=logger)

    def init_env(self, logger=None):

        if not(logger is None):
            logger.info("ETL process initialization")
        if not(os.path.exists(self.data_tidy_dir)):
            if not(logger is None):
                logger.info(
                    f"> Create tidy data directory: {self.data_tidy_dir}")

            os.mkdir(self.data_tidy_dir)

        if self.reload_data:
            if not(logger is None):
                logger.info(f"> Data reloading requested")
            self.data_tidy_df = pd.DataFrame()
        else:
            mycotoxins_filename = \
                os.path.join(self.data_tidy_dir,
                             self.mycotoxins_filename)

            if os.path.exists(mycotoxins_filename):
                if not(logger is None):
                    logger.info(
                        f"> Load mycostoxins data from file: {mycotoxins_filename}")

                self.data_tidy_df = pd.read_csv(mycotoxins_filename, sep=";",
                                                encoding="utf-8",
                                                parse_dates=["analysis_date"])

                self.postprocess_data_tidy(logger=logger)

        data_raw_status_filename = \
            os.path.join(self.data_tidy_dir,
                         self.data_raw_status_filename)

        if os.path.exists(data_raw_status_filename) and \
           len(self.data_tidy_df) > 0:

            self.data_raw_status_df = pd.read_csv(
                data_raw_status_filename, sep=";")
        else:
            self.data_raw_status_df = \
                pd.DataFrame(columns=["filename",
                                      "path",
                                      "status",
                                      "last_update",
                                      "nb_data"])

        if not(logger is None):
            logger.info(
                f"> Collect raw data filenames from directory: {self.data_raw_dir}")

        data_raw_path_list = \
            [{"filename": os.path.basename(path),
              "path": path}
             for path in sorted(glob.glob(os.path.join(self.data_raw_dir,
                                                       self.data_raw_pattern)))]

        data_raw_new_df = pd.DataFrame(data_raw_path_list)

        self.data_raw_status_df = \
            pd.concat([self.data_raw_status_df,
                       data_raw_new_df],
                      sort=False,
                      ignore_index=True,
                      axis=0).drop_duplicates(subset=["filename"],
                                              keep="first")

        self.data_raw_status_df["status"] = \
            self.data_raw_status_df["status"].fillna("New")

        self.data_raw_status_df["nb_data"] = \
            self.data_raw_status_df["nb_data"].fillna(0).astype(int)

        self.data_raw_status_df["last_update"] = \
            self.data_raw_status_df["last_update"]\
                .fillna(datetime.datetime.now())\
                .astype("datetime64")

    def post_process_data_tidy(self, logger=None):
        """Method called just after loading tidy data.

        Overload this method to put postprocessing code here
        """
        # Convert departments in string
        self.data_tidy_df['department'] = \
            self.data_tidy_df['department'].astype(str)

        # TODO: Is this relevant ?
        self.data_tidy_df['analysis_date'] = \
            pd.to_datetime(
                self.data_tidy_df['analysis_date'], 'coerce')

        # pass

    def run(self, logger=None, progress_mode=True):

        # self.init_env(logger=logger)

        # Get raw data filename to be integrated
        data_raw_status_to_process_df = \
            self.data_raw_status_df[self.data_raw_status_df["status"] != "OK"]

        for data_index in tqdm.tqdm(
                data_raw_status_to_process_df.index,
                desc="Processing mycotoxins raw data files",
                unit="File",
                disable=not(progress_mode)):

            data_raw_pathname = data_raw_status_to_process_df.loc[data_index, "path"]
            data_raw_filename = data_raw_status_to_process_df.loc[data_index, "filename"]

            if not(logger is None):
                logger.info(
                    f"> Try to process data file: {data_raw_pathname}")

            try:

                data_raw_cur_df = \
                    self.load_data_raw(data_raw_pathname,
                                       logger=logger)

                data_tidy_new_df = self.transform_data_raw(data_raw_cur_df,
                                                           logger=logger)
                # Add columns to keep track of origin filename
                data_tidy_new_df["from_file"] = data_raw_filename

                self.update_data_tidy(data_tidy_new_df,
                                      logger=logger)

                self.save_data_tidy(data_tidy_new_df,
                                    logger=logger)

                status = "OK"

            except Exception as ex:
                logger.error(f"> Processing error: {str(ex)}")
                data_tidy_new_df = []  # To register nb new data
                status = "Error: " + str(ex)

            self.update_data_raw_status(data_index=data_index,
                                        status=status,
                                        nb_data=len(data_tidy_new_df),
                                        logger=logger)
            # ipdb.set_trace()

    def load_data_raw(self, filename, logger=None):
        return pd.DataFrame()

    def transform_data_raw(self, data_raw_df, logger=None):
        return data_raw_df

    def update_data_tidy(self, data_tidy_new_df, logger=None):

        if len(data_tidy_new_df) == 0:
            logger.warning("> No tidy data to update")
            return

        self.data_tidy_df = pd.concat(
            [self.data_tidy_df, data_tidy_new_df],
            ignore_index=True,
            axis=0)

        logger.info(
            f"> # new data added: {len(data_tidy_new_df)}")

        # TODO: MOVE THIS IN transform_method
        # Replace empty field with "no info" string
        self.data_tidy_df.fillna('no info', inplace=True)

    def save_data_tidy(self, logger=None):

        data_tidy_filename = os.path.join(self.data_tidy_dir,
                                          self.data_tidy_filename)
        self.data_tidy_df.to_csv(data_tidy_filename,
                                 sep=";",
                                 encoding="utf-8",
                                 index=False)
        logger.info(
            f"> Tidy data saved in file: {data_tidy_filename}")

    def update_data_raw_status(self,
                               data_index,
                               status,
                               nb_data,
                               logger=None):

        self.data_raw_status_df.loc[data_index, "status"] = status
        self.data_raw_status_df.loc[data_index, "nb_data"] = nb_data
        self.data_raw_status_df.loc[data_index,
                                    "last_update"] = datetime.datetime.now()

        data_raw_status_filename = os.path.join(self.data_tidy_dir,
                                                self.data_raw_status_filename)

        self.data_raw_status_df.to_csv(data_raw_status_filename,
                                       sep=";",
                                       index=False)
