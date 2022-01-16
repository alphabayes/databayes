import gspread
import gspread_pandas as gspd
import pydantic
import typing
import logging
import pandas as pd
import glob
import os
import pathlib
import numpy as np


from .db_base import DBBase

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


PandasDataFrame = typing.TypeVar('pandas.core.frame.DataFrame')


class DataFrameConfig(pydantic.BaseModel):

    # filename: str = pydantic.Field("data.xlsx",
    #                                description="Data filename.")
    path: str = pydantic.Field(
        None, description="Data path")

    index_prefix: str = pydantic.Field("idx__",
                                       description="Prefix to identify data indexes")

    load_params: dict = \
        pydantic.Field({},
                       description="Params to be passed to data backend loading function")

    dump_params: dict = \
        pydantic.Field({},
                       description="Params to be passed to data backend writing function")


class CSVConfig(DataFrameConfig):

    path: str = pydantic.Field(".",
                               description="Base directory where the csv directory will be created")

    file_pattern: str = pydantic.Field("*.csv",
                                       description="Data pattern to get data into directory.")


class ExcelConfig(DataFrameConfig):

    path: str = pydantic.Field(".",
                               description="Base directory where the csv directory will be created")


class GSpreadConfig(DataFrameConfig):

    path: str = pydantic.Field(...,
                               description="Spreadsheet ID")
    credentials_filename: str = \
        pydantic.Field(None,
                       description="Google service account filename")


# Abstract class for DataFrame based storage system
class DBDataFrame(DBBase):

    # DataFrame

    db: typing.Dict[str, PandasDataFrame] = pydantic.Field(dict(),
                                                           description="Data as a dictionary of dataframe")

    is_modified: typing.Dict[str, bool] = \
        pydantic.Field(dict(),
                       description="Indicates if a dataframe has been modified")

    config: DataFrameConfig = pydantic.Field(DataFrameConfig(),
                                             description="The data backend configuration")

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        self.connect()

    def connect(self):
        """ Backend connection."""
        pass

    def get_data_to_dump(self, data_list=None):

        if data_list is None:
            data_list = self.db.keys()

        data_dump_dfd = {}

        for data_name, data_df in self.db.items():

            if data_name in data_list:

                if data_df.index.name:
                    # Save indexes columns
                    idx_name = data_df.index.name
                    data_dump_dfd[data_name] = data_df.reset_index()
                    data_dump_dfd[data_name].rename(
                        columns={idx_name: self.config.index_prefix + idx_name},
                        inplace=True)

                else:
                    data_dump_dfd[data_name] = data_df.copy()

        return data_dump_dfd

    @classmethod
    def load_csv(cls, **specs):
        """ Create a DBCSV backend."""

        db = DBCSV(**specs)
        db.load()

        return db

    def to_csv(self, data_list=None, config: CSVConfig = CSVConfig()):
        """Save data as csv directory."""

        db = DBCSV(config=config,
                   **self.dict(exclude={"config"}))
        db.dump(data_list=data_list)
        return db

    @classmethod
    def load_excel(cls, **specs):
        """ Create a DBExcel backend."""

        db = DBExcel(**specs)
        db.load()

        return db

    def to_excel(self, data_list=None, config: ExcelConfig = ExcelConfig()):
        """Save data as excel file."""

        db = DBExcel(config=config,
                     **self.dict(exclude={"config"}))
        db.dump(data_list=data_list)
        return db

    @classmethod
    def load_gspread(cls, **specs):
        """ Create a DBGSpread backend."""

        db = DBGSpread(**specs)
        db.load()

        return db

    def to_gspread(self, config: GSpreadConfig, data_list=None):
        """Save data as gspread file."""

        db = DBGSpread(config=config,
                       **self.dict(exclude={"config"}))
        db.dump(data_list=data_list)
        return db

    def post_load(self):

        # if os.path.exists(self.config.filename):
        #     self.db = pd.read_excel(self.config.filename,
        #                             engine='openpyxl',
        #                             sheet_name=None)

        # Reconstruct index
        for data_name, data_df in self.db.items():

            index_col_tagged = [col for col in data_df.columns
                                if col.startswith(self.config.index_prefix)]

            len_index_prefix = len(self.config.index_prefix)

            if len(index_col_tagged) > 0:
                index_col_rename = {col: col[len_index_prefix:]
                                    for col in index_col_tagged}
                data_df.rename(columns=index_col_rename, inplace=True)
                data_df.set_index(
                    list(index_col_rename.values()), inplace=True)

        self.is_modified = {data_name: False for data_name in self.db}

        # else:
        #     self.db = dict()

    def count(self, endpoint=0, **params):

        return len(self.db[str(endpoint)])

    def get(self, endpoint=0,
            limit=None,
            **params):

        return self.db[str(endpoint)]

    def put(self, endpoint=0,
            data=[],
            header=None,
            clear=False,
            update=True,
            commit=False,
            logging=logging,
            **params):

        # xlsx write does not support int data_name name
        endpoint = str(endpoint)

        if not(endpoint in self.db) or clear:
            self.db[endpoint] = pd.DataFrame()

        if (header is None) and isinstance(data, pd.DataFrame):
            header = True

        if isinstance(data, list) and not(isinstance(data[0], list)):
            data = [data]
        elif isinstance(data, dict):
            data = pd.DataFrame([data])

        if isinstance(data, list):
            if header:
                data = pd.DataFrame(data[1:], columns=data[0])
            else:
                data = pd.DataFrame(data)

        if update and self.db[endpoint].index.name and data.index.name:
            idx_inter = data.index.intersection(self.db[endpoint].index)
            idx_diff = data.index.difference(self.db[endpoint].index)

            self.db[endpoint].loc[idx_inter] = data.loc[idx_inter]
            self.db[endpoint] = self.db[endpoint].append(data.loc[idx_diff])

        else:
            self.db[endpoint] = self.db[endpoint].append(data)

        self.is_modified[endpoint] = True

        if commit:
            self.dump(**params)
        # ipdb.set_trace()


class DBCSV(DBDataFrame):

    config: CSVConfig = pydantic.Field(CSVConfig(),
                                       description="The data backend configuration")

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        if not(self.name is None):
            self.config.path = os.path.join(self.config.path, self.name)
        else:
            self.name = self.config.path.split(os.sep)[-1]

    def load(self):
        """ Load data."""

        filenames = glob.glob(os.path.join(self.config.path,
                                           self.config.file_pattern))

        for filename in filenames:
            filename_base = os.path.basename(filename).split(".")[0]
            self.db[filename_base] = pd.read_csv(filename,
                                                 **self.config.load_params)

        self.post_load()

    def dump(self, data_list=None, config: CSVConfig = None, **params):
        """Save data as csv directory."""

        if not(config is None):
            self.config = config

        data_dump_dfd = self.get_data_to_dump(data_list=data_list)

        if not(self.name is None):
            path = os.path.join(self.config.path, self.name)
        else:
            path = self.config.path

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        self.config.dump_params.update({"index": False})

        for data_name, data_df in data_dump_dfd.items():

            filename = os.path.join(path,
                                    f"{data_name}.csv")

            data_df.to_csv(filename,
                           **self.config.dump_params)

            self.is_modified[data_name] = False


class DBExcel(DBDataFrame):

    config: ExcelConfig = pydantic.Field(ExcelConfig(),
                                         description="The data backend configuration")

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        if self.name is None:
            self.name = self.config.path.split(os.sep)[-1].split(".")[0]

    def load(self):

        self.config.load_params.update(engine='openpyxl',
                                       sheet_name=None)

        self.db = pd.read_excel(self.config.path,
                                **self.config.load_params)

        self.post_load()

    def dump(self, data_list=None, config: ExcelConfig = None, **params):

        if not(config is None):
            self.config = config

        data_dump_dfd = self.get_data_to_dump(data_list=data_list)

        basepath = os.path.sep.join(self.config.path.split(os.path.sep)[:-1])
        pathlib.Path(basepath).mkdir(parents=True, exist_ok=True)

        self.config.dump_params.update({"index": False})

        writer = pd.ExcelWriter(self.config.path,
                                engine='xlsxwriter')

        for data_name, data_df in data_dump_dfd.items():

            data_df.to_excel(writer,
                             sheet_name=data_name,
                             **self.config.dump_params)

            self.is_modified[data_name] = False

        writer.save()


class DBGSpread(DBDataFrame):

    config: GSpreadConfig = pydantic.Field(...,
                                           description="The data backend configuration")

    backend: typing.Any = pydantic.Field(None,
                                         description="Spreadsheet data handler")

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        if self.name is None:
            self.name = self.backend.spread.title

    def connect(self):

        if self.config.credentials_filename:
            gc = gspread.service_account(
                filename=self.config.credentials_filename)
        else:
            gc = gspread.service_account()

        self.backend = gspd.Spread(self.config.path, creds=gc.auth)

    def post_load(self):
        super().post_load()

        parse_dates = self.config.load_params.get("parse_dates", [])
        for data_df in self.db.values():
            for var in data_df.columns:
                if len(var) == 0:
                    continue
                if data_df.dtypes[var] != "object":
                    continue

                if var in parse_dates:
                    data_df.loc[:, var] = pd.to_datetime(data_df.loc[:, var])
                else:
                    try:
                        data_df.loc[:, var] = \
                            pd.to_numeric(
                                data_df.loc[:, var].str.replace(",", "."))
                    except (ValueError, TypeError):
                        pass

    def load(self):

        sheet_to_df_params = ["index", "header_rows", "start_row"]

        self.config.load_params.update(index=None)

        load_params = {k: v for k, v in self.config.load_params.items()
                       if k in sheet_to_df_params}

        for ss in self.backend.sheets:
            self.db[ss.title] = self.backend.sheet_to_df(
                sheet=ss.title,
                **load_params)

        self.post_load()

    def dump(self, data_list=None,
             if_modified=False,
             config: GSpreadConfig = None,
             **params):

        if not(config is None):
            self.config = config
            self.connect()

        data_dump_dfd = self.get_data_to_dump(data_list=data_list)

        self.config.dump_params.update(
            index=False)

        ss_list = self.backend.sheets
        ss_name_list = [ss.title for ss in ss_list]

        for data_name, data_df in data_dump_dfd.items():

            dump_condition = \
                not(if_modified) or \
                not(data_name in ss_name_list) or \
                (if_modified and self.is_modified[data_name])

            if dump_condition and len(data_df) > 0:

                # Convert '.' decimal to ',' for GSheet
                num_var_list = data_df.select_dtypes(include=["float"]).columns
                for num_var in num_var_list:
                    data_df[num_var] = \
                        data_df[num_var].astype(str)\
                                        .str.replace(".", ",",
                                                     regex=False)\
                                        .replace("nan", "")

                # Write data
                self.backend.df_to_sheet(
                    data_df,
                    sheet=data_name,
                    **self.config.dump_params)

                # TODO: Add an option for this
                # Clean remaining data on the sheet
                data_nb_rows = len(data_df) + 1
                data_nb_cols = len(data_df.columns)

                # Blank potentially other data in the sheet
                sheet_nb_rows, sheet_nb_cols = \
                    self.backend.get_sheet_dims(data_name)

                bdata_nb_rows = sheet_nb_rows - data_nb_rows
                bdata_nb_cols = sheet_nb_cols - data_nb_cols

                if bdata_nb_rows > 0:

                    rows_bdf = pd.DataFrame(np.chararray(
                        (sheet_nb_rows, sheet_nb_cols), unicode=True))

                    self.backend.df_to_sheet(
                        rows_bdf,
                        start=(data_nb_rows + 1, 1),
                        sheet=data_name,
                        index=False,
                        headers=False)

                if bdata_nb_cols > 0:
                    cols_bdf = pd.DataFrame(np.chararray(
                        (data_nb_rows, bdata_nb_cols), unicode=True))

                    self.backend.df_to_sheet(
                        cols_bdf,
                        start=(1, data_nb_cols + 1),
                        sheet=data_name,
                        index=False,
                        headers=False)

                self.is_modified[data_name] = False
