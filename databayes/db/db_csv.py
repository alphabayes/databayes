import pydantic
import typing
import logging
import pandas as pd
import os
import pathlib
import shutil
import glob

from .db_dataframe import DBDataFrame, DBDataFrameConfig

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401

# TODO: TRY TO REFACTOR DBCSV and DBXLSX with a parent class that shares common behaviour


class CSVConfig(DBDataFrameConfig):

    data_pattern: str = pydantic.Field("*.csv",
                                       description="Data pattern to get data into directory.")


class DBCSV(DBDataFrame):

    config: CSVConfig = pydantic.Field(CSVConfig(),
                                       description="The data backend configuration")

    @classmethod
    def from_dict(cls, config: CSVConfig = CSVConfig()):

        cls(**config)

    def connect(self, config: CSVConfig = CSVConfig(), reset=False, **params):

        if reset and os.path.exists(self.config.directory):
            shutil.rmtree(self.config.directory)

        self.update()

    def set_name(self, name, **params):
        self.config.directory = name.replace(".", "/")
        self.connect(**params)

    def update(self):

        self.db = dict()

        if os.path.exists(self.config.directory):

            data_filenames_list = \
                glob.glob(os.path.join(self.config.directory,
                                       self.config.data_pattern))

            # Reconstruct index
            for data_filename in data_filenames_list:

                data_name = os.path.splitext(
                    os.path.basename(data_filename))[0]

                data_df = \
                    pd.read_csv(data_filename, **self.config.read_params)

                index_col_tagged = [col for col in data_df.columns
                                    if col.startswith(self.config.index_prefix)]

                len_index_prefix = len(self.config.index_prefix)

                if len(index_col_tagged) > 0:
                    index_col_rename = {col: col[len_index_prefix:]
                                        for col in index_col_tagged}
                    data_df.rename(columns=index_col_rename, inplace=True)
                    data_df.set_index(list(index_col_rename.values()),
                                      inplace=True)

                self.db[data_name] = data_df

    def commit(self, sheet_list=None):

        pathlib.Path(self.config.directory).mkdir(parents=True, exist_ok=True)

        for data_name, data_df in self.db.items():

            data_filename = os.path.join(self.config.directory,
                                         data_name + ".csv")

            if data_df.index.name:
                # Save indexes columns
                idx_name = data_df.index.name
                data_bis_df = data_df.reset_index()
                data_bis_df.rename(columns={idx_name: self.config.index_prefix + idx_name},
                                   inplace=True)
                data_bis_df.to_csv(data_filename,
                                   index=False,
                                   **self.config.write_params)

            else:
                data_df.to_csv(data_filename,
                               index=False,
                               **self.config.write_params)

    def count(self, endpoint=0, **params):

        return len(self.db[str(endpoint)])

    def get(self, endpoint=0,
            limit=None,
            **params):

        return self.db.get(str(endpoint), pd.DataFrame())

    def put(self, endpoint=0,
            data=[],
            header=False,
            clear=False,
            update=True,
            commit=False,
            logging=logging,
            **params):

        # xlsx write does not support int sheet name
        endpoint = str(endpoint)

        if not(endpoint in self.db) or clear:
            self.db[endpoint] = pd.DataFrame()

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

        if commit:
            self.commit()
