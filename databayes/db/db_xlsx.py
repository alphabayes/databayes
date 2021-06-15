import pydantic
import typing
import logging
import pandas as pd
import os

from .db_base import DBBase

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


class XLSXConfig(pydantic.BaseModel):

    filename: str = pydantic.Field("data.xlsx",
                                   description="Data filename.")

    index_prefix: str = pydantic.Field("__",
                                       description="Prefix to identify data indexes")


class DBXLSX(DBBase):

    config: XLSXConfig = pydantic.Field(XLSXConfig(),
                                        description="The data backend configuration")

    def connect(self, config: XLSXConfig = XLSXConfig(), reset=False, **params):

        if reset and os.path.exists(self.config.filename):
            os.remove(self.config.filename)

        self.update()

    def set_name(self, name, **params):
        self.config.filename = name + ".xlsx"
        self.connect(**params)

    def update(self):

        if os.path.exists(self.config.filename):
            self.db = pd.read_excel(self.config.filename,
                                    engine='openpyxl',
                                    sheet_name=None)

            # Reconstruct index
            for sheet, data_df in self.db.items():
                index_col_tagged = [col for col in data_df.columns
                                    if col.startswith(self.config.index_prefix)]

                len_index_prefix = len(self.config.index_prefix)

                if len(index_col_tagged) > 0:
                    index_col_rename = {col: col[len_index_prefix:]
                                        for col in index_col_tagged}
                    data_df.rename(columns=index_col_rename, inplace=True)
                    data_df.set_index(
                        list(index_col_rename.values()), inplace=True)

        else:
            self.db = dict()

    def commit(self, sheet_list=None):

        writer = pd.ExcelWriter(self.config.filename,
                                engine='xlsxwriter')

        if not(sheet_list):
            sheet_list = self.db.keys()

        for sheet, data_df in self.db.items():

            if sheet in sheet_list:

                if data_df.index.name:
                    # Save indexes columns
                    idx_name = data_df.index.name
                    data_bis_df = data_df.reset_index()
                    data_bis_df.rename(columns={idx_name: self.config.index_prefix + idx_name},
                                       inplace=True)
                    data_bis_df.to_excel(writer,
                                         sheet_name=sheet,
                                         index=False)

                else:
                    data_df.to_excel(writer,
                                     sheet_name=sheet,
                                     index=False)

        writer.save()

    def count(self, endpoint=0, **params):

        return len(self.db[str(endpoint)])

    def get(self, endpoint=0,
            limit=None,
            **params):

        return self.db[str(endpoint)]

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
        # ipdb.set_trace()

    # def delete(self, endpoint,
    #            filter={},
    #            logger=None,
    #            **params):

    #     res_dict = {"data_name": endpoint,
    #                 "ops_type": "delete",
    #                 "ops": dict([
    #                     ("nb_deletions", 0)])}

    #     db_coll, index = self.prepare_and_get_coll(endpoint)

    #     ops_res_cur = db_coll.delete_many(filter=filter)

    #     res_dict["ops"]["nb_deletions"] = ops_res_cur.deleted_count

    #     mongoDataBackend.log_db_ops(res_dict,
    #                                 logger=logger)

    #     return res_dict
