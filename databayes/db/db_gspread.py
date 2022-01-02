import pydantic
import typing
import logging
import pandas as pd
import numpy as np

from .db_base import DBBase

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401

if 'gspread' in installed_pkg:
    import gspread  # noqa: F401
else:
    raise ModuleNotFoundError("Please install gspread module")


class GSpreadConfig(pydantic.BaseModel):

    ss_id: str = pydantic.Field(None,
                                description="Spread sheet name, id or url.")

    credentials_filename: str = pydantic.Field(
        None, description="Google Cloud JSON credential filename")


class DBGSpread(DBBase):

    worksheet_names_: list = pydantic.Field([],
                                            description="List of worksheet names")
    config: GSpreadConfig = pydantic.Field(GSpreadConfig(),
                                           description="The data backend configuration")

    def connect(self, config: GSpreadConfig = GSpreadConfig(), **params):

        if config.credentials_filename:
            self.config.credentials_filename = config.credentials_filename

        if config.ss_id:
            self.config.ss_id = config.ss_id

        if self.config.credentials_filename:
            gc = gspread.service_account(
                filename=self.config.credentials_filename)
        else:
            gc = gspread.service_account()

        self.db = gc.open_by_url(self.config.ss_id)

        self.update_worksheet_names_()

    def update_worksheet_names_(self):
        self.worksheet_names_ = [wk.title for wk in self.db.worksheets()]

    def get_worksheet(self, endpoint=0):
        return self.db.get_worksheet(endpoint) if isinstance(endpoint, int) \
            else self.db.worksheet(endpoint)

    def count(self, endpoint=0, **params):

        worksheet = self.get_worksheet(endpoint)

        return len(worksheet.get_all_values())

    def get(self, endpoint=0,
            limit=0,
            **params):

        worksheet = self.get_worksheet(endpoint)

        return worksheet.get_all_records()

    def put(self, endpoint=0,
            data=[],
            header=False,
            logging=logging,
            add_worksheet=True,
            add_worksheet_rows=1000,
            add_worksheet_cols=26,
            clear=False,
            index=False,
            **params):

        if isinstance(data, list) and not(isinstance(data[0], list)):
            data = [data]

        if isinstance(endpoint, str):
            if not(endpoint in self.worksheet_names_) and add_worksheet:
                self.db.add_worksheet(endpoint,
                                      rows=add_worksheet_rows,
                                      cols=add_worksheet_cols)
                self.update_worksheet_names_()

        worksheet = self.get_worksheet(endpoint)

        if clear:
            worksheet.clear()

        # Manage header
        if isinstance(data, pd.DataFrame):
            worksheet.update([data.columns.values.tolist()])
        else:
            if header:
                worksheet.update(data[0:1])

        try:
            if isinstance(data, pd.DataFrame):
                data_export = data.fillna("")
                if index:
                    data_export = data_export.reset_index()

                # Clean not json serialisable
                var_datetime = \
                    data_export.select_dtypes(include=[np.datetime64])\
                               .columns
                data_export[var_datetime] = \
                    data_export[var_datetime].astype(str)

                worksheet.append_rows(data_export.values.tolist(), **params)
            else:
                if header:
                    worksheet.append_rows(data[1:], **params)
                else:
                    worksheet.append_rows(data, **params)

        except Exception as err:
            if logging:
                logging.error(
                    f"Problem occurred inserting data in {worksheet.title} : {err}")

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
