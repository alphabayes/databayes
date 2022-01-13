import databayes.db as db
import yaml
import json
import numpy as np
import pandas as pd
import logging
import pytest
import os
import pkg_resources
import copy

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


logger = logging.getLogger()

# DATA_DIR = "data"

# Util function


# def ppcpt(c): print(gum_pp.cpt2txt(c))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
EXPECTED_PATH = os.path.join(os.path.dirname(__file__), "expected")


@pytest.fixture(scope="module")
def data_btc_usdc_1m_100():

    data_filename = os.path.join(DATA_PATH, "data_btc_usdc_1m_100.csv")
    data_ohlcv_df = pd.read_csv(data_filename, sep=";",
                                index_col="timestamp",
                                parse_dates=["datetime"])
    return data_ohlcv_df


@pytest.fixture(scope="module")
def data_edf_1d_1mo():

    data_filename = os.path.join(DATA_PATH, "data_edf_1d_1mo.csv")
    data_ohlcv_df = pd.read_csv(data_filename, sep=",",
                                index_col="timestamp",
                                parse_dates=["timestamp"])
    return data_ohlcv_df


def test_db_dataframe_001():

    path = os.path.join(DATA_PATH, "db_dataframe_001")

    db_df_config = dict(path=path,
                        load_params=dict(
                            parse_dates=["timestamp"]
                        ))

    db_df = db.DBDataFrame.load_csv(config=db_df_config)

    path_res = os.path.join(EXPECTED_PATH, "db_dataframe_001")
    db_df_config.update(path=path_res)
    db_df_bis = db_df.to_csv(config=db_df_config)

    assert db_df_bis.name == db_df.name

    for data_name, data_df in db_df.db.items():
        pd.testing.assert_frame_equal(
            data_df,
            db_df_bis.db[data_name])

    db_df_ter = db.DBDataFrame.load_csv(name=db_df_bis.name,
                                        config=db_df_config)

    assert db_df.name == db_df_ter.name

    for data_name, data_df in db_df.db.items():
        pd.testing.assert_frame_equal(
            data_df,
            db_df_ter.db[data_name])


def test_db_dataframe_002():

    path = os.path.join(DATA_PATH, "db_dataframe_002.xlsx")

    db_df_config = dict(path=path)

    db_df = db.DBDataFrame.load_excel(config=db_df_config)

    path_res = \
        os.path.join(EXPECTED_PATH, path.split(os.path.sep)[-1])
    db_df_config.update(path=path_res)

    db_df_bis = db_df.to_excel(config=db_df_config)

    assert db_df_bis.name == db_df.name

    for data_name, data_df in db_df.db.items():
        pd.testing.assert_frame_equal(
            data_df,
            db_df_bis.db[data_name])

    db_df_config = dict(path=path,
                        load_params=dict(
                            parse_dates=["timestamp"]
                        ))
    db_df_ter = \
        db.DBDataFrame.load_csv(
            config=dict(path=os.path.join(DATA_PATH, "db_dataframe_001"),
                        load_params=dict(
                            parse_dates=["timestamp"]
            )))

    for data_name, data_df in db_df.db.items():
        pd.testing.assert_frame_equal(
            data_df,
            db_df_ter.db[data_name])


def test_db_dataframe_003():

    path = os.path.join(DATA_PATH, "db_dataframe_002.xlsx")
    db_df_excel_config = dict(path=path)
    db_df_excel = db.DBDataFrame.load_excel(config=db_df_excel_config)

    db_df_config = dict(
        path="https://docs.google.com/spreadsheets/d/1WWx0KhunGOcnSHNgfHbSUDpr1jGK_YRtc0dJSUXTskc",
        credentials_filename="mosaic-337020-bb33a992f590.json")

    db_df = db_df_excel.to_gspread(config=db_df_config)
    assert db_df.name == "db_dataframe_002"

    for data_name, data_df in db_df.db.items():
        pd.testing.assert_frame_equal(
            data_df,
            db_df_excel.db[data_name])

    db_df_config.update(
        load_params=dict(
            parse_dates=["timestamp"]
        ))
    db_df_bis = db.DBDataFrame.load_gspread(
        config=db_df_config
    )

    assert db_df_bis.name == "test_db_dataframe_003"

    for data_name, data_df in db_df.db.items():
        pd.testing.assert_frame_equal(
            data_df,
            db_df_bis.db[data_name])
