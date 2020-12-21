# -*- coding: utf-8 -*-

import os
import yaml
import pydantic
import typing
from databayes.utils import etl
from datetime import date, datetime, timedelta
import logging
import pandas as pd
import numpy as np
import pkg_resources
from deepmerge import conservative_merger

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class OHLCVIndicatorBase(pydantic.BaseModel):

    name: str = pydantic.Field(
        ..., description="Indicator name")
    name_label: str = pydantic.Field(
        None, description="Indicator name label used in pretty pretting")
    description: str = pydantic.Field(
        None, description="Indicator description")
    num_suffix: str = pydantic.Field(
        "__num", description="Suffix related to the numeric version of the indicator if exists")
    factor_suffix: str = pydantic.Field(
        "__factor", description="Suffix related to the categorical version of the indicator if exists")

    discretization: dict = pydantic.Field(
        None, description="Discretization specifications")

    indic_num: pd.Series = pydantic.Field(
        None, description="Numeric values of the indicator")
    indic_factor: pd.Series = pydantic.Field(
        None, description="Categorical values of the indicator")

    open_var: str = pydantic.Field(
        "open", description="Open data variable name")
    high_var: str = pydantic.Field(
        "high", description="High data variable name")
    low_var: str = pydantic.Field(
        "low", description="Low data variable name")
    close_var: str = pydantic.Field(
        "close", description="Close data variable name")
    volume_var: str = pydantic.Field(
        "volume", description="Volume data variable name")

    plot_styling: dict = pydantic.Field(
        {}, description="Indicator plot styling parameters")

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator('name_label', always=True)
    def set_default_name_label(cls, name_label, values):
        return name_label if not(name_label is None)\
            else values.get("name")

    def __init__(self, logging=logging, **data: typing.Any):
        super().__init__(**data)

        # Default styling parameters
        default_parameters_filename = \
            os.path.join(os.path.dirname(__file__),
                         "ohlcv_indicator_plot_default.yaml")

        with open(default_parameters_filename, 'r', encoding="utf-8") \
                as yaml_file:
            try:
                plot_styling_default = yaml.load(yaml_file,
                                                 Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                logging.error(exc)

        conservative_merger.merge(self.plot_styling,
                                  plot_styling_default)

    def compute(self, data, logging=logging, **kwrds):
        logging.info(f">> Compute {self.name}")

        self.compute_numeric(data, logging=logging)

        if not(self.indic_num is None):
            self.indic_num.name += self.num_suffix

        self.compute_factor(data, logging=logging)

        if not(self.indic_factor is None):
            self.indic_factor.name += self.factor_suffix

        return self.indic_num, self.indic_factor

    def compute_numeric(self, data, logging=logging, **kwrds):
        logging.info(f">>> Indicator {self.name} has no numeric version")

        return self.indic_num

    def compute_factor(self, data, logging=logging, **kwrds):

        if not(self.discretization is None) and not(self.indic_num is None):
            self.indic_factor = pd.cut(self.indic_num, **self.discretization)
            self.indic_factor.name = \
                self.indic_factor.name.replace(self.num_suffix, "")
        else:
            logging.info(f">>> Indicator {self.name} has no factor version")

        return self.indic_factor


class RSIIndicator(OHLCVIndicatorBase):

    window_length: int = pydantic.Field(
        ..., description="Time window of rolling mean to be "
        "considered in RSI computation")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Relative Strength Index (RSI) from equity close data"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else f"RSI T{values.get('window_length')}"

        return values

    def compute_numeric(self,
                        data_ohlcv,
                        **kwrds):

        close_delta = data_ohlcv[self.close_var].diff()

        delta_up = close_delta.copy(deep=True)
        delta_up[delta_up < 0] = 0
        delta_down = close_delta.copy(deep=True)
        delta_down[delta_down > 0] = 0

        roll_up = delta_up.rolling(self.window_length).mean()
        roll_down = delta_down.abs().rolling(self.window_length).mean()

        rs = roll_up.copy(deep=True)
        idx_rdp = roll_down > 0
        rs.loc[idx_rdp] = rs.loc[idx_rdp].div(roll_down.loc[idx_rdp], axis=0)
        self.indic_num = 100.0 - (100.0/(1.0 + rs))

        self.indic_num.name = self.name


class HammerIndicator(OHLCVIndicatorBase):
    """ Hammer signal as describe in https://www.whselfinvest.fr/fr-fr/plateforme-de-trading/strategies-trading-gratuites/systeme/22-hammer """

    body_min_threshold: float = pydantic.Field(
        0, description="Body lower bound to consider hammer calculation")
    past_direction_order: int = pydantic.Field(
        2, description="Number of time units at same direction in the past just before the hammer")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Relative generalized hammer indicators"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else "Generalized hammer"

        return values

    def compute_numeric(self,
                        data_ohlcv_df,
                        **kwrds):
        # DEBUG
        # datetime_start = datetime(2020, 10, 19, 0, 0)
        # datetime_end = datetime(2020, 10, 19, 12, 0)
        # data_ohlcv_df = data_ohlcv_df.loc[datetime_start:datetime_end]

        body_range = \
            data_ohlcv_df[self.close_var] - data_ohlcv_df[self.open_var]

        body_past_range_list = \
            [body_range.shift(i)
             for i in range(self.past_direction_order + 1, 1, -1)]

        body_past_range_df = pd.concat(body_past_range_list, axis=1)

        body_past_indic = \
            -(body_past_range_df < 0).all(axis=1).astype(int) + \
            (body_past_range_df >= 0).all(axis=1).astype(int)

        hammer_df = self.compute_hammer(data_ohlcv_df)

        body_direction_cur = -(body_range < 0).astype(int) + \
            (body_range >= 0).astype(int)

        hammer_indic = -(hammer_df.hammer.shift(1) < 0).astype(int) + \
            (hammer_df.hammer.shift(1) >= 0).astype(int)

        # Indicate if hammer has the opposite direction with the past
        hammer_trend_indic = \
            ((body_past_indic*hammer_indic) < 0).astype(int)
        # Indicate if we have a change of trends
        trend_change_indic = \
            ((body_past_indic*body_direction_cur) < 0).astype(int)

        self.indic_num = \
            hammer_df.hammer.shift(1)*hammer_trend_indic*trend_change_indic
        # self.indic_num = \
        #     hammer_df.hammer.shift(1)*hammer_trend_indic

        # ipdb.set_trace()

        # Cleaning
        # Remove NaN
        self.indic_num = \
            self.indic_num.replace([np.inf, -np.inf], np.nan)\
                          .fillna(0)

        self.indic_num.name = self.name

        return self.indic_num

    def compute_hammer(self, data_ohlcv_df):

        body_range = \
            data_ohlcv_df[self.close_var] - data_ohlcv_df[self.open_var]

        data_open = data_ohlcv_df[self.open_var]
        data_close = data_ohlcv_df[self.close_var]
        data_low = data_ohlcv_df[self.low_var]
        data_high = data_ohlcv_df[self.high_var]
        body_range = data_close - data_open

        shadow_low_base = data_open*(body_range > 0) + \
            data_close*(body_range <= 0)
        shadow_low = shadow_low_base - data_low

        shadow_up_base = data_close*(body_range > 0) + \
            data_open*(body_range <= 0)
        shadow_up = data_high - shadow_up_base

        # Cancel effect of tiny body
        idx_body_threshold = body_range.abs() < self.body_min_threshold
        body_range.loc[idx_body_threshold] = np.nan

        hammer = (shadow_low - shadow_up)/body_range.abs()
        # Cleaning
        # Remove NaN
        hammer = \
            hammer.replace([np.inf, -np.inf], np.nan)\
                  .fillna(0)

        hammer_all_df = pd.concat([hammer.rename("hammer"),
                                   shadow_low.rename("shadow_low"),
                                   shadow_up.rename("shadow_up"),
                                   body_range.rename("body")], axis=1)

        return hammer_all_df
