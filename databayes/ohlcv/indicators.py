# -*- coding: utf-8 -*-

import pydantic
import typing
import math
import logging
import pandas as pd
import numpy as np
import pkg_resources
from ..utils import get_subclasses

#from deepmerge import conservative_merger

PandasSeries = typing.TypeVar('pandas.core.frame.Series')

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class OHLCVIndicatorBase(pydantic.BaseModel):

    name: str = pydantic.Field(
        None, description="Indicator name")
    name_label: str = pydantic.Field(
        None, description="Indicator name label used in pretty pretting")
    description: str = pydantic.Field(
        None, description="Indicator description")

    values: PandasSeries = pydantic.Field(
        None, description="Values of the indicator")

    ohlcv_names: dict = pydantic.Field(
        {}, description="OHLCV name dictionnary")
    # plot_styling: dict = pydantic.Field(
    #     {}, description="Indicator plot styling parameters")

    # class Config:
    #     arbitrary_types_allowed = True

    @pydantic.validator('name_label', always=True)
    def set_default_name_label(cls, name_label, values):
        return name_label if not(name_label is None)\
            else values.get("name")

    def __init__(self, logging=logging, **data: typing.Any):
        super().__init__(**data)

        # # Default styling parameters
        # default_parameters_filename = \
        #     os.path.join(os.path.dirname(__file__),
        #                  "ohlcv_indicator_plot_default.yaml")

        # with open(default_parameters_filename, 'r', encoding="utf-8") \
        #         as yaml_file:
        #     try:
        #         plot_styling_default = yaml.load(yaml_file,
        #                                          Loader=yaml.SafeLoader)
        #     except yaml.YAMLError as exc:
        #         logging.error(exc)

        # conservative_merger.merge(self.plot_styling,
        #                           plot_styling_default)

    @classmethod
    def from_dict(basecls, **specs):
        cls_sub_dict = {cls.__name__: cls for cls in get_subclasses(basecls)}

        clsname = specs.pop("cls")
        cls = cls_sub_dict.get(clsname)

        if cls is None:
            raise ValueError(
                f"{clsname} is not a subclass of {basecls.__name__}")

        return cls(**specs)

    def get_required_past_horizon(self):
        past_hrz = [0]
        if hasattr(self, "lag"):
            past_hrz.append(self.lag - 1)
        elif hasattr(self, "window_size"):
            past_hrz.append(self.window_size - 2)

        return max(past_hrz)

    def split_ohlcv(self, data_ohlcv_df):

        open_var = self.ohlcv_names.get("open", "open")
        high_var = self.ohlcv_names.get("high", "high")
        low_var = self.ohlcv_names.get("low", "low")
        close_var = self.ohlcv_names.get("close", "close")
        volume_var = self.ohlcv_names.get("volume", "volume")

        return \
            data_ohlcv_df[open_var], \
            data_ohlcv_df[high_var], \
            data_ohlcv_df[low_var], \
            data_ohlcv_df[close_var], \
            data_ohlcv_df[volume_var]

    def compute(self, data, logging=logging, **kwrds):
        logging.debug(f">> Compute {self.name}")
        raise NotImplementedError(
            f"Indicator {self.name} has no compute method")


class RSI(OHLCVIndicatorBase):

    window_size: int = pydantic.Field(
        ..., description="Time window of rolling mean to be "
        "considered in RSI computation")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Relative Strength Index (RSI) from equity close data"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else f"RSI T{values.get('window_size')}"

        return values

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        close_delta = data_close.diff()

        delta_up = close_delta.copy(deep=True)
        delta_up[delta_up < 0] = 0
        delta_down = close_delta.copy(deep=True)
        delta_down[delta_down > 0] = 0

        roll_up = delta_up.rolling(self.window_size).mean()
        roll_down = delta_down.abs().rolling(self.window_size).mean()

        rs = roll_up.copy(deep=True)
        idx_rdp = roll_down > 0
        rs.loc[idx_rdp] = rs.loc[idx_rdp].div(roll_down.loc[idx_rdp], axis=0)

        self.values = 100.0 - (100.0/(1.0 + rs))
        self.values.name = self.name

        return self.values


class MAReturns(OHLCVIndicatorBase):
    """ Compute (low shadow - high shadow)/body ratio. """

    window_size: int = pydantic.Field(
        0, description="Time unit lag to compute the indicator")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Moving Average Returns"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else f"MAR{values.get('window_size', '')}"

        return values

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        returns = \
            1 - data_close/data_open

        self.values = returns.rolling(self.window_size).mean()
        self.values.name = self.name

        return self.values


class Returns(OHLCVIndicatorBase):
    """ Compute (low shadow - high shadow)/body ratio. """

    lag: int = pydantic.Field(
        0, description="Time unit lag to compute the indicator")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Returns"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else "Returns"

        return values

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        self.values = \
            1 - data_close.shift(self.lag)/data_open.shift(self.lag)

        self.values.name = self.name

        return self.values


class GeneralizedHammer(OHLCVIndicatorBase):
    """ Compute (low shadow - high shadow)/body ratio. """

    lag: int = pydantic.Field(
        0, description="Time unit lag to compute the indicator")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Generalized hammer"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else "Generalized hammer"

        return values

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        data_close = data_close.shift(self.lag)
        data_open = data_open.shift(self.lag)
        data_high = data_high.shift(self.lag)
        data_low = data_low.shift(self.lag)

        body_range = data_close - data_open

        shadow_low_base = data_open*(body_range > 0) + \
            data_close*(body_range <= 0)

        shadow_high_base = data_open*(body_range < 0) + \
            data_close*(body_range >= 0)

        shadow_low = shadow_low_base - data_low
        shadow_high = data_high - shadow_high_base

        shadow_delta = shadow_low - shadow_high

        self.values = shadow_delta/body_range.abs()
        # To avoid numerical problem :
        idx_body_range_0 = body_range.abs() == 0
        self.values.loc[idx_body_range_0] = 0

        self.values.name = self.name

        return self.values


class MovingVolumeQuantile(OHLCVIndicatorBase):
    """ Compute moving volume quantile. """

    window_size: int = pydantic.Field(
        50, description="Time window size of rolling quantile to be "
        "considered in computations")

    bins: list = pydantic.Field(
        [0.5, 0.75, 0.9], description="Quantile bins to consider")

    returns_dummies: bool = pydantic.Field(
        False, description="Indicates if we want to return dummy indicators")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Moving volume quantile"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else "Moving volume quantile"

        return values

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        # TO BE CONTINUED HERE !
        # Add an attribute to control the quantile bins
        # This indicator produce a categorical series
        indic_name = self.name

        self.values = pd.DataFrame(index=data_open.index,
                                   columns=[indic_name])
        q_bin_list = []

        data_volume_roll = data_volume.rolling(self.window_size)

        bin_left = 0
        bin_right = self.bins[0]
        volume_q_left = 0
        volume_q_right = data_volume_roll.quantile(bin_right)
        q_bin_str = f"q{100*bin_left:.0f}_{100*bin_right:.0f}"
        q_bin_list.append(q_bin_str)

        idx_q_bin = (volume_q_left <= data_volume) & \
            (data_volume < volume_q_right)
        if self.returns_dummies:
            self.values[f"{indic_name}_{q_bin_str}"] = idx_q_bin
        self.values.loc[idx_q_bin, indic_name] = q_bin_str

        for bin_left, bin_right in zip(self.bins[:-1], self.bins[1:]):

            volume_q_left = data_volume_roll.quantile(bin_left)
            volume_q_right = data_volume_roll.quantile(bin_right)
            q_bin_str = f"q{100*bin_left:.0f}_{100*bin_right:.0f}"
            q_bin_list.append(q_bin_str)

            idx_q_bin = (volume_q_left <= data_volume) & \
                (data_volume < volume_q_right)
            if self.returns_dummies:
                self.values[f"{indic_name}_{q_bin_str}"] = idx_q_bin
            self.values.loc[idx_q_bin, indic_name] = q_bin_str

        bin_left = self.bins[-1]
        bin_right = 1
        volume_q_left = data_volume_roll.quantile(bin_left)
        volume_q_right = float("inf")
        q_bin_str = f"q{100*bin_left:.0f}_{100*bin_right:.0f}"
        q_bin_list.append(q_bin_str)

        idx_q_bin = (volume_q_left <= data_volume) & \
            (data_volume < volume_q_right)
        if self.returns_dummies:
            self.values[f"{indic_name}_{q_bin_str}"] = idx_q_bin
        self.values.loc[idx_q_bin, indic_name] = q_bin_str

        q_bin_cats = pd.api.types.CategoricalDtype(categories=q_bin_list,
                                                   ordered=True)

        self.values[indic_name] = self.values[indic_name].astype(q_bin_cats)

        return self.values


class RangeIndex(OHLCVIndicatorBase):
    """ Compute normalized close position from range. """

    window_size: int = pydantic.Field(
        120, description="Time windows to compute range")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "RangeIndex"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else "RangeIndex"

        return values

    def update(self, data_ohlcv_df, **kwrds):
        return self.compute(data_ohlcv_df, **kwrds)

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        close_min = data_close.rolling(self.window_size).min()
        close_max = data_close.rolling(self.window_size).max()

        close_range = close_max - close_min

        self.values = (data_close - close_min).div(close_range)

        self.values.name = self.name

        return self.values


# TODO: TO BE CONTINUED
class Support(OHLCVIndicatorBase):
    """ Compute (low shadow - high shadow)/body ratio. """

    window_size: int = pydantic.Field(
        120, description="Time windows to search supports")

    bin_width: int = pydantic.Field(
        100, description="Time windows to search supports")

    @pydantic.validator('description', always=True)
    def set_default_description(cls, description):
        return "Support"

    @pydantic.root_validator(pre=True)
    def set_default_name_label(cls, values):
        values["name_label"] = values.get("name_label")\
            if not(values.get("name_label", None) is None)\
            else "Support"

        return values

    def compute_support_indic(self, data_close):

        # idx_support = \
        #     (data_close.shift(2) > data_close.shift(1)) & \
        #     (data_close.shift(1) > data_close) & \
        #     (data_close < data_close.shift(-1)) & \
        #     (data_close.shift(-1) < data_close.shift(-2))

        idx_support = \
            (data_close.shift(1) > data_close) & \
            (data_close < data_close.shift(-1))

        close_min = math.floor(data_close.min()/self.bin_width)*self.bin_width
        close_max = math.ceil(data_close.max()/self.bin_width)*self.bin_width

        bins = \
            np.arange(close_min - self.bin_width/2,
                      close_max + self.bin_width/2,
                      self.bin_width)

        data_close_disc = pd.cut(data_close, bins)

        data_close_disc.loc[idx_support].value_counts()

        return (data_close_disc == data_close_disc.iloc[-1]).sum()

    def compute(self,
                data_ohlcv_df,
                **kwrds):

        data_open, data_high, data_low, data_close, data_volume = \
            self.split_ohlcv(data_ohlcv_df)

        # TO BE CONTINUED

        ddd = data_close.rolling(self.window_size).apply(
            self.compute_support_indic)

        ipdb.set_trace()

        return idx_support

        body_range = data_close - data_open

        shadow_low_base = data_open*(body_range > 0) + \
            data_close*(body_range <= 0)

        shadow_high_base = data_open*(body_range < 0) + \
            data_close*(body_range >= 0)

        shadow_low = shadow_low_base - data_low
        shadow_high = data_high - shadow_high_base

        shadow_delta = shadow_low - shadow_high

        self.values = shadow_delta/body_range.abs()
        # To avoid numerical problem :
        idx_body_range_0 = body_range.abs() == 0
        self.values.loc[idx_body_range_0] = 0

        self.values.name = self.name

        return self.values
