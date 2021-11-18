import abc
import typing

from itertools import chain, combinations
import pandas as pd
import numpy as np
import pydantic
import textwrap
from intervals import FloatInterval

from ..utils import get_subclasses
from ..modelling import DiscreteDistribution

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401

if 'plotly' in installed_pkg:
    import plotly.io as pio  # noqa: F401
    from plotly.subplots import make_subplots  # noqa: F401
    import plotly.graph_objects as go  # noqa: F401
    import plotly.offline as pof  # noqa: F401

PandasDataFrame = typing.TypeVar('pd.core.dataframe')
PandasSeries = typing.TypeVar('pd.core.series')
DiscreteDistributionType = typing.TypeVar('DiscreteDistribution')


class PerformanceMeasureBase(pydantic.BaseModel, abc.ABC):

    name: str = pydantic.Field(
        "", description="Measure label")
    variables: list = pydantic.Field(
        [], description="Variables considered by the measure")

    result: typing.Dict[str, typing.Any] = pydantic.Field(
        None, description="Measurement result")

    group_by: typing.List[str] = pydantic.Field(
        [], description="Group by arguments")

    pred_prob: typing.Dict[str, DiscreteDistributionType] = pydantic.Field(
        None, description="Data test")

    data_test: PandasDataFrame = pydantic.Field(
        None, description="Data test")

    @staticmethod
    def merge(a, b, path=None):
        """merges dict b into dict a"""
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    PerformanceMeasureBase.merge(
                        a[key], b[key], path + [str(key)])
                elif isinstance(a[key], list) and isinstance(b[key], list):

                    a[key].extend(b[key])
                elif a[key] == b[key]:
                    pass  # same leaf value
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    @classmethod
    def from_dict(cls_self, **specs):

        sub_cls_dict = {sub_cls.__name__: sub_cls
                        for sub_cls in get_subclasses(cls_self)}

        obj_classname = specs.pop("cls")
        cls_obj = sub_cls_dict.get(obj_classname)

        if cls_obj is None:
            raise ValueError(f"{cls_obj} is not a subclass of {cls_self}")

        obj = cls_obj(**specs)

        return obj

    def evaluate(self, data_test=None, pred_prob=None, **kwargs):
        if not(data_test is None):
            self.data_test = data_test
        if not(pred_prob is None):
            self.pred_prob = pred_prob

        # Perform this measure only on numeric/interval variables
        self.variables = [tv for tv, dd in self.pred_prob.items()
                          if dd["scores"].variable.domain_type != "label"]

        self.result = {tv: None for tv in self.variables}

    # def get_dash_layout(self, app):
    #     pass

    # def run_app(self):
    #     app = dash.Dash(__name__, suppress_callback_exceptions=True,
    #                     external_stylesheets=[dbc.themes.BOOTSTRAP])
    #     server = app.server
    #     app.layout = self.get_dash_layout(app)
    #     app.run_server(debug=True)


class ConfusionMatrixMeasure(PerformanceMeasureBase):

    name: str = pydantic.Field(
        "Confusion matrix", description="Measure label")

    def evaluate(self, data_test, pred_prob):

        # Perform only on labelized variables
        self.variables = [tv for tv, dd in pred_prob.items()
                          if dd["scores"].variable.domain_type != "numeric"]

        self.result = dict()
        for tv in self.variables:
            network_pred = pred_prob[tv]["scores"].get_map()
            self.result[tv] = pd.crosstab(
                data_test[tv], network_pred['map_1'], dropna=False).to_dict('records')

        return self.result


class SuccessMeasure(PerformanceMeasureBase):
    # __slots__ = ('pred_success',)
    name: str = pydantic.Field(
        "Success", description="Measure label")

    map_k: typing.List[int] = pydantic.Field(
        [1], description="Number of most probable labels to be considered in accuracy computation")
    spread_threshold: float = pydantic.Field(
        1.0, description="Tolerance between MAP probability and k-th most label probability to be accepted",
        gte=0, lte=1)

    result: dict = pydantic.Field({"indep": [],
                                   "joint": [],
                                   "aggreg": []})

    pred_success: dict = pydantic.Field({})

    # Dict of DiscreteDistribution
    pred_prob: dict = pydantic.Field(
        {}, description="Data prediction probability")

    class Config:
        arbitrary_types_allowed = True

    def json(self, exclude=None, **kwargs):
        return super().json(exclude={"data_test", "pred_prob", "pred_success"}, **kwargs)

    def dict(self, exclude=None, **kwargs):
        return super().dict(exclude={"data_test", "pred_prob", "pred_success"}, **kwargs)

    # def __setattr__(self, attr, value):
    #     if attr in self.__slots__:
    #         object.__setattr__(self, attr, value)
    #     else:
    #         super(self.__class__, self).__setattr__(attr, value)

    def approx_equal(self, other, **kwargs):

        if self.map_k != other.map_k:
            return False

        if self.spread_threshold != other.spread_threshold:
            return False

        if set(self.result.keys()) != set(other.result.keys()):
            return False

        self_result_dfd = self.result_to_frame()
        other_result_dfd = other.result_to_frame()

        for result_key in self.result.keys():
            self_result_arr = self_result_dfd[result_key].to_numpy()
            other_result_arr = other_result_dfd[result_key].to_numpy()

            if not(np.allclose(self_result_arr,
                               other_result_arr,
                               **kwargs)):
                return False

        return True

    def evaluate_pred_success(self):

        self.pred_success = {k: pd.DataFrame(index=self.data_test.index,
                                             columns=self.variables)
                             for k in range(1, max(self.map_k) + 1)}

        for tv, prob in self.pred_prob.items():

            pred_map_kmax = prob["scores"].get_map(
                max(self.map_k))
            data_test_cur = self.data_test[tv]
            map_prob = pd.DataFrame(
                np.sort(-prob["scores"].values, axis=1)[:, :max(self.map_k)], index=pred_map_kmax.index)
            for k in range(1, max(self.map_k) + 1):

                if len(self.map_k) == 1:
                    pred_map_k = pred_map_kmax[:]
                else:
                    pred_map_k = pred_map_kmax.iloc[:, :k]

                self.pred_success[k][tv] = pd.Series([test_val in map_k
                                                      for test_val, map_k
                                                      in zip(data_test_cur.tolist(),
                                                             pred_map_k.values.tolist())],
                                                     index=pred_map_k.index)

                if (k > 1) and (self.spread_threshold < 1):
                    map_prob_threshold = (
                        map_prob.iloc[:, 0] - map_prob.iloc[:, k-1]).abs() < self.spread_threshold
                    self.pred_success[k][tv] = self.pred_success[k][tv] & map_prob_threshold

        return self.pred_success

    def evaluate(self, data_test, pred_prob):

        self.data_test = data_test

        # Perform measure only on labelized or interval variables
        self.pred_prob = {tv: p_prob for tv, p_prob in pred_prob.items()
                          if p_prob["scores"].variable.domain_type != "numeric"}

        self.variables = list(self.pred_prob.keys())

        self.evaluate_pred_success()

        for key in self.result.keys():
            evaluate_method = getattr(self, f"evaluate_{key}", None)
            if callable(evaluate_method):
                evaluate_method()

        return self.result

    def evaluate_indep(self):
        self.result["indep"] = [{
            "map_k": k,
            **self.pred_success[k].agg(["mean", "sum"]).to_dict()
        } for k in self.map_k]

    def evaluate_joint(self):

        nb_variables = len(self.variables)
        var_comb_list = list(chain.from_iterable(combinations(self.variables, r)
                                                 for r in range(nb_variables + 1)))

        var_comb_str = ["--".join(cmb) for cmb in var_comb_list]
        var_comb_str[0] = "None"

        joint_k_df = pd.DataFrame(index=self.map_k,
                                  columns=var_comb_str)
        joint_k_df.index.name = "map_k"

        for k in self.map_k:

            for var_joint, var_str in zip(var_comb_list, var_comb_str):
                var_joint_cmpl = [v for v in self.variables
                                  if not(v in var_joint)]
                # Testing XOR
                success_joint = self.pred_success[k].loc[:, var_joint].all(
                    axis=1) if len(var_joint) > 0 else True
                fail_cmpl = ~self.pred_success[k].loc[:, var_joint_cmpl].any(
                    axis=1) if len(var_joint_cmpl) > 0 else True

                joint_k_df.loc[k, var_str] = (
                    success_joint & fail_cmpl).mean(axis=0)

        self.result["joint"] = joint_k_df.reset_index().to_dict("records")

    def evaluate_aggreg(self):
        # Implement the possibility of using custom aggreg function
        self.result["aggreg"] = [{
            "map_k": k,
            **self.pred_success[k].sum(axis=1).agg(["mean", "sum", "std"]).to_dict()
        } for k in self.map_k]

    def result_indep_to_frame(self):
        indep_index = pd.MultiIndex.from_product([self.map_k],
                                                 names=["map_k"])
        indep_columns = pd.MultiIndex.from_product(
            [self.variables,
             ["mean", "sum"]], names=["variable", "stats"])

        indep_df = pd.DataFrame(index=indep_index,
                                columns=indep_columns)

        result_indep_df = pd.DataFrame(self.result["indep"]).set_index("map_k")

        for var, data in result_indep_df.items():
            indep_df.loc[:, (var, slice(None))] = pd.DataFrame(
                data.to_list()).values

        return indep_df

    def result_aggreg_to_frame(self):
        return pd.DataFrame(self.result["aggreg"]).set_index("map_k")

    def result_joint_to_frame(self):
        return pd.DataFrame(self.result["joint"]).set_index("map_k")

    def result_to_frame(self):

        dfd = {}
        for key in self.result.keys():
            to_frame_method = getattr(self, f"result_{key}_to_frame", None)
            if callable(to_frame_method):
                dfd[key] = to_frame_method()

        return dfd


class ErrorResult(pydantic.BaseModel):

    err: PandasSeries = pydantic.Field(
        None, description="Error for each prediction")
    ae: PandasSeries = pydantic.Field(
        None, description="Absolute Error for each prediction")
    ape: PandasSeries = pydantic.Field(
        None, description="Absolute Percentage Error for each prediction")
    sape: PandasSeries = pydantic.Field(
        None, description="Symmetric APE for each prediction")
    se: PandasSeries = pydantic.Field(
        None, description="Squared Error for each prediction")
    pred: PandasSeries = pydantic.Field(
        None, description="Predictions")
    summary_group: PandasDataFrame = pydantic.Field(
        None, description="Indicators summary by group")


class ErrorMeasure(PerformanceMeasureBase):

    name: str = pydantic.Field(
        "Error measures", description="Measure label")

    calculation_method: str = pydantic.Field('eap', description="Whether we calculate the predicted scalar by \
                                                    calculating the eap of the DiscreteDistribution predicted or the map of this distribution")

    ensure_finite: bool = pydantic.Field(
        True, description="Ensure finite error computation by replacing infinite bounds")

    upper_bound: float = pydantic.Field(
        float("inf"), description="Upper bound definition for EAP estimation. Useful in cas of infinite interval")
    lower_bound: float = pydantic.Field(
        -float("inf"), description="Upper bound definition for EAP estimation. Useful in cas of infinite interval")

    result: typing.Dict[str, ErrorResult] = pydantic.Field(
        None, description="Results")

    @pydantic.validator('calculation_method')
    def check_type(cls, val):
        if not (val in ['eap', 'map']):
            raise ValueError(f"{val} isn't a calculation method")
        return val

    def evaluate(self, data_test=None, pred_prob=None, **kwargs):
        super().evaluate(data_test, pred_prob, **kwargs)

        self.result = {}
        # def evaluate_ae(self):

        #     # self.variables = list(pred_prob.keys())

        #     # self.result["ae"] = dict()
        #     # self.result["pred"] = dict()
        numeric_comp_params = {
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "ensure_finite": self.ensure_finite,
        }

        for tv in self.variables:

            self.result[tv] = ErrorResult()
            # TODO:
            # Make .E() and .get_map() work transparently whatever is the domain type !
            # - OK for E()
            # - TODO for .get_map add options to return float and ensure finite

            # Compute prediction with respect
            if self.pred_prob[tv]['scores'].variable.domain_type == 'numeric':

                if self.calculation_method == 'eap':
                    self.result[tv].pred = \
                        self.pred_prob[tv]['scores'].E(**numeric_comp_params)
                elif self.calculation_method == 'map':
                    self.result[tv].pred = \
                        self.pred_prob[tv]['scores']\
                        .get_map().loc[:, 'map_1'].astype(float)
                else:
                    raise ValueError(
                        f"Calculation method not supported for numeric "
                        f"variable: {self.calculation_method}")

            elif self.pred_prob[tv]['scores'].variable.domain_type == 'interval':

                if self.calculation_method == 'eap':
                    self.result[tv].pred = \
                        self.pred_prob[tv]['scores'].E(**numeric_comp_params)

                elif self.calculation_method == 'map':

                    map_it = self.pred_prob[tv]['scores'].get_map(
                    ).loc[:, 'map_1']

                    self.result[tv].pred = \
                        map_it.apply(lambda x: FloatInterval.from_string(x).centre)\
                              .astype(float)
                else:
                    raise ValueError(
                        f"Calculation method not supported "
                        f"for interval variable: {self.calculation_method}")

            if self.data_test[tv].dtype.name == "interval":
                d_test_tv = self.data_test[tv].apply(lambda x: x.mid)
            else:
                d_test_tv = self.data_test[tv].astype(float)

            self.result[tv].err = \
                (self.result[tv].pred - d_test_tv)
            self.result[tv].err.name = "error"

            self.result[tv].ae = self.result[tv].err.abs()
            self.result[tv].ae.name = "AE"

            idx_0 = d_test_tv == 0
            self.result[tv].ape = self.result[tv].ae/d_test_tv
            self.result[tv].ape.loc[idx_0] = np.nan
            self.result[tv].ape.name = "APE"

            self.result[tv].sape = self.result[tv].ae / \
                (self.result[tv].pred + d_test_tv)/2
            self.result[tv].sape.name = "sAPE"

            self.result[tv].se = self.result[tv].err.pow(2)
            self.result[tv].se.name = "SE"

        self.update_summary()

    def update_summary(self):

        data_group_df = self.data_test[self.group_by]

        for tv, result in self.result.items():

            data_indic_df = \
                pd.concat([data_group_df] +
                          [
                              result.ae,
                              result.ape,
                              result.sape,
                              result.se,
                ], axis=1)

            data_perf_grp = data_indic_df.groupby(self.group_by) \
                if len(self.group_by) > 0 else data_indic_df

            mae_s = \
                data_perf_grp["AE"].mean()\
                                   .rename("MAE")

            mape_s = \
                data_perf_grp["APE"].mean()\
                .rename("MAPE")
            smape_s = \
                data_perf_grp["sAPE"].mean()\
                .rename("sMAPE")

            rmse_s = \
                data_perf_grp["SE"].mean()\
                                   .pow(0.5)\
                                   .rename("RMSE")

            self.result[tv].summary_group = \
                pd.concat([mae_s, mape_s, smape_s, rmse_s],
                          axis=1)

    def get_summary(self, tv=None, indics=[]):

        tv = tv or self.variables[0]

        if len(indics) == 0:
            indics = self.result[tv].summary_group.columns

        summary_df = self.result[tv].summary_group[indics]
        summary_df.columns.name = "indic"

        return summary_df


class QuantilesResult(pydantic.BaseModel):

    spread: PandasDataFrame = pydantic.Field(
        None, description="Quantiles interval spreads")
    accuracy: PandasSeries = pydantic.Field(
        None, description="Indicates if the real value is in the quantile intervals")
    pred: PandasDataFrame = pydantic.Field(
        None, description="Predictions")
    spread_group: PandasDataFrame = pydantic.Field(
        None, description="Indicators summary by group")
    accuracy_group: PandasDataFrame = pydantic.Field(
        None, description="Indicators summary by group")


class QuantilesMeasure(PerformanceMeasureBase):

    name: str = pydantic.Field(
        "Quantiles measures", description="Measure label")

    quantiles: typing.List[float] = pydantic.Field(
        [0.1, 0.25], description="Lower quantiles lists")

    result: typing.Dict[str, QuantilesResult] = pydantic.Field(
        None, description="Results")

    indicator_names: typing.List[str] = pydantic.Field(
        ["spread", "accuracy"], description="Name of computed indicators")

    def evaluate(self, data_test=None, pred_prob=None, **kwargs):
        super().evaluate(data_test, pred_prob, **kwargs)

        self.result = {}

        for tv in self.variables:
            pred_prob = self.pred_prob[tv]['scores']

            self.result[tv] = QuantilesResult()
            self.result[tv].pred = pd.DataFrame(index=self.data_test.index)
            self.result[tv].spread = pd.DataFrame(index=self.data_test.index)
            self.result[tv].accuracy = pd.DataFrame(
                index=self.data_test.index)

            for q in self.quantiles:
                self.result[tv].pred[q] = \
                    pred_prob.quantile(q, ensure_finite=True)
                self.result[tv].pred[1 - q] = \
                    pred_prob.quantile(1 - q, ensure_finite=True)

                self.result[tv].spread[q] = \
                    self.result[tv].pred[1 - q] - self.result[tv].pred[q]
                self.result[tv].accuracy[q] = \
                    (self.result[tv].pred[q] < self.data_test[tv]) & \
                    (self.result[tv].pred[1 - q] > self.data_test[tv])

        self.update_summary()

        # self.compute_qspread_timeline()

    def update_summary(self):

        data_group_df = self.data_test[self.group_by]

        for tv, result in self.result.items():

            data_spread_df = \
                pd.concat([data_group_df, result.spread],
                          axis=1)

            data_spread_grp = data_spread_df.groupby(self.group_by) \
                if len(self.group_by) > 0 else data_spread_df

            self.result[tv].spread_group = \
                data_spread_grp.mean()

            # ddd = data_spread_df[data_spread_df["id"] == "S00095"]
            # ipdb.set_trace()

            data_accuracy_df = \
                pd.concat([data_group_df, result.accuracy],
                          axis=1)

            data_accuracy_grp = data_accuracy_df.groupby(self.group_by) \
                if len(self.group_by) > 0 else data_accuracy_df

            self.result[tv].accuracy_group = \
                data_accuracy_grp.mean()

    def get_summary(self, tv=None, indics=["spread", "accuracy"]):

        tv = tv or self.variables[0]

        summary_df_list = []
        for indic in indics:
            summary_df_cur = getattr(self.result[tv],
                                     f"{indic}_group")

            summary_df_cur = \
                summary_df_cur.rename(
                    columns={q: f"{q:.0%}-{indic[:3]}"
                             for q in summary_df_cur.columns})

            summary_df_list.append(summary_df_cur)

        # indics_disp = indics
        # quantiles_disp = [f"{q:.0%}"
        #                   for q in self.quantiles]
        # col_index = \
        #     pd.MultiIndex.from_product(
        #         [indics_disp, quantiles_disp])
        summary_df = pd.concat(summary_df_list, axis=1)
        summary_df.columns.name = "indic"
        #summary_df.columns = col_index

        return summary_df
