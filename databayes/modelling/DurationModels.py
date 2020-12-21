import numpy as np
import pandas as pd
import pydantic
import typing
from reliability.Fitters import Fit_Weibull_2P

from .DiscreteDistribution import DiscreteDistribution
from .MLModel import PredictParametersBase, FitParametersBase, MLModel, HyperParametersBase
import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


class DurationModelSingleStateBase(MLModel):

    filter: dict = pydantic.Field(
        {}, description="Filtering specs to prepare data")

    def json(self, exclude=None, **kwargs):
        return super().json(exclude={"model"}, **kwargs)

    def dict(self, exclude=None, **kwargs):
        return super().dict(exclude={"model"}, **kwargs)

    def get_event_data(self, data_df):

        idx_filter = (data_df[list(self.filter)] ==
                      pd.Series(self.filter)).all(axis=1)
        return data_df[idx_filter]


class Weibull(DurationModelSingleStateBase):

    model: typing.Dict[str, typing.Any] = pydantic.Field(
        {}, description="Model storage structure")

    def fit_specs(self, data_df, logger=None, **kwds):

        data_fit_df = self.get_event_data(data_df)

        for tv in self.var_targets:
            model = Fit_Weibull_2P(failures=data_fit_df[tv].to_list(),
                                   print_results=False)
            self.model[tv] = model

    def predict_specs(self, data_df, logger=None, progress_mode=False, **kwds):

        pred_res = {}
        for tv in self.var_targets:
            var_bins = \
                self.predict_parameters.var_discrete_support\
                                       .get(tv, {})\
                                       .get("bins", None)
            # ipdb.set_trace()
            if not(var_bins):
                raise ValueError(
                    "Duration models needs target prediction bins")

            pred_res[tv] = \
                {"scores": DiscreteDistribution(index=data_df.index,
                                                bins=var_bins)}

            cdf = np.array(self.model[tv].distribution.CDF(
                xvals=var_bins, show_plot=False))
            probs = cdf[1:] - cdf[:-1]

            pred_res[tv]["scores"].values[:] = probs

        return pred_res
