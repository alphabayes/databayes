from .MLModel import FitParametersBase, MLModel
from .Potential import DFCPD
import pandas as pd
import pydantic
import typing
import pkg_resources
import warnings

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


class BayesianFitParameters(FitParametersBase):
    update_fit: bool = pydantic.Field(
        False, description="Indicates if fitting process will update current CPT parameters during the fitting process (update_fit=True) or erase current CPT parameters with results of the last fitting process (update_fit=False)")
    update_decay: float = pydantic.Field(0, description="Fitting Update decay")


class NaturalBayesianModel(MLModel):

    type: str = pydantic.Field(
        "NaturalBayesianModel", description="Type of the model")

    model: typing.Dict[str, DFCPD] = pydantic.Field(
        {}, description="CPD object for eache target variable")

    fit_parameters: BayesianFitParameters = \
        pydantic.Field(BayesianFitParameters(),
                       description="Bayesian fit parameters object")

    @pydantic.validator('type')
    def check_type(cls, val):
        if val != "NaturalBayesianModel":
            raise ValueError("Not BayesianNetworkModel object")
        return val

    def fit_specs(self, data, logger=None, **kwds):

        for var_target in self.var_targets:

            var = [var_target] + self.var_features + self.var_extra

            if self.model.get(var_target, None) is None:
                self.model[var_target] = \
                    DFCPD.init_from_dataframe(data[var],
                                              var_norm=var_target)

            self.model[var_target].fit(data, **self.fit_parameters.dict(),
                                       logger=logger, **kwds)
