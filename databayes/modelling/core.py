import pydantic
import typing_extensions
import abc
from datetime import datetime
import typing

import os
import sys
import subprocess
import pkg_resources

import datautils_common.modelling.DiscreteDistribution as dd

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class ModelMetaInfo(pydantic.BaseModel):
    """Model Meta Info."""
    update_date: datetime = pydantic.Field(default=datetime.now(),
                                           description="Update date")


class FitParametersBase(pydantic.BaseModel):
    pass


class PredictParametersBase(pydantic.BaseModel):
    pass


class HyperParametersBase(pydantic.BaseModel):
    pass


class MLModel(pydantic.BaseModel):
    """DEEP model schema."""
    id: str = pydantic.Field("", description="Unique id of the model")

    type: str = pydantic.Field(None, description="Model type")

    tags: typing.List[str] = pydantic.Field([], description="The model tags")

    fit_parameters: FitParametersBase = pydantic.Field(FitParametersBase(),
                                                       description="Model fitting parameters")

    predict_parameters: PredictParametersBase = pydantic.Field(
        PredictParametersBase(), description="Model prediction method parameters")

    hyper_parameters: HyperParametersBase = pydantic.Field(
        HyperParametersBase(), description="Hyper parameters")

    var_features: typing.List[str] = pydantic.Field(
        default=[], description="List of features variables")

    var_targets: typing.List[str] = pydantic.Field(
        default=[], description="List of target variables")

    model: typing.Any = pydantic.Field(
        None, description="Model storage structure")

    meta_info: ModelMetaInfo = pydantic.Field(default={},
                                              description="Model meta info")

    def __str__(self):

        return "\n\n".join([str(attr) + ": " + str(val) for attr, val in self.__dict__.items()])

    def init_from_dataframe(self, df):
        init_from_dataframe = getattr(self.model, "init_from_dataframe", None)
        if callable(init_from_dataframe):
            init_from_dataframe(df)

    def fit(self, data, logger=None, **kwds):
        # TODO: initialiser var_features à data.columns si vide
        self.model.fit(data, **self.fit_parameters.dict(),
                       logger=logger, **kwds)

    def predict(self, data, logger=None, **kwds):
        # ipdb.set_trace()
        return self.model.predict(data[self.var_features], self.var_targets,
                                  **self.predict_parameters.dict(),
                                  logger=logger, **kwds)

    def change_var_features_from_feature_selection(self, evaluate_scores):
        removed_variables = \
            [v for v in self.var_features
             if not (v in evaluate_scores.scores.keys())]
        self.var_features = [*evaluate_scores.scores.keys()]
        self.change_var_features(removed_variables, inplace=True)

    def new_features(self, removed_variables, inplace=False):
        new_var_features = self.var_features[:]
        for feature in removed_variables:
            new_var_features.remove(feature)
        if inplace:
            self.var_features = new_var_features
            return self.var_features
        else:
            return new_var_features

    def change_var_features(self, removed_variables, inplace):
        """Must return the new model (e.g. self if inplace)"""
        pass


class RandomUniformModel(MLModel):

    def fit(self, data, logger=None, **kwds):
        pass

    def predict(self, data, logger=None, progress_mode=False, **kwds):

        pred_res = {}
        for tv in self.var_targets:
            var_domain = data[tv].cat.categories.to_list()
            ddist = dd.DiscreteDistribution(index=data.index,
                                            domain=var_domain)
            ddist.values[:] = 1/len(var_domain)
            pred_res.setdefault(tv, {"scores": ddist})

        return pred_res


class ModelException(Exception):
    """ Exception type used to raise exceptions within Model derived classes """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
