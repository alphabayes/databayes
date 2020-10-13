import pydantic
from datetime import datetime
import typing

import pkg_resources

from .DiscreteDistribution import DiscreteDistribution
from .DiscreteVariable import DiscreteVariable

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401

if 'scipy' in installed_pkg:
    import scipy.stats  # noqa: F401


class ModelMetaData(pydantic.BaseModel):
    """Model Meta Info."""
    predict_index: str = pydantic.Field(
        None, description="Columns name to be used as prediction index")

    update_date: datetime = pydantic.Field(default=datetime.now(),
                                           description="Update date")


class FitParametersBase(pydantic.BaseModel):
    pass


class PredictParametersBase(pydantic.BaseModel):
    target_predict_mode: typing.Dict[str, dict] = pydantic.Field(
        {}, description="Optional predict mode for variable")

    smoothing: typing.Dict[str, dict] = pydantic.Field(
        {}, description="Optional smoothing of prediction")


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

    metadata: ModelMetaData = pydantic.Field(default={},
                                             description="Model metadata")

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

    def predict_specs(self, data, logger=None, **kwds):
        return self.model.predict(data[self.var_features], self.var_targets,
                                  logger=logger, **kwds)

    def predict(self, data, logger=None, **kwds):
        data_pred = self.predict_specs(data, logger=logger, **kwds)

        # Check special predict mode
        for var, predic_mode in self.predict_parameters.target_predict_mode.items():
            if var in data_pred.keys():
                if predic_mode.get("mode", None) == "RUL":
                    # Embed this into a method !
                    var_condition = predic_mode.get("var_condition", None)

                    scores_df = data_pred[var]["scores"].copy(deep=True)
                    scores_df.index = data[var_condition]

                    def apply_conditionning(dist):
                        cond_value = dist.name
                        dist_cond_idx = dist.index.get_loc(cond_value)
                        dist_shifted = dist.shift(-dist_cond_idx).fillna(0)
                        dist_cond = dist_shifted/dist_shifted.sum()
                        return dist_cond.fillna(0)

                    scores_cond_df = scores_df.apply(
                        apply_conditionning, axis=1)
                    # ipdb.set_trace()

                    data_pred[var]["scores"].values[:] = scores_cond_df.values[:]

        for var, smoothing in self.predict_parameters.smoothing.items():
            if var in data_pred.keys():
                scores_df = data_pred[var]["scores"].copy(deep=True)

                smoothing_param = smoothing.pop("mode_params", {})
                scores_smoothed_df = \
                    scores_df.rolling(axis=1, min_periods=0, center=True,
                                      **smoothing)\
                    .mean(**smoothing_param)

                # ipdb.set_trace()

                data_pred[var]["scores"].values[:] = scores_smoothed_df.values[:]

        return data_pred

    # TODO: IS IT RELEVANT TO KEEP FEATURE EXTRACTION METHOD HERE ?
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


# TODO: PUT THIS IN A SEPARATE FILE
class RandomUniformModel(MLModel):
    var_targets_dv: typing.Dict[str, DiscreteVariable] = pydantic.Field(
        {}, description="Discrete variable associated to target variables")

    def init_from_dataframe(self, data):
        for tv in self.var_targets:
            if data[tv].dtypes.name == "int64":
                self.var_targets_dv[tv] = \
                    DiscreteVariable(name=tv,
                                     domain=list(
                                         range(data[tv].max() + 1)))
            elif data[tv].dtypes.name == "category":
                self.var_targets_dv[tv] = \
                    DiscreteVariable(name=tv,
                                     domain=list(data[tv].cat.categories))

            else:
                self.var_targets_dv[tv] = \
                    DiscreteVariable(name=tv,
                                     domain=list(data[tv].unique()))

    def fit(self, data, logger=None, **kwds):
        self.init_from_dataframe(data)

    def predict_specs(self, data, logger=None, progress_mode=False, **kwds):

        # ipdb.set_trace()
        pred_res = {}
        for tv in self.var_targets:
            var_domain = self.var_targets_dv[tv].domain
            ddist = DiscreteDistribution(index=data.index,
                                         domain=var_domain)
            ddist.values[:] = 1/len(var_domain)
            pred_res.setdefault(tv, {"scores": ddist})

        return pred_res


class RandomGaussianHyperParameters(HyperParametersBase):
    mean_range: dict = pydantic.Field(
        {"min": -10, "max": 10}, description="")
    std_range: dict = pydantic.Field(
        {"min": 0.5, "max": 1.5}, description="")


class RandomGaussianModel(MLModel):
    var_targets_dv: typing.Dict[str, DiscreteVariable] = pydantic.Field(
        {}, description="Discrete variable associated to target variables")

    hyper_parameters: RandomGaussianHyperParameters = pydantic.Field(
        RandomGaussianHyperParameters(), description="")

    def init_from_dataframe(self, data):
        for tv in self.var_targets:
            if data[tv].dtypes.name == "int64":
                self.var_targets_dv[tv] = DiscreteVariable(name=tv,
                                                           domain=list(
                                                               range(data[tv].max() + 1)))
            else:
                self.var_targets_dv[tv] = DiscreteVariable(name=tv,
                                                           domain=data[tv].unique())

    def fit(self, data, logger=None, **kwds):
        pass

    def predict_specs(self, data, logger=None, progress_mode=False, **kwds):

        # ipdb.set_trace()
        pred_res = {}
        for tv in self.var_targets:
            var_domain = self.var_targets_dv[tv].domain
            ddist = DiscreteDistribution(index=data.index,
                                         domain=var_domain)
            ipdb.set_trace()
            ddist.values[:] = 1/len(var_domain)
            pred_res.setdefault(tv, {"scores": ddist})

        return pred_res


class ModelException(Exception):
    """ Exception type used to raise exceptions within Model derived classes """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
