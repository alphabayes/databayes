import pydantic
from datetime import datetime
import typing

import pkg_resources

from .DiscreteDistribution import DiscreteDistribution
from .DiscreteVariable import DiscreteVariable
from ..utils import Discretizer

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

    var_discrete_support: typing.Dict[str, dict] = pydantic.Field(
        {}, description="Dictionary specifying for variables support (bins or domain)")

    predict_postprocess: typing.Dict[str, dict] = pydantic.Field(
        {}, description="Optional predict postprocessing for variable")

    # var_smoothing: typing.Dict[str, dict] = pydantic.Field(
    #     {}, description="Optional smoothing of prediction")


class HyperParametersBase(pydantic.BaseModel):
    pass


# class MLModelVarPreprocess(pydantic.BaseModel):
#     discretize: dict = pydantic.Field(
#         {}, description="Discretization specifications")


class MLModel(pydantic.BaseModel):
    """ML model schema."""

    # FUTUR USAGE: DB Storing
    id: str = pydantic.Field("", description="Unique id of the model")
    tags: typing.List[str] = pydantic.Field([], description="The model tags")

    type: str = pydantic.Field(None, description="Model type")

    fit_parameters: FitParametersBase = pydantic.Field(FitParametersBase(),
                                                       description="Model fitting parameters")

    predict_parameters: PredictParametersBase = pydantic.Field(
        PredictParametersBase(), description="Prediction method parameters")

    hyper_parameters: HyperParametersBase = pydantic.Field(
        HyperParametersBase(), description="Hyper parameters")

    var_features: typing.List[str] = pydantic.Field(
        default=[], description="List of features variables")

    var_targets: typing.List[str] = pydantic.Field(
        default=[], description="List of target variables")

    var_extra: typing.List[str] = pydantic.Field(
        default=[], description="List of extra variables not used in the ML process but in pre or post processing")

    var_discretizer: Discretizer = pydantic.Field(
        None, description="Variable discretization specifications")

    # pdf_discretizer: Discretizer = pydantic.Field(
    #     None, description="Variable discretization specifications")

    model: typing.Any = pydantic.Field(
        None, description="Model storage structure")

    nb_data_fit: int = pydantic.Field(
        0, description="Number of data used to fit the model")

    metadata: ModelMetaData = pydantic.Field(default={},
                                             description="Model metadata")

    def __str__(self):

        return "\n\n".join([str(attr) + ": " + str(val) for attr, val in self.__dict__.items()])

    def init_from_dataframe(self, df):
        init_from_dataframe = getattr(self.model, "init_from_dataframe", None)
        if callable(init_from_dataframe):
            init_from_dataframe(df)

    def prepare_fit_data(self, data, logger=None, **kwds):
        """ Data preparation method. This method
        aims to be overloaded if needed"""

        if not(self.var_discretizer is None):
            data = self.var_discretizer.discretize(data, logger=logger, **kwds)

        return data

    def fit_specs(self, data, logger=None, **kwds):
        """ This is the specific fitting method for each Model. This method
        aims to be overloaded if needed"""
        self.model.fit(data, **self.fit_parameters.dict(),
                       logger=logger, **kwds)

    def fit(self, data, logger=None, **kwds):
        data_fit = self.prepare_fit_data(data, logger=logger, **kwds)
        self.fit_specs(data_fit, logger=logger, **kwds)

        self.nb_data_fit = len(data_fit)

        return data_fit

    def predict_specs(self, data, logger=None, **kwds):
        """ This is the specific prediction method for each Model. This method
        aims to be overloaded if needed"""
        return self.model.predict(data[self.var_features], self.var_targets,
                                  logger=logger, **kwds)

    def prepare_predict_data(self, data, logger=None, **kwds):
        """ Data preparation method. This method
        aims to be overloaded if needed"""
        if not(self.var_discretizer is None):
            data = self.var_discretizer.discretize(data, logger=logger, **kwds)

        return data

    def predict(self, data, logger=None, **kwds):

        # Check if some predict_parameters are overloaded in kwds
        var_discrete_support = kwds.get("var_discrete_support")
        if var_discrete_support:
            self.predict_parameters.var_discrete_support.update(
                **var_discrete_support)

        predict_postprocess = kwds.get("predict_postprocess")
        if predict_postprocess:
            self.predict_parameters.predict_postprocess.update(
                **predict_postprocess)

        data_predict = self.prepare_predict_data(data, logger=logger, **kwds)

        predictions = self.predict_specs(data_predict, logger=logger, **kwds)

        # Add DD variable name if None
        for tv, pred in predictions.items():
            if not(pred["scores"].variable.name):
                pred["scores"].variable.name = tv

        # Check special predict mode
        for var, predic_postproc in self.predict_parameters.predict_postprocess.items():
            if var in predictions.keys():

                # Conditioning var > var_condition
                if predic_postproc.get("var_condition_gt", None):
                    var_condition = predic_postproc.get(
                        "var_condition_gt", None)

                    # # ALERT: HUGE BOTTLENECK HERE !
                    # # TODO: FIND A WAY TO OPTIMIZE THIS !
                    scores_cond_df = \
                        predictions[var]["scores"].condition_gt(
                            data_predict[var_condition])

                    # scores_df = predictions[var]["scores"].copy(deep=True)
                    # scores_df.index = data_predict[var_condition].fillna(
                    #     method="bfill")
                    # scores_df.columns = scores_df.columns.astype(str)

                    # # ipdb.set_trace()

                    # def apply_condition_gt(dist):
                    #     cond_value = dist.name
                    #     dist_cond_idx = dist.index.get_loc(cond_value)

                    #     dist_shifted = dist.shift(-dist_cond_idx).fillna(0)
                    #     if 'inf' in dist.index[-1]:
                    #         # Deal with the case of the upport bound is an open interval
                    #         nb_val_p_inf = dist_cond_idx + 1
                    #         dist_shifted.iloc[-nb_val_p_inf:] = \
                    #             dist.iloc[-1]
                    #     dist_cond = dist_shifted/dist_shifted.sum()
                    #     return dist_cond
                    #     # return dist_cond.fillna(0)

                    # # ALERT: HUGE BOTTLENECK HERE !
                    # # TODO: FIND A WAY TO OPTIMIZE THIS !
                    # scores_cond_df = scores_df.apply(
                    #     apply_condition_gt, axis=1)

                    predictions[var]["scores"].values[:] = \
                        scores_cond_df.values[:]

                # Smoothing
                if predic_postproc.get("smoothing", None):
                    scores_df = predictions[var]["scores"].copy(deep=True)

                    smoothing = predic_postproc["smoothing"]
                    smoothing_param = smoothing.pop("mode_params", {})
                    scores_smoothed_df = \
                        scores_df.rolling(axis=1, min_periods=0, center=True,
                                          **smoothing)\
                        .mean(**smoothing_param)

                    # ipdb.set_trace()

                    predictions[var]["scores"].values[:] = \
                        scores_smoothed_df.values[:]

        return predictions

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
            # ipdb.set_trace()
            ddist.values[:] = 1/len(var_domain)
            pred_res.setdefault(tv, {"scores": ddist})

        return pred_res


class ModelException(Exception):
    """ Exception type used to raise exceptions within Model derived classes """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
