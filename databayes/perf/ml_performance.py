# -*- coding: utf-8 -*-

import typing

import tqdm
import pydantic
from ..modelling.MLModel import MLModel
from ..modelling.DiscreteDistribution import DiscreteDistribution
from .performance_measure import PerformanceMeasureBase
from ..utils import get_subclasses

import pandas as pd
import numpy as np

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}

if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


class MLPerformanceFitParameters(pydantic.BaseModel):

    is_test_pct: bool = pydantic.Field(
        True, description="Considers data test percentage if True and number else", show_edit=False)
    is_train_pct: bool = pydantic.Field(
        True, description="Considers data train percentage if True and number else", show_edit=False)
    percentage_training_data: float = pydantic.Field(
        0.75, description="Percentage of data in the training set", ge=0, le=1, multiple_of=0.05, percentage=True)
    training_sliding_window_size: float = pydantic.Field(
        1, description="Size of the training window", show_edit=False)
    testing_sliding_window_size: float = pydantic.Field(
        1, description="Size of the testing window", show_edit=False)
    group_by: typing.List[str] = pydantic.Field(
        [], description="Trajectory variables", title="Trajectory variables", from_data_columns=True)

    @pydantic.validator('testing_sliding_window_size')
    def is_test_pct_test(cls, v, values):
        if values['is_test_pct'] and v > 1:
            raise ValueError("Must be a percentage")
        if not(values['is_test_pct']) and (v < 1 or int(v) != v):
            raise ValueError("Must be a number")
        return v

    @pydantic.validator('training_sliding_window_size')
    def is_train_pct_test(cls, v, values):
        if values['is_train_pct'] and v > 1:
            raise ValueError("Must be a percentage")
        if not(values['is_train_pct']) and (v < 1 or int(v) != v):
            raise ValueError("Must be a number")
        return v


class MLPerformance(pydantic.BaseModel):
    # TODO:
    # It may be interesting (for plotting for example) to add data_test and pred_prob as class attributes
    # with options to export it in the json or dict methods
    model: MLModel = pydantic.Field(...,
                                    description="Machine learning model")
    measures: typing.Dict[str, PerformanceMeasureBase] = pydantic.Field(
        {}, description="Dictionary of performance measures")

    fit_parameters: MLPerformanceFitParameters = pydantic.Field(
        MLPerformanceFitParameters(), description="Fitting hyper parameters")

    data_test_index: typing.Any = pydantic.Field(
        None, description="Internal attribute to store data test indexes")

    # data_test_group_index: dict = pydantic.Field(
    #     {}, description="Internal attribute to store data test group indexes if needed")

    data_test: pd.DataFrame = pydantic.Field(
        pd.DataFrame(), description="Data test")

    # Dict of DiscreteDistribution
    pred_prob: dict = pydantic.Field(
        {}, description="Data prediction probability")

    @pydantic.root_validator(pre=True)
    def cls_validator(cls, obj):
        # Validate measures
        measure_classes_d = {cls.__name__: cls
                             for cls in get_subclasses(PerformanceMeasureBase)}

        for measure_name, measure_specs in obj["measures"].items():

            if any([isinstance(measure_specs, mcls)
                    for mcls in measure_classes_d.values()]):
                continue

            obj["measures"][measure_name] = PerformanceMeasureBase.from_dict(
                **measure_specs)

        return obj

    class Config:
        arbitrary_types_allowed = True

    def json(self, exclude=None, **kwargs):
        return super().json(exclude={"data_test", "pred_prob"}, **kwargs)

    def dict(self, exclude=None, **kwargs):
        return super().dict(exclude={"data_test", "pred_prob"}, **kwargs)

    def measures_approx_equal(self, other, **kwargs):

        if set(self.measures.keys()) != set(other.measures.keys()):
            return False

        for m_key in self.measures.keys():
            m_self = self.measures[m_key]
            m_other = other.measures[m_key]
            if not(m_self.approx_equal(m_other)):
                return False

        return True

    def prepare_data(self, data):

        data_prepared = data.copy(deep=True)

        for var in self.model.var_targets:
            if not(data_prepared[var].dtypes.name == "category"):
                data_prepared[var] = \
                    data_prepared[var].astype("category")
                # data_prepared[var].astype(str)\
                #                   .astype("category")

        return data_prepared

    def split_data(self, data, **kwargs):
        """Return splited data: percentage_training_data is
        the percentage of data in the training set."""
        percent_train = self.fit_parameters.percentage_training_data

        if self.fit_parameters.group_by == []:
            data_train_idx = data.index[:int(
                percent_train * len(data))].to_list()
            data_test_idx = data.index[int(
                percent_train * len(data)):].to_list()

            self.data_test_index = data_test_idx
        else:
            data_grp = data.groupby(self.fit_parameters.group_by)
            group_list = list(data_grp.indices.keys())
            data_train_idx = group_list[:int(percent_train * len(group_list))]
            data_test_idx = group_list[int(percent_train * len(group_list)):]
            index_name = data.index.name if not(data.index.name is None) \
                else "index"
            data_index_grp_df = data.reset_index().set_index(self.fit_parameters.group_by)
            data_test = data_index_grp_df.loc[data_test_idx]\
                                         .reset_index().set_index(index_name)
            # data_index_grp_df = data.set_index(self.fit_parameters.group_by)
            # data_test = data_index_grp_df.loc[data_test_idx].reset_index()

            self.data_test_index = data_test.index

        return data_train_idx, data_test_idx

    def sliding_split(self, data_train_idx, data_test_idx,
                      progress_mode=False, **kwargs):
        """Generator returning step by step the last training_sliding_window_size indexes of the training set
        and the first testing_sliding_window_size indexes of the testing set

        Keywords arguments:
        data_train_idx -- indexes of the data train set
        data_test_idx -- indexes of the data test set
        """

        if self.fit_parameters.is_train_pct:
            train_idx = data_train_idx[int(
                (1-self.fit_parameters.training_sliding_window_size)*len(data_train_idx)):]
        else:
            nb_data_train = int(
                len(data_train_idx) - self.fit_parameters.training_sliding_window_size)
            # if training_sliding_window_size > len(data_train), we want to take all datas of data_train
            if nb_data_train < 0:
                nb_data_train = 0
            train_idx = data_train_idx[nb_data_train:]

        if self.fit_parameters.is_test_pct:
            length_test_idx = int(
                self.fit_parameters.testing_sliding_window_size*len(data_test_idx))
            nb_data_test = int(
                1/self.fit_parameters.testing_sliding_window_size)
        else:
            length_test_idx = int(
                self.fit_parameters.testing_sliding_window_size)
            nb_data_test = int(len(data_test_idx) /
                               self.fit_parameters.testing_sliding_window_size)

        for idx_split in tqdm.tqdm(range(nb_data_test+1),
                                   disable=not(progress_mode),
                                   desc="Sliding prediction process"):
            test_idx = data_test_idx[idx_split *
                                     length_test_idx:(idx_split+1)*length_test_idx]
            if (len(test_idx) == 0) or (len(train_idx) == 0):
                continue
            yield train_idx, test_idx
            train_idx = train_idx[length_test_idx:] + test_idx

    def fit_and_predict_slide(self, data_df, data_train_idx, data_test_idx,
                              logger=None,
                              progress_mode=False,
                              **kwargs):
        """Return pred_prob : Probability of predictions for each data in data_test using sliding data_train to fit the model"""

        # Initialize the results array

        pred_prob = dict()
        for tv in self.model.var_targets:
            # tv_disc_specs = self.model.predict_parameters.var_discrete_support\
            #                                           .get(tv, {})
            tv_disc = self.model.var_discretizer.variables.get(tv)
            if tv_disc:
                tv_disc_specs = {"bins": tv_disc.get_bins()}

            if not(tv_disc_specs):
                tv_dtype = data_df[tv].dtypes.name
                if tv_dtype == "category":
                    target_labels = data_df[tv].cat.categories.tolist()
                elif tv_dtype == "object":
                    target_labels = data_df[tv].unique().tolist()
                else:
                    raise ValueError(
                        f"A discrete domain must be specified for "
                        f"Target variable {tv} of type {tv_dtype}")

                tv_disc_specs = {"domain": target_labels}

            tv_dd = DiscreteDistribution(name=tv,
                                         index=self.data_test_index,
                                         **tv_disc_specs)
            pred_prob[tv] = {"scores": tv_dd}

        data_train_test_slide = self.sliding_split(
            data_train_idx, data_test_idx, progress_mode=progress_mode)

        # for train_idx, test_idx in tqdm.tqdm(data_train_test_slide,
        #                                      disable=not(progress_mode),
        #                                      desc="Sliding prediction process"):

        for train_idx, test_idx in data_train_test_slide:

            if self.fit_parameters.group_by != []:
                index_name = data_df.index.name \
                    if not(data_df.index.name is None) \
                    else "index"

                data_index_grp_df = \
                    data_df.reset_index().set_index(self.fit_parameters.group_by)

                d_train = \
                    data_index_grp_df.loc[train_idx].reset_index()\
                                                    .set_index(index_name)
                d_test = \
                    data_index_grp_df.loc[test_idx].reset_index()\
                                                   .set_index(index_name)

            else:
                d_train = data_df.loc[train_idx]
                d_test = data_df.loc[test_idx]

            d_train = d_train[self.model.var_features +
                              self.model.var_targets + self.model.var_extra]
            d_test = d_test[self.model.var_features + self.model.var_extra]

            self.model.fit(d_train)

            pred_res = self.model.predict(d_test,
                                          logger=logger,
                                          progress_mode=progress_mode,
                                          **kwargs)

            # REMINDER: pred_res[tv]["scores"] is a DiscreteDistribution
            for tv in self.model.var_targets:
                pred_prob[tv]["scores"].loc[d_test.index, :] = \
                    pred_res[tv]["scores"].loc[:]
                # WARNING: DiscreteDistribution from MLP has not the correct domain
                # => NaNs can be produced => replacing them with 0s
                pred_prob[tv]["scores"].values[:] = \
                    np.nan_to_num(pred_prob[tv]["scores"].values)

        return pred_prob

    def run(self, data_df, logger=None, progress_mode=False, **kwargs):
        """Returns a dict : - keys : different model's evaluation
                            - values : dict representing the result of the related eval method"""

        if not(logger is None):
            logger.info("Start performance analysis")

        # TODO: data preparation should be done by model class
        # TOREMOVE
        #data_df = self.prepare_data(data_df)
        data_train_idx, data_test_idx = self.split_data(data_df)

        if not(logger is None):
            logger.info("Compute predictions")
        self.pred_prob = \
            self.fit_and_predict_slide(data_df,
                                       data_train_idx,
                                       data_test_idx,
                                       logger=logger,
                                       progress_mode=progress_mode)

        if not(logger is None):
            logger.info("Compute performance measures")

        self.data_test = \
            data_df.loc[self.data_test_index,
                        self.fit_parameters.group_by +
                        self.model.var_features +
                        self.model.var_targets +
                        self.model.var_extra]

        for pm_name, performance_measure \
            in tqdm.tqdm(self.measures.items(),
                         disable=not(progress_mode),
                         desc="Performance evaluation"):
            performance_measure.group_by = self.fit_parameters.group_by
            performance_measure.pred_prob = self.pred_prob
            performance_measure.data_test = self.data_test
            performance_measure.evaluate()
            #performance_measure.evaluate(self.data_test, self.pred_prob)

        return self.measures
