# -*- coding: utf-8 -*-

import typing

import databayes.modelling.core as dmc
import numpy as np
import pandas as pd
import tqdm
import pydantic
import databayes.modelling.DiscreteDistribution as dd
import databayes.utils.performance_measure as pm

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class FitParameters(pydantic.BaseModel):

    is_test_pct: bool = pydantic.Field(
        True, description="Considers data test percentage if True and number else")
    is_train_pct: bool = pydantic.Field(
        True, description="Considers data train percentage if True and number else")
    percentage_training_data: float = pydantic.Field(
        0.75, description="Percentage of data in the training set")
    training_sliding_window_size: float = pydantic.Field(
        1, description="Size of the training window")
    testing_sliding_window_size: float = pydantic.Field(
        1, description="Size of the testing window")

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

    model: dmc.MLModel = pydantic.Field(...,
                                        description="Machine learning model")
    measures: typing.Dict[str, pm.PerformanceMeasureBase] = pydantic.Field(
        {}, description="Dictionary of performance measures")

    fit_parameters: FitParameters = pydantic.Field(
        FitParameters(), description="Fitting Parameters")

    group_by: typing.List[str] = pydantic.Field([], description="Group by attributes")

    data_test_index: typing.Any = pydantic.Field(None, description="Internal attribute to store data test indexes")

    @pydantic.validator('measures', pre=True)
    def match_dict_attribut(cls, measures):

        measure_classes_d = {cls.__name__: cls
                             for cls in pm.PerformanceMeasureBase.__subclasses__()}

        for measure_name, measure_specs in measures.items():
            measure_class_name = \
                "".join([word.capitalize()
                         for word in measure_name.split("_")])

            if not(measure_class_name.endswith("Measure")):
                measure_class_name += "Measure"

            if not(measure_class_name in measure_classes_d.keys()):
                raise ValueError(
                    f"Measure {measure_name} (or class {measure_class_name}) not supported")
            if measure_specs is None:
                measure_specs = {}
            measures[measure_name] = \
                measure_classes_d.get(measure_class_name)(**measure_specs)

        return measures

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
        """Return splited data: percentage_training_data is the percentage of data in the training set."""

        percent_train = self.fit_parameters.percentage_training_data
        
        if self.group_by == []:
            data_train_idx = data.index[:int(percent_train * len(data))].to_list()
            data_test_idx = data.index[int(percent_train * len(data)):].to_list()
            
            self.data_test_index = data_test_idx
        else:
            data_grp=data.groupby(self.group_by)
            group_list = list(data_grp.indices.keys())
            data_train_idx = group_list[:int(percent_train * len(group_list))]
            data_test_idx = group_list[int(percent_train * len(group_list)):]

            #TODO: mettre en cohérence les index après group by
            #Le code ci-dessous est censé marcher mais un bug apparait, à creuser ..
            data_index_grp_df = data.reset_index().set_index(self.group_by)
            data_test = data_index_grp_df.loc[data_test_idx].reset_index().set_index("index")
            # data_index_grp_df = data.set_index(self.group_by)
            # data_test = data_index_grp_df.loc[data_test_idx].reset_index()

            self.data_test_index = data_test.index
            
        return data_train_idx, data_test_idx


    def sliding_split(self, data_train_idx, data_test_idx, **kwargs):
        """Generator returning step by step the last training_sliding_window_size indexes of the training set
        and the first testing_sliding_window_size indexes of the testing set

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
            length_test_idx = int(self.fit_parameters.testing_sliding_window_size*len(data_test_idx))
            nb_data_test = int(1/self.fit_parameters.testing_sliding_window_size)
        else:
            length_test_idx = int(self.fit_parameters.testing_sliding_window_size)
            nb_data_test = int(len(data_test_idx)/self.fit_parameters.testing_sliding_window_size)

        for idx_split in range(nb_data_test+1):
            test_idx = data_test_idx[idx_split*length_test_idx:(idx_split+1)*length_test_idx]
            if (len(test_idx) == 0) or (len(train_idx) == 0):
                continue
            yield train_idx, test_idx
            train_idx =  train_idx[length_test_idx:] + test_idx
    

    def fit_and_predict_slide(self, data_df, data_train_idx, data_test_idx,
                              calculate_map=False,
                              logger=None,
                              progress_mode=False,
                              **kwargs):
        """Return pred_prob : Probability of predictions for each data in data_test using sliding data_train to fit the model"""

        # Initialize the results array

        pred_prob = dict()
        for tv in self.model.var_targets:
            pred_prob[tv] = dict()
            target_labels = data_df.loc[self.data_test_index, tv].cat.categories.to_list()
            
            pred_prob[tv]["scores"] = dd.DiscreteDistribution(domain=target_labels,
                                                              index=self.data_test_index)
            
        data_train_test_slide = self.sliding_split(data_train_idx, data_test_idx)

        for train_idx, test_idx in tqdm.tqdm(data_train_test_slide,
                                         disable=not(progress_mode),
                                         desc="Sliding prediction process"):

            if self.group_by != []:
                data_index_grp_df = data_df.reset_index().set_index(self.group_by)
                d_train = data_index_grp_df.loc[train_idx].reset_index().set_index("index")
                d_test = data_index_grp_df.loc[test_idx].reset_index().set_index("index")
            else:
                d_train = data_df.loc[train_idx]
                d_test = data_df.loc[test_idx]
            
            d_train = d_train[self.model.var_features + self.model.var_targets]
            d_test = d_test[self.model.var_features]

            self.model.fit(d_train)
            
            pred_res = self.model.predict(d_test,
                                          logger=logger,
                                          progress_mode=progress_mode,
                                          **kwargs)
            
            # pred_res[tv]["scores"] va renvoyer un DiscreteDistribution
            for tv in self.model.var_targets:
                pred_prob[tv]["scores"].loc[d_test.index,
                                            :] = pred_res[tv]["scores"].loc[:]

        return pred_prob

    def run(self, data, logger=None, progress_mode=False):
        """Returns a dict : - keys : different model's evaluation
                            - values : dict representing the result of the related eval method"""

        if not(logger is None):
            logger.info("Start performance analysis")

        data_prepared_df = self.prepare_data(data)
        data_train_idx, data_test_idx = self.split_data(data_prepared_df)

        if not(logger is None):
            logger.info("Compute predictions")
        pred_prob = self.fit_and_predict_slide(data_prepared_df, data_train_idx, data_test_idx,
                                               logger=logger,
                                               progress_mode=progress_mode)
        
        if not(logger is None):
            logger.info("Compute performance measures")

        data_test = data_prepared_df.loc[self.data_test_index, self.group_by + self.model.var_features + self.model.var_targets]

        for pm_name, performance_measure \
            in tqdm.tqdm(self.measures.items(),
                         disable=not(progress_mode),
                         desc="Performance evaluation"):
            performance_measure.group_by = self.group_by
            performance_measure.evaluate(data_test, pred_prob)

        return self.measures
        



    ###############################################
    #                                             #
    #               VISUALISATIONS                #
    #                                             #
    ###############################################

    def get_dash_layout(self, app):

        layout = html.Div([
            dcc.Tabs(id='tabs-content', value=list(self.measures.keys())[0], 
                children=[
                dcc.Tab(label=f'{pm_name}', value=f'{pm_name}', children=performance_measure.get_dash_layout(app))
                    for pm_name, performance_measure in self.measures.items()
                ]
            )
        ])
        
        return layout

    
    def run_app(self, data, **kwargs):

        self.run(data, progress_mode=True)

        app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
        server = app.server
        
        app.layout = self.get_dash_layout(app)
        
        app.run_server(**kwargs)