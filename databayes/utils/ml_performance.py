# -*- coding: utf-8 -*-

import typing

import tqdm
import pydantic
from ..modelling.MLModel import MLModel
from ..modelling.DiscreteDistribution import DiscreteDistribution
from .performance_measure import PerformanceMeasureBase

import pandas as pd

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}

if "dash" in installed_pkg:
    import dash
    import dash_bootstrap_components as dbc
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output

if "plotly" in installed_pkg:
    import plotly.graph_objects as go

if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


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
    model: MLModel = pydantic.Field(...,
                                    description="Machine learning model")
    measures: typing.Dict[str, PerformanceMeasureBase] = pydantic.Field(
        {}, description="Dictionary of performance measures")

    fit_parameters: FitParameters = pydantic.Field(
        FitParameters(), description="Fitting Parameters")

    group_by: typing.List[str] = pydantic.Field(
        [], description="Group by attributes")

    data_test_index: typing.Any = pydantic.Field(
        None, description="Internal attribute to store data test indexes")

    # data_test_group_index: dict = pydantic.Field(
    #     {}, description="Internal attribute to store data test group indexes if needed")

    data_test: pd.DataFrame = pydantic.Field(
        pd.DataFrame(), description="Data test")

    # Dict of DiscreteDistribution
    pred_prob: dict = pydantic.Field(
        {}, description="Data prediction probability")

    @pydantic.validator('measures', pre=True)
    def match_dict_attribut(cls, measures):

        measure_classes_d = {cls.__name__: cls
                             for cls in PerformanceMeasureBase.__subclasses__()}

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

            new_measure = \
                measure_classes_d.get(measure_class_name)(**measure_specs)

            if len(new_measure.name) == 0:
                new_measure.name = measure_name
            measures[measure_name] = new_measure

        return measures

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

        if self.group_by == []:
            data_train_idx = data.index[:int(
                percent_train * len(data))].to_list()
            data_test_idx = data.index[int(
                percent_train * len(data)):].to_list()

            self.data_test_index = data_test_idx
        else:
            data_grp = data.groupby(self.group_by)
            group_list = list(data_grp.indices.keys())
            data_train_idx = group_list[:int(percent_train * len(group_list))]
            data_test_idx = group_list[int(percent_train * len(group_list)):]
            index_name = data.index.name if not(data.index.name is None) \
                else "index"
            data_index_grp_df = data.reset_index().set_index(self.group_by)
            data_test = data_index_grp_df.loc[data_test_idx]\
                                         .reset_index().set_index(index_name)
            # data_index_grp_df = data.set_index(self.group_by)
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
            tv_support = self.model.predict_parameters.var_discrete_support\
                                                      .get(tv, {})

            # ipdb.set_trace()
            if not(tv_support):
                tv_dtype = data_df[tv].dtypes.name
                if tv_dtype == "category":
                    target_labels = data_df[tv].cat.categories.tolist()
                elif tv_dtype == "object":
                    target_labels = data_df[tv].unique().tolist()
                else:
                    raise ValueError(
                        f"A discrete domain must be specified for "
                        f"Target variable {tv} of type {tv_dtype}")

                tv_support = {"domain": target_labels}

            tv_dd = DiscreteDistribution(name=tv,
                                         index=self.data_test_index,
                                         **tv_support)
            pred_prob[tv] = {"scores": tv_dd}

        data_train_test_slide = self.sliding_split(
            data_train_idx, data_test_idx, progress_mode=progress_mode)

        # for train_idx, test_idx in tqdm.tqdm(data_train_test_slide,
        #                                      disable=not(progress_mode),
        #                                      desc="Sliding prediction process"):

        for train_idx, test_idx in data_train_test_slide:

            if self.group_by != []:
                index_name = data_df.index.name \
                    if not(data_df.index.name is None) \
                    else "index"

                data_index_grp_df = \
                    data_df.reset_index().set_index(self.group_by)

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
                        self.group_by +
                        self.model.var_features +
                        self.model.var_targets +
                        self.model.var_extra]

        for pm_name, performance_measure \
            in tqdm.tqdm(self.measures.items(),
                         disable=not(progress_mode),
                         desc="Performance evaluation"):
            performance_measure.group_by = self.group_by
            performance_measure.evaluate(self.data_test, self.pred_prob)

        return self.measures


class MLPerformanceDashboard(MLPerformance):

    dash_title: str = pydantic.Field("Dashboard",
                                     description="Dashboard title")
    dash_version: str = pydantic.Field(None,
                                       description="Dashboard version")

    dash_app: typing.Any = pydantic.Field(
        None,
        description="Dashboard backend")

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        # # Default dashboard parameters
        # default_parameters_filename = \
        #     os.path.join(os.path.dirname(__file__),
        #                  "XXX.yaml")
        # with open(default_parameters_filename, 'r', encoding="utf-8") \
        #         as yaml_file:
        #     try:
        #         self.dashboard = yaml.load(yaml_file,
        #                                    Loader=yaml.SafeLoader)
        #     except yaml.YAMLError as exc:
        #         logging.error(exc)

        if self.dash_app is None:
            self.dash_app = dash.Dash(__name__,
                                      external_stylesheets=[dbc.themes.BOOTSTRAP])

    def get_title_layout(self):

        layout = html.Div([
            html.H1(children=f'{self.dash_title}'),
        ])

        return layout

    def get_tabs_layout(self):

        layout = html.Div([
            dcc.Tabs(id='tabs-content',
                     children=[self.create_tab_predict_result()] +
                     [
                         dcc.Tab(label=f'{performance_measure.name}',
                                 children=performance_measure.get_dash_layout(self.dash_app))
                         for pm_name, performance_measure in self.measures.items()
                     ]
                     )
        ])

        return layout

    def get_app_layout(self):

        layout = html.Div(
            [
                self.get_title_layout(),
                self.get_tabs_layout(),
            ])

        return layout

    def create_tab_predict_result(self, **tab_specs):
        tab_id = "predict_result"

        # Init figures
        # graph_list = []
        # for value_name in tab_specs.get("values").keys():
        #     graph_list.append(dcc.Graph(
        #         figure=self.create_rfa_figure(value_name, **tab_specs),
        #         id=f"{tab_id}_graph_{value_name}")
        #     )
        tab = dcc.Tab(id=tab_id,
                      label="Predictions",
                      children=dbc.Row([
                          dbc.Col(self.tab_predict_result_ctrl(
                              id=f"{tab_id}_ctrl",
                              **tab_specs), width=2),
                          dbc.Col(self.tab_predict_result_graphs(
                              id=f"{tab_id}_graphs", **tab_specs)),
                      ]))

        # hidden signal value for cache management
        hidden_update_div = \
            html.Div(id=tab_id + '_update_signal', style={'display': 'none'})

        # # Create cache
        # self.cache["rfa"] = dict(
        #     data_sel_df=pd.DataFrame()
        # )

        # Create callbacks
        self.tab_predict_result_cb(**tab_specs)

        return html.Div([tab, hidden_update_div])

    def tab_predict_result_cb(self,
                              **params):
        tab_id = "predict_result"

        inputs = [Input(f'{tab_id}_ctrl_var_target', 'value')]
        if len(self.group_by) >= 1:
            inputs.append(Input(f'{tab_id}_ctrl_group', 'value'))

        @self.dash_app.callback(
            [
                Output(f'{tab_id}_graphs_dd_frames', 'figure'),
                Output(f'{tab_id}_graphs_dd_all', 'figure')
            ],
            inputs)
        def cb_fun(var_target, group=None):
            index_filter = None
            data_index = None
            if not(group is None):
                data_test_gb = self.data_test.groupby(self.group_by)
                index_filter = list(data_test_gb.groups.values())[group]

            # ipdb.set_trace()
            if not(self.model.metadata.predict_index is None):
                data_index = self.data_test[self.model.metadata.predict_index]

            fig_dd_frames = \
                self.pred_prob[var_target]["scores"]\
                    .get_plotly_dd_frames_specs(index_filter=index_filter,
                                                data_index=data_index)
            fig_dd_all = \
                self.pred_prob[var_target]["scores"]\
                    .get_plotly_dd_all_specs(index_filter=index_filter,
                                             data_index=data_index)

            # return None, go.Figure(fig_dd_all)
            return go.Figure(fig_dd_frames), go.Figure(fig_dd_all)

        # return cb_fun

    def tab_predict_result_ctrl(self,
                                id="predict_result_ctrl",
                                **params):

        control_layout = []

        # if len(self.group_by) >= 2:
        #     raise ValueError(
        #         "Only variable grouping of size 1 is supported for now")

        if len(self.group_by) >= 1:
            data_test_gb = self.data_test.groupby(self.group_by)
            data_test_groups_idx = list(data_test_gb.groups.keys())
            if len(self.group_by) == 1:
                data_test_groups_idx = [(idx,) for idx in data_test_groups_idx]

            data_test_groups_labels = \
                {i: ", ".join([f"{gvar}={gidx_val}"
                               for gvar, gidx_val in zip(self.group_by, gidx)])
                 for i, gidx in enumerate(data_test_groups_idx)}

            control_layout.append(
                dbc.Row([
                    dbc.Col(
                        width=1),
                    dbc.Col(
                        html.Label("Group"))
                ]))
            control_layout.append(
                dbc.Row([
                    dbc.Col(
                        width=1),
                    dbc.Col(
                        dcc.Dropdown(
                            value=0,
                            options=[{'label': gval, 'value': i}
                                     for i, gval in data_test_groups_labels.items()],
                            id=id + "_group"))
                ]))

        var_target = list(self.pred_prob.keys())
        control_layout.append(
            dbc.Row([
                dbc.Col(
                    width=1),
                dbc.Col(
                    html.Label("Target variable"))
            ]))
        control_layout.append(
            dbc.Row([
                dbc.Col(
                    width=1),
                dbc.Col(
                    dcc.Dropdown(
                        value=var_target[0],
                        options=[{'label': val, 'value': val}
                                 for val in var_target],
                        id=id + "_var_target"))
            ]))

        return control_layout

    def tab_predict_result_graphs(self,
                                  id="predict_result_graphs",
                                  **params):

        layout = []

        layout.append(
            dbc.Col([
                dcc.Graph(id=id + "_dd_frames"),
                dcc.Graph(id=id + "_dd_all")
            ]))

        return layout

    def run_app(self, data,
                logger=None,
                progress_mode=False,
                **kwargs):

        self.run(data,
                 logger=logger,
                 progress_mode=progress_mode)

        self.dash_app.layout = self.get_app_layout()

        self.dash_app.run_server(**kwargs)
