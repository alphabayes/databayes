import abc
import typing

from itertools import chain, combinations
import pandas as pd
import numpy as np
import pydantic
import textwrap

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


class PerformanceMeasureBase(pydantic.BaseModel, abc.ABC):

    name: str = pydantic.Field(
        "", description="Measure label")
    variables: list = pydantic.Field(
        [], description="Variables considered by the measure")
    result: typing.Any = pydantic.Field(None, description="Measurement result")
    group_by: typing.List[str] = pydantic.Field(
        [], description="Group by arguments")

    def evaluate(self, data_test, pred_prob):
        pass

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

    def get_dash_layout(self, app):
        pass

    def run_app(self):
        app = dash.Dash(__name__, suppress_callback_exceptions=True,
                        external_stylesheets=[dbc.themes.BOOTSTRAP])
        server = app.server
        app.layout = self.get_dash_layout(app)
        app.run_server(debug=True)


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

    def plotly_specs(self):
        cm = {}
        fig_specs = {}

        for tv in self.variables:
            cm_df = pd.DataFrame.from_records(self.result[tv])

            cm_df.index = cm_df.columns
            cm[tv] = cm_df

            fig_specs[tv] = {'data': [{'x': cm_df.index.to_list(),
                                       'y': cm_df.columns.to_list(),
                                       'z': cm_df.values.tolist(),
                                       'xgap': 3,
                                       'ygap': 3,
                                       'type': 'heatmap',
                                       'colorscale': 'Greens',
                                       'colorbar': {
                'x': 1.5
            }
                # 'hoverinfo': 'text',
                # 'hovertext': [f'Prediction: {y}<br>Exact label: {x}<br><br><b>Number = {cm_df.loc[x,y]}<b>'
                #                 for x in cm_df.index for y in cm_df.columns]
            }
            ],
                'layout': {
                'template': 'plotly_white',
                'xaxis': {
                            'title': {
                                'text': "Exact predictions",
                                'font': {
                                        'family': "sans-serif",
                                        'size': 8,
                                        'color': "black"}
                            },
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True
                },
                'yaxis': {
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True,
                    'title': {
                        'text': 'Network prediction',
                                'font': {
                                    'family': "sans-serif",
                                    'size': 8,
                                    'color': "black"}
                    }
                }

            }
            }

        return fig_specs

    def get_dash_layout(self, app):

        fig_specs = self.plotly_specs()

        layout = html.Div([
            dcc.Tabs(id='tabs-confusion-matrix-content', value=self.variables[0],
                     children=[
                dcc.Tab(label=tv, value=tv, children=dcc.Graph(figure=go.Figure(fig_specs[tv]))) for tv in self.variables
            ]
            ),
            html.Div(id='confusion-matrix-content')
        ])

        return layout


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

    data_test: pd.DataFrame = pydantic.Field(
        pd.DataFrame(), description="Data test")

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

    ###############################################
    #                                             #
    #         SUCCESS  VISUALISATIONS             #
    #                                             #
    ###############################################

    def plotly_aggreg_specs(self):

        result = self.result_to_frame()

        fig_specs = {'data': [{'x': result["aggreg"].index.get_level_values('map_k').to_list(),
                               'y': result["aggreg"]["mean"].to_list(),
                               'type': 'bar',
                               'hoverinfo': 'text',
                               'hovertext': [f'K = {k}<br><br><b>Aggregation rate = {success} /{len(self.variables)}<b>'
                                             for k, success in zip(result["aggreg"].index.get_level_values('map_k'), result["aggreg"]["mean"])]
                               }
                              ],
                     'layout': {
            'template': 'plotly_white',
                        'xaxis_type': 'category',
                        'showlegend': False,
                        'title': {
                            'text': 'Aggreg',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {
                                'family': "sans-serif",
                                'size': 18,
                                'color': "black"
                            }
                        },
            'xaxis': {
                            'title': {
                                'text': "Number of most probable labels to be considered in accuracy computation",
                                'font': {
                                        'family': "sans-serif",
                                        'size': 12,
                                        'color': "black"}
                            },
                            'showline': True,
                            'linewidth': 1,
                            'linecolor': 'black',
                            'mirror': True
                        },
            'yaxis': {
                            'nticks': 20,
                            'showline': True,
                            'linewidth': 1,
                            'linecolor': 'black',
                            'mirror': True,
                            'title': {
                                'text': 'Mean success rate',
                                'font': {
                                        'family': "sans-serif",
                                        'size': 12,
                                        'color': "black"}
                            }
                        }

        }
        }
        return fig_specs

    def plotly_joint_specs(self):

        result = self.result_to_frame()

        fig_specs = {'data': [{'y': result["joint"].iloc[0, :].to_list(),
                               'x': result["joint"].columns.to_list(),
                               'type': 'bar',
                               'hoverinfo': 'text',
                               'hovertext': [f"K = {1}<br><br>Success of the <br>combination: {x} <br> if the other variables<br>aren't predicted<br><b>Success: {s}<b>"
                                             for s, x in zip(result["joint"].loc[1, :], result["joint"].columns)]
                               }],
                     'frames':
                     [
            {'data': {
                'y': result["joint"].loc[k, :].to_list(),
                'x': result["joint"].columns.to_list(),
                'type': 'bar',
                'hoverinfo': 'text',
                'hovertext': [f"K = {k}<br><br>Success of the <br>combination: {x} <br> if the other variables<br>aren't predicted<br><b>Success: {s}<b>"
                              for s, x in zip(result["joint"].loc[k, :], result["joint"].columns)]
            },
                'name': str(k)
            }
            for k in result["joint"].index.get_level_values('map_k').to_list()],
            'layout': {
            'template': 'plotly_white',
                        'sliders': [{
                            "active": 0,
                            "currentvalue": {
                                "font": {"size": 20},
                                "prefix": "K=",
                                "visible": True,
                                "xanchor": "right"
                            },
                            "transition": {"duration": 300, "easing": "cubic-in-out"},
                            "pad": {"b": 10, "t": 50},
                            "len": 0.1*len(result["joint"].index.get_level_values('map_k').to_list()),
                            "x": 0.5,
                            'xanchor': 'center',
                            "yanchor": "top",
                            "y": -0.2,
                            "steps": [
                                {"args": [
                                    [k],
                                    {"frame": {"duration": 300, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 300}}
                                ],
                                    "label": k,
                                    "method": "animate"}
                                for k in result["joint"].index.get_level_values('map_k').to_list()
                            ]
                        }],
            'title': {
                            'text': 'Joint',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {
                                'family': "sans-serif",
                                'size': 18,
                                'color': "black"
                            }
                        },
            'yaxis': {
                            'title': {
                                'text': "Number of most probable labels to be considered in accuracy computation",
                                'font': {
                                        'family': "sans-serif",
                                        'size': 12,
                                        'color': "black"}
                            },
                            'showline': True,
                            'linewidth': 1,
                            'linecolor': 'black',
                            'mirror': True
                        },
            'xaxis': {
                            'showline': True,
                            'linewidth': 1,
                            'linecolor': 'black',
                            'mirror': True,
                            'title': {
                                'text': 'Labels combinations',
                                'font': {
                                        'family': "sans-serif",
                                        'size': 12,
                                        'color': "black"}
                            }
                        }

        }
        }
        return fig_specs

    def plotly_indep_specs(self):

        result = self.result_to_frame()

        fig_specs = {'data': [{'x': result["indep"].index.get_level_values('map_k').to_list(),
                               'y': result["indep"][tv]["mean"].to_list(),
                               'type': 'bar',
                               'name': f"{tv}",
                               'hoverinfo': 'text',
                               'hovertext': [f'Target variable: {tv} <br>K = {k}<br><br><b>Sucess rate = {success}<b>'
                                             for k, success in zip(result["indep"].index.get_level_values('map_k'), result["indep"][tv]["mean"])]
                               }
                              # 'hovertemplate': f'Target variable: {tv} <br>K = {tv}<br><br><b>Sucess rate = {tv}<b>'}
                              for tv in self.variables],
                     'layout': {
            'template': 'plotly_white',
                        'xaxis_type': 'category',
                        'showlegend': True,
                        # 'legend_title_text': 'Target variables',
                        'legend_orientation': 'v',
                        'legend': {
                            'title': {
                                'text': 'Target variables',
                                'font': {
                                    'family': "sans-serif",
                                    'size': 12,
                                    'color': "black"},
                            },
                            'x': 1.02,
                            'y': 0.97,
                            'traceorder': "normal",
                            'font': {
                                'family': "sans-serif",
                                'size': 12,
                                'color': "black"},
                            'bgcolor': "white",
                            'bordercolor': "Black",
                            'borderwidth': 1},
                        'title': {
                            'text': 'Indep',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {
                                'family': "sans-serif",
                                'size': 18,
                                'color': "black"
                            }
                        },
            'xaxis': {
                            'title': {
                                'text': "Number of most probable labels to be considered in accuracy computation",
                                'font': {
                                        'family': "sans-serif",
                                        'size': 12,
                                        'color': "black"}
                            },
                            'showline': True,
                            'linewidth': 1,
                            'linecolor': 'black',
                            'mirror': True
                        },
            'yaxis': {
                            'nticks': 20,
                            'showline': True,
                            'linewidth': 1,
                            'linecolor': 'black',
                            'mirror': True,
                            'title': {
                                'text': 'Success rate',
                                'font': {
                                        'family': "sans-serif",
                                        'size': 12,
                                        'color': "black"}
                            }
                        }

        }
        }
        return fig_specs

    ######################################
    #  Affichage de la table des données #
    ######################################

    def get_success_table_layout(self, app, tv):

        pred_map_kmax = self.pred_prob[tv]["scores"].get_map(
            max(self.map_k))
        map_prob = - pd.DataFrame(
            np.sort(-self.pred_prob[tv]["scores"].values, axis=1)[:, :max(self.map_k)], index=pred_map_kmax.index)

        success_k = pd.DataFrame(np.zeros((len(self.data_test), max(self.map_k))), columns=[
                                 f'map_{k}' for k in range(1, max(self.map_k) + 1)], index=self.data_test.index)

        map_prob_length = min(max(self.map_k), len(pred_map_kmax.columns))
        map_prob.columns = [f'map_{k}' for k in range(1, map_prob_length + 1)]

        for k in range(1, max(self.map_k) + 1):
            success_k.loc[:, f'map_{k}'] = self.pred_success[k].loc[:, tv]

        success_k_length = len(success_k)
        success_first_k = pd.Series(np.array([max(self.map_k) + 1 for i in range(
            success_k_length)]), index=self.data_test.index) - success_k.sum(axis=1)
        success_first_k[success_first_k == max(self.map_k) + 1] = -1
        success_first_k = success_first_k.astype('str')

        d_test_features = self.data_test.drop(
            labels=self.variables + self.group_by, axis=1)
        features_columns = d_test_features.columns
        df_table = pd.concat(
            [d_test_features, self.data_test[tv], success_first_k], axis=1).reset_index()
        df_table.columns = df_table.columns[:-1].to_list() + ['map_k']

        # Inline definition of tooltip function
        def create_tooltip(i, col):
            # current_index = self.data_test.index[0] + i
            # first_k = success_first_k.loc[current_index]
            current_index = self.data_test.index[i]
            first_k = success_first_k.loc[current_index]

            if col == "map_k" and first_k != '-1':

                col_map = f'map_{first_k}'
                label = pred_map_kmax.loc[current_index, col_map]

                prob = map_prob.loc[current_index, col_map]

                return textwrap.dedent('''
                    Exact label: 
                    **{value1}**.

                    Probability to find exact label:
                    **{value2}**

                    **K = {value3}**
                    '''.format(value1=label,
                               value2=prob,
                               value3=first_k)
                )
            elif col == "map_k":
                return textwrap.dedent(
                    '''
                    Label not find
                    '''
                )
            else:
                return textwrap.dedent(
                    '''
                    Column: **{value}**.
                    '''.format(value=col)
                )
        # ---- End of tooltip function

        colorscale = [
            'rgb(0,68,27)',
            'rgb(0,109,44)',
            'rgb(35,139,69)',
            'rgb(65,171,93)',
            'rgb(116,196,118)',
            'rgb(161,217,155)',
            # 'rgb(199,233,192)',
            # 'rgb(229,245,224)',
            # 'rgb(247,252,245)'
        ]
        colorscale.reverse()
        if max(self.map_k) <= 6:
            adapted_colorscale = colorscale[:max(self.map_k) + 1]
        else:
            adapted_colorscale = colorscale[:] + \
                ['rgb(0,50,15)' for j in range(max(self.map_k) - 6)]

        layout = [

            dash_table.DataTable(
                id='success-table-bis',
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                page_size=15,
                page_current=0,
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {
                            'column_id': 'map_k',
                            'filter_query': '{map_k} = ' + str(i)
                        },
                        'backgroundColor': color,
                        # 'color': color,
                        # 'font-family': 'cursive',
                        # 'font-size': '26px'
                    }
                    for i, color in zip([j for j in range(1, max(self.map_k) + 1)], adapted_colorscale)
                ] + [
                    {
                        'if': {
                            'column_id': 'map_k',
                            'filter_query': '{map_k} = ' + str(-1)
                        },
                        'backgroundColor': 'rgb(226, 89, 89)',
                        'color': 'rgb(226, 89, 89)',
                        # 'font-family': 'cursive',
                        # 'font-size': '26px'
                    }
                ],
                css=[{"selector": ".show-hide", "rule": "display: none"},  # hide toggle button
                     {
                    'selector': 'td.cell--selected, td.focused',
                    'rule': 'background-color: grey !important;'  # background color of selected data
                }, {
                    'selector': 'td.cell--selected *, td.focused *',
                    'rule': 'color: white !important;'  # text color of selected data
                }],
                tooltip_data=[
                    {
                        col: {
                            'type': 'markdown',
                            'value': create_tooltip(i, col),
                            'delay': 180,
                            'duration': 10000
                        }

                        for col in df_table.columns
                    }
                    for i in range(len(df_table))
                ]
            ),
            dcc.Checklist(
                id='checkbox-hide-columns',
                options=[
                    {'label': 'Hide features columns',
                        'value': 'hide_features_columns'}
                ],
                value=[]
            ),
            html.Div(id='map-prob-bis-content')
        ]

        @app.callback(
            Output('success-table-bis', 'hidden_columns'),
            [Input('checkbox-hide-columns', 'value')]
        )
        def render_hide_features_columns(value):
            if 'hide_features_columns' in value:
                return features_columns.to_list()
            else:
                return []

        ####################################################################################
        #   Affichage de la table des prédictions après un clic sur la table des données   #
        ####################################################################################
        @app.callback(
            Output("map-prob-bis-content", "children"),
            [Input('success-table-bis', 'active_cell'),
             Input('success-table-bis', "page_current"),
             Input('success-table-bis', "page_size")]
        )
        def render_map_prob(active_cell_dict, page_current, page_size):

            if not(active_cell_dict is None):
                # current_index = self.data_test.index[0] + \
                #     page_current*page_size + active_cell_dict['row']
                table_cell_index_cur = \
                    page_current*page_size + active_cell_dict['row']
                current_index = self.data_test.index[table_cell_index_cur]

                table_length = min(max(self.map_k), len(pred_map_kmax.columns))

                map_prob_table = pd.DataFrame(np.zeros((table_length, 2)), columns=[
                                              'Predicted labels', 'probs'], index=range(1, table_length+1))
                map_prob_table.loc[:, 'Predicted labels'] = pred_map_kmax.loc[current_index].to_list(
                )
                map_prob_table.loc[:,
                                   'probs'] = map_prob.loc[current_index].to_list()
                return dash_table.DataTable(
                    id='map-prob-table-bis',
                    data=map_prob_table.to_dict('records'),
                    columns=[{'id': c, 'name': c}
                             for c in map_prob_table.columns],
                    page_size=5,
                    style_cell={'textAlign': 'center'},
                )
            else:
                return None

        return layout

    def get_dash_layout(self, app):
        fig_aggreg_specs = self.plotly_aggreg_specs()
        fig_indep_specs = self.plotly_indep_specs()
        fig_joint_specs = self.plotly_joint_specs()

        #################################
        #  Affichage de l'onglet GRAPHS #
        #################################
        if len(self.variables) == 1:

            layout_graph_content = \
                html.Div([
                    html.Div(dcc.Graph(id='indep-graph', figure=go.Figure(fig_indep_specs)),
                             style={'display': 'block', 'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}),
                ], style={'width': '100%'})

        else:
            layout_graph_content = \
                [
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(id='indep-graph', figure=go.Figure(fig_indep_specs)))),
                        dbc.Col(
                            html.Div(dcc.Graph(id='aggreg-graph', figure=go.Figure(fig_aggreg_specs))))
                    ]),
                    dbc.Row(
                        html.Div(
                            id='joint-graph',
                            children=dcc.Graph(
                                figure=go.Figure(fig_joint_specs)),
                            style={"width": "100%"}
                        )
                    )
                ]

        layout = \
            html.Div(
                [
                    dcc.Tabs(
                        [
                            dcc.Tab(label="Graphs",
                                    children=layout_graph_content
                                    ),

                            #################################
                            #  Affichage de l'onglet DATA   #
                            #################################
                            dcc.Tab(label="Data",
                                    children=[
                                        dcc.Dropdown(
                                            id='tv-success-dropdown',
                                            options=[
                                                {'label': f'Target variable = {tv}',
                                                    'value': tv}
                                                for tv in self.variables
                                            ],
                                            value=self.variables[0]
                                        ),
                                        html.Div(
                                            id='success-table-content', children=self.get_success_table_layout(app, self.variables[0]))
                                    ]
                                    )
                        ]
                    )
                ]
            )

        @app.callback(
            Output("success-table-content", "children"),
            [Input('tv-success-dropdown', 'value')]
        )
        def render_success_table_k(tv):
            if not(tv is None):
                return self.get_success_table_layout(app, tv)

        return layout


class AbsoluteErrorMeasure(PerformanceMeasureBase):

    name: str = pydantic.Field(
        "Absolute error", description="Measure label")

    calculation_method: str = pydantic.Field('eap', description="Whether we calculate the predicted scalar by \
                                                    calculating the eap of the DiscreteDistribution predicted or the map of this distribution")

    plot_trajectories: dict = pydantic.Field(
        {}, description="Plotting datas for trajectories")

    ae_trajectories: dict = pydantic.Field(
        {}, description="Histogram datas for trajectories")

    data_test: pd.DataFrame = pydantic.Field(
        pd.DataFrame(), description="Data test")

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator('calculation_method')
    def check_type(cls, val):
        if not (val in ['eap', 'map']):
            raise ValueError(f"{val} isn't a calculation method")
        return val

    def evaluate_ae(self, data_test, pred_prob):

        #self.variables = list(pred_prob.keys())

        self.result["ae"] = dict()
        self.result["pred"] = dict()

        for tv in self.variables:

            if self.calculation_method == 'eap':
                self.result["pred"][tv] = pred_prob[tv]['scores'].E()
            else:
                self.result["pred"][tv] = \
                    pred_prob[tv]['scores'].get_map().loc[:, 'map_1']

            if pred_prob[tv]['scores'].variable.domain_type == 'numeric':
                # if self.calculation_method == 'eap':
                #     # NOTE: On change le type categorical de data_test en float pour pouvoir faire la soustraction
                #     self.result["pred"] = pred_prob[tv]['scores'].E()
                self.result["ae"][tv] = \
                    (self.result["pred"][tv].astype(float) -
                     data_test[tv].astype(float)).abs()

                # elif self.calculation_method == 'map':
                #     self.result["ae"][tv] = abs(pred_prob[tv]['scores'].get_map(
                #     ).loc[:, 'map_1'].astype('float') - data_test[tv].astype('float'))

            elif pred_prob[tv]['scores'].variable.domain_type == 'interval':
                # TODO: Update code for interval
                interval_values = \
                    [
                        (float(lab.split(",")[0][1:]),
                         float(lab.split(",")[1][0:-1]))
                        for lab in data_test[tv]
                    ]

                data_test_intervals = [pd.Interval(it[0], it[1]).mid
                                       for it in interval_values]

                if self.calculation_method == 'eap':
                    self.result["ae"][tv] = abs(
                        pred_prob[tv]['scores'].E() - data_test_intervals)

                elif self.calculation_method == 'map':

                    interval_map_values = \
                        [
                            (float(lab.split(",")[0][1:]),
                                float(lab.split(",")[1][0:-1]))
                            for lab in pred_prob[tv]['scores'].get_map().loc[:, 'map_1']
                        ]

                    map_intervals = pd.Series([pd.Interval(it[0], it[1]).mid
                                               for it in interval_map_values], index=data_test.index)

                    self.result["ae"][tv] = abs(
                        map_intervals - data_test_intervals)

    def evaluate_mae(self, data_test, pred_prob):

        if self.group_by == []:
            pass
            # raise ValueError("No group_by argument has been applied")
        else:

            data_grp = data_test.groupby(self.group_by)
            self.result.setdefault("mae", {})
            for tv in self.variables:
                mae_raw = self.result["ae"][tv].groupby(
                    data_grp.ngroup()).mean()
                mae_raw.name = "MAE"
                mae_raw.index = data_grp.groups.keys()
                mae_raw.index.set_names(self.group_by, inplace=True)
                #mae_raw.index.name = self.group_by
                self.result["mae"][tv] = mae_raw.reset_index()

            # TOO UGLY FOR ME !!!!
            data_test_group_list = list(data_grp.indices.keys())
            data_index_grp_df = data_test.reset_index().set_index(self.group_by)
            #self.variables = list(pred_prob.keys())
            res = self.result["ae"]
            for tv in self.variables:

                pd.concat([data_test.loc[:, self.group_by],
                           self.result["ae"][tv]], axis=1)
                self.plot_trajectories[tv] = dict()
                self.ae_trajectories[tv] = dict()

                for d_test_group_index in data_test_group_list:
                    # for group_idx, d_test in data_grp:

                    d_test = data_index_grp_df.loc[d_test_group_index].set_index(
                        "index")

                    mae = round(res[tv].loc[d_test.index].sum() /
                                res[tv].loc[d_test.index].count(), 3)
                    #dscr = res[tv].loc[d_test.index].describe()

                    self.plot_trajectories[tv][mae] = \
                        {
                        'predicted': {
                            'x': [i for i in range(len(d_test.index))],
                            # On affiche la prediction de la RUL la plus probable (ie le map)
                            'y': pred_prob[tv]['scores'].get_map().loc[d_test.index, 'map_1'].astype('float')
                        },
                        'exact': {
                            'x': [i for i in range(len(d_test.index))],
                            'y': d_test[tv]
                        }

                    }

                    self.ae_trajectories[tv][mae] = res[tv].loc[d_test.index]

                    # self.result["mae"][tv].append(mae)

            # ipdb.set_trace()

    def evaluate(self, data_test, pred_prob):
        self.data_test = data_test
        # Perform this measure only on numeric/interval variables
        self.variables = [tv for tv, dd in pred_prob.items()
                          if dd["scores"].variable.domain_type != "label"]
        self.result = dict()
        self.evaluate_ae(data_test, pred_prob)
        self.evaluate_mae(data_test, pred_prob)

    ###############################################
    #                                             #
    #      ABSOLUTE ERROR VISUALISATIONS          #
    #                                             #
    ###############################################

    def plotly_specs_ae(self):

        fig_specs = {
            'data':
                [
                    {
                        'x': self.result["ae"][tv].to_list(),
                        'type': 'histogram',
                        # 'marginal': 'box',
                        'name': f"{tv}",
                        'boxmean': 'sd',  # represent mean and standard deviation
                        'boxpoints': 'all',  # can also be outliers, or suspectedoutliers, or False
                        'jitter': 0.3,  # add some jitter for a better separation between points
                        'pointpos': -1.8  # relative position of points wrt box
                    }
                    for tv in self.variables
                ],
            'layout': {
                'template': 'plotly_white',
                'showlegend': True,
                'legend_orientation': 'v',
                'legend': {
                    'title': {
                        'text': 'Target variables',
                        'font': {
                            'family': "sans-serif",
                            'size': 12,
                            'color': "black"},
                    },
                    'x': 1.02,
                    'y': 0.97,
                    'traceorder': "normal",
                    'font': {
                        'family': "sans-serif",
                        'size': 12,
                        'color': "black"},
                    'bgcolor': "white",
                    'bordercolor': "Black",
                    'borderwidth': 1},
                'title': {
                    'text': 'Absolute error',
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {
                        'family': "sans-serif",
                        'size': 18,
                        'color': "black"
                    }
                },
                'xaxis': {
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True
                },
                'yaxis': {
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True
                }

            }
        }

        return fig_specs

    ####################################################################################
    #  Affichage de la table donnant l'Absolute Error pour chaque ligne du jeu de test #
    ####################################################################################

    def get_table_ae(self, app, tv):

        d_test_features = self.data_test.drop(
            labels=self.variables + self.group_by, axis=1)
        features_columns = d_test_features.columns
        df_table = pd.concat([d_test_features, self.data_test[tv], pd.Series(
            self.result["ae"][tv]).round(3)], axis=1).reset_index()
        df_table.columns = df_table.columns[:-1].to_list() + ['AE']

        layout = [

            dash_table.DataTable(
                id='table-ae',
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                page_size=15,
                page_current=0,
                style_cell={'textAlign': 'center'},
                css=[
                    # hide toggle button
                    {"selector": ".show-hide", "rule": "display: none"},
                    {
                        'selector': 'td.cell--selected, td.focused',
                        'rule': 'background-color: grey !important;'  # background color of selected data
                    },
                    {
                        'selector': 'td.cell--selected *, td.focused *',
                        'rule': 'color: white !important;'  # text color of selected data
                    }
                ],

            ),
            dcc.Checklist(
                id='checkbox-show-columns-ae',
                options=[
                    {'label': 'Show features columns',
                        'value': 'show_features_columns'}
                ],
                value=[]
            )
        ]

        # Check button to hide features in the table
        @app.callback(
            Output('table-ae', 'hidden_columns'),
            [Input('checkbox-show-columns-ae', 'value')]
        )
        def render_hide_features_columns(value):
            if 'show_features_columns' in value:
                return []
            else:
                return features_columns.to_list()

        return layout

    def get_dash_layout_ae(self, app):

        layout = \
            html.Div(
                [
                    dcc.Dropdown(
                        id='tv-ae-dropdown',
                        options=[
                            {'label': f'Target variable = {tv}',
                                'value': tv}
                            for tv in self.variables
                        ],
                        value=self.variables[0]
                    ),
                    dbc.Col(
                        html.Div(
                            id="ae-table-content",
                            children=self.get_table_ae(app, self.variables[0])
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Graph(
                            figure=self.plotly_specs_ae()
                        ),
                        width=4
                    )
                ]
            )

        # Dropdown to select target variable
        @app.callback(
            Output("ae-table-content", "children"),
            [Input('tv-ae-dropdown', 'value')]
        )
        def render_success_table_k(tv):
            if not(tv is None):
                return self.get_table_ae(app, tv)

        return layout

    ###############################################
    #                                             #
    #      MEAN ABSOLUTE ERROR VISUALISATIONS     #
    #                                             #
    ###############################################

    def plotly_hist_specs_mae(self, tv, mae):

        fig_specs = {
            'data':
                [
                    {
                        'x': self.ae_trajectories[tv][mae],
                        'type': 'histogram'
                    }
                ],
            'layout': {
                'template': 'plotly_white',
                'showlegend': True,
                'title': {
                    'text': 'Mean absolute error',
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {
                        'family': "sans-serif",
                        'size': 18,
                        'color': "black"
                    }
                },
                'showlegend': False,
                'xaxis': {
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True
                },
                'yaxis': {
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True
                }
            }
        }

        return fig_specs

    def plotly_real_vs_pred_specs(self, tv, idx):

        data_test_grp = self.data_test.groupby(list(idx.keys()))
        # WARNING: This will fails with multi groupby
        idx_values = tuple(idx.values())[0]
        data_test = data_test_grp.get_group(idx_values)
        ae_val = self.result["ae"][tv].loc[data_test.index]
        pred_val = self.result["pred"][tv].loc[data_test.index]

        # TODO: Add attribute to pass x-axis name (e.g. cycle_id)
        # TODO: Move predict_index to MLPerformance level
        x_values = list(range(len(data_test)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values,
                                 y=data_test[tv],
                                 mode='lines+markers',
                                 name='real values'))
        fig.add_trace(go.Scatter(x=x_values,
                                 y=pred_val,
                                 mode='lines+markers',
                                 name='predictions'))
        # fig.add_trace(go.Scatter(x=random_x, y=random_y2,
        #                          mode='markers', name='markers'))
        idx_name = ", ".join([f"{key}: {val}" for key, val in idx.items()])
        fig.update_layout(title=f'{tv} prediction ({self.calculation_method}) vs real for {idx_name}',
                          #                          xaxis_title='Month',
                          yaxis_title=f'{tv}')

        fig_specs = fig.to_dict()

        return fig_specs

    def plotly_ae_trajectory_specs(self, tv, idx):

        data_test_grp = self.data_test.groupby(list(idx.keys()))
        # WARNING: This will fails with multi groupby
        idx_values = tuple(idx.values())[0]
        data_test = data_test_grp.get_group(idx_values)
        ae_val = (self.result["ae"][tv].loc[data_test.index] /
                  data_test[tv].astype(float)).fillna(0)
        # TODO: Add attribute to pass x-axis name (e.g. cycle_id)
        # TODO: Move predict_index to MLPerformance level
        x_values = list(range(len(data_test)))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_values,
                             y=ae_val))
        # fig.add_trace(go.Scatter(x=random_x, y=random_y2,
        #                          mode='markers', name='markers'))
        idx_name = ", ".join([f"{key}: {val}" for key, val in idx.items()])
        fig.update_layout(title=f'{tv} prediction ({self.calculation_method}) relative AE for {idx_name}',
                          #                          xaxis_title='Month',
                          yaxis_title=f'Relative AE (%)',
                          yaxis=dict(
                              tickformat=',.0%',
                              range=[0, ae_val.quantile(0.95)]
                          ))

        fig_specs = fig.to_dict()

        return fig_specs

    def get_table_mae(self, app, tv):

        # df_table = pd.concat(
        #     [
        #         # self.data_test.loc[:,self.group_by].unique(),
        #         pd.Series(self.result["mae"][tv]).round(3)
        #     ],
        #     axis=1
        # ).reset_index()
        # df_table.columns = df_table.columns[:-1].to_list() + ['MAE']

        df_table = self.result["mae"][tv]
        layout = [

            dash_table.DataTable(
                id='table-mae',
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                page_size=8,
                page_current=0,
                style_cell={'textAlign': 'center'},
                css=[{"selector": ".show-hide", "rule": "display: none"},  # hide toggle button
                     {
                    'selector': 'td.cell--selected, td.focused',
                    'rule': 'background-color: grey !important;'  # background color of selected data
                }, {
                    'selector': 'td.cell--selected *, td.focused *',
                    'rule': 'color: white !important;'  # text color of selected data
                }]
            )
        ]

        @app.callback(
            [Output("ae-trajectory-plot", "children"),
             Output("ae-pred-vs-real-plot", "children")],
            [Input('table-mae', 'active_cell'),
             Input('table-mae', "page_current"),
             Input('table-mae', "page_size")]
        )
        def render_graphs_mae(active_cell_dict, page_current, page_size):

            if not(active_cell_dict is None):
                # current_index = self.data_test.index[0] + page_current*page_size + active_cell_dict['row']
                current_index = page_current * \
                    page_size + active_cell_dict['row']
                # mae = df_table.loc[current_index,
                #                    active_cell_dict['column_id']]

                idx = df_table.iloc[current_index].drop("MAE").to_dict()

                return dcc.Graph(figure=self.plotly_ae_trajectory_specs(tv, idx)), \
                    dcc.Graph(figure=self.plotly_real_vs_pred_specs(tv, idx))

                # return dcc.Graph(figure=self.plotly_hist_specs_mae(tv, mae)), \
                #     dcc.Graph(figure=self.plotly_real_vs_pred_specs(tv, mae))
            else:
                return [], []

        return layout

    def get_dash_layout_mae(self, app):

        if len(self.variables) > 1:
            layout_content = [
                dcc.Dropdown(
                    id='tv-mae-dropdown',
                    options=[
                        {'label': f'Target variable = {tv}',
                            'value': tv}
                        for tv in self.variables
                    ],
                    value=self.variables[0]
                )
            ]
        else:
            layout_content = []

        layout = \
            html.Div(
                [
                    html.Div(id='mae-dropdown', children=layout_content),
                    dbc.Row(
                        [
                            dbc.Col(id='table-mae-content',
                                    children=self.get_table_mae(
                                        app, self.variables[0])
                                    ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                id='ae-pred-vs-real-plot',
                                width=6
                            ),
                            dbc.Col(
                                id='ae-trajectory-plot',
                                width=6
                            )
                        ]
                    ),

                ]
            )

        # TODO: Update that !
        @app.callback(
            Output("table-mae-content", "children"),
            [Input('tv-mae-dropdown', 'value')]
        )
        def render_success_table_k(tv):
            if not(tv is None):
                return self.get_table_mae(app, tv)

        return layout

    def get_dash_layout(self, app):

        if self.group_by == []:
            layout_content = self.get_dash_layout_ae(app)
        else:
            layout_content = \
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Grouped data",
                            children=self.get_dash_layout_mae(app)
                        ),
                        dcc.Tab(
                            label="Independent data",
                            children=self.get_dash_layout_ae(app)
                        ),
                    ]
                )

        layout = \
            html.Div(
                [
                    layout_content
                ]
            )

        return layout
