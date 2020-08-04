import typing
import inspect

import datautils_common.modelling.core as dmc
import numpy as np
import pandas as pd
import tqdm
import pydantic
import datautils_common.modelling.DiscreteDistribution as dd
import datautils_common.utils.performance_measure as pm
import datautils_common.utils.ml_performance as ml_perf

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb

if 'plotly' in installed_pkg:
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as pof

class MLPerformanceComparator(pydantic.BaseModel):

    ml_perf_models: typing.Dict[str, ml_perf.MLPerformance] = pydantic.Field(dict(), description="Dict of MLPerformance objects")

    def add_ml_perf(self, perf):
        self.ml_perf_models[perf.model.id] = perf



    ###############################################
    #                                             #
    #               VISUALISATIONS                #
    #                                             #
    ###############################################


    def plotly_indep_specs_on_target_variable(self, tv):
        
        result = dict()
        for id_model, ml_perf_model in self.ml_perf_models.items():
            result[id_model] = ml_perf_model.measures["success"].result_to_frame()

        fig_specs = \
            {
                'data': 
                    [
                        {
                            'x': result[id_model]["indep"].index.get_level_values('map_k').to_list(),
                            'y': result[id_model]["indep"][tv]["mean"].to_list(),
                            'type': 'scatter',
                            'name': f"{id_model}",
                            'hoverinfo': 'text',
                            'hovertext': [f'Target variable: {tv} <br>K = {k}<br><br><b>Sucess rate = {success}<b>'
                                            for k, success in zip(result[id_model]["indep"].index.get_level_values('map_k'), result[id_model]["indep"][tv]["mean"])]
                        }
                        for id_model in self.ml_perf_models.keys() if tv in result[id_model]["indep"]
                    ],
                'layout': {
                    'template': 'plotly_white',
                    'xaxis_type': 'category',
                    'showlegend': True,
                    # 'legend_title_text': 'Target variables',
                    'legend_orientation': 'v',
                    'legend': 
                        {
                            'title': {
                                'text': 'Models',
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
                            'borderwidth': 1
                        },
                    'title': {
                        'text': 'Success rates',
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
                    'yaxis':
                        {
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

    def plotly_indep_specs_on_k(self, k):
        
        result = dict()
        for id_model, ml_perf_model in self.ml_perf_models.items():
            result[id_model] = ml_perf_model.measures["success"].result_to_frame()
        
        fig_specs = \
            {
                'data': 
                    [
                        {
                            'x': result[id_model]['indep'].columns.get_level_values('variable').unique().to_list(),
                            'y': result[id_model]['indep'].xs('mean', level='stats', axis=1).loc[k].values[0].tolist(),
                            'type': 'bar',
                            'name': f"{id_model}",
                            'hoverinfo': 'text',
                            'hovertext': [f'Model: {mod} <br>K = {k}<br><br><b>Sucess rate = {success}<b>'
                                            for mod, success in zip(result[id_model]['indep'].columns.get_level_values('variable').unique(), result[id_model]['indep'].xs('mean', level='stats', axis=1).loc[k].values[0].tolist())]
                        }
                        for id_model in self.ml_perf_models.keys() if k in result[id_model]["indep"].index.get_level_values('map_k').to_list()
                    ],
                'layout': {
                    'template': 'plotly_white',
                    'xaxis_type': 'category',
                    'showlegend': True,
                    # 'legend_title_text': 'Target variables',
                    'legend_orientation': 'v',
                    'legend': 
                        {
                            'title': {
                                'text': 'Models',
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
                            'borderwidth': 1
                        },
                    'title': {
                        'text': 'Success rates',
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
                    'yaxis':
                        {
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


    def get_dash_layout(self, app):
        
        union_target_variables = []
        for id_mod, ml_perf_mod in self.ml_perf_models.items():
            for var in ml_perf_mod.model.var_targets:
                if not var in union_target_variables:
                    union_target_variables.append(var)

        union_k = []
        for id_mod, ml_perf_mod in self.ml_perf_models.items():
            for k in ml_perf_mod.measures.get('success').map_k:
                if not k in union_k:
                    union_k.append(k)
        
        layout = \
            html.Div(
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button("Change axis", 
                                    id='change-axis-button', 
                                    color="primary",
                                    n_clicks=0),
                                width=1
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='tv-success-dropdown1',
                                    options=[
                                        {'label': f'Target variable = {tv}',
                                            'value': tv}
                                        for tv in union_target_variables
                                    ],
                                    value=union_target_variables[0]
                                ),
                                width=4
                            )
                        ], 
                        no_gutters=True,
                        justify='start'
                    ),
                    dbc.Row(
                        html.Div(
                            dcc.Graph(id='indep-on-tv-k-graph', figure=go.Figure(self.plotly_indep_specs_on_target_variable(union_target_variables[0]))),
                            style={'width': '95%'}
                        )
                    )
                    
                ]
            )

        @app.callback(
            [Output("tv-success-dropdown1", "options"),
            Output("tv-success-dropdown1", "value")],
            [Input('change-axis-button', 'n_clicks')]
        )
        def change_axis_button_click(n):
            if n % 2 == 0:
                return [
                        {'label': f'Target variable = {tv}',
                            'value': tv}
                        for tv in union_target_variables
                        ], union_target_variables[0]
            else:
                return [
                        {'label': f'K = {k}',
                            'value': k}
                        for k in union_k
                        ], union_k[0]

        @app.callback(
            Output("indep-on-tv-k-graph", "figure"),
            [Input('tv-success-dropdown1', 'value')]
        )
        def render_success_indep_graph_on_tv(val):
            if not(val is None) and val in union_target_variables:
                return go.Figure(self.plotly_indep_specs_on_target_variable(val))
            elif not(val is None) and val in union_k:
                return go.Figure(self.plotly_indep_specs_on_k(val))

        return layout




    def make_nav(self, app):

        union_measures = []
        for id_mod, ml_perf_mod in self.ml_perf_models.items():
            for m in ml_perf_mod.measures.keys():
                if not m in union_measures:
                    union_measures.append(m)
        
        style_button = {
            'background': 'none!important',
            'border': 'none',
            'width': '100%!important',
            #'color': '#444',
            'cursor': 'pointer',
            'padding': '18px',
            'width': '100%',
            'text-align': 'left',
            'outline': 'none',
            'transition': '0.4s'
            }

        def make_comparison_nav():
        # we use this function to make the example items to avoid code duplication
            return dbc.Card(
                [
                    dbc.CardHeader(
                        
                        html.Button(
                            "MODELS COMPARISON",
                            id="group-1-toggle",
                            style=style_button
                        )
                        
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            [   
                                dbc.Nav(
                                    [
                                        dbc.NavLink(
                                            "".join([word.capitalize() + " " for word in measure_name.split("_")]),
                                            id=f'{measure_name}-button',
                                            href=f'/{measure_name}',
                                            className='measure-button'
                                        )
                                        for measure_name in union_measures
                                    ],
                                    vertical=True,
                                    pills=True
                                )
                            ]
                        ),
                        id=f"collapse-1",
                    ),
                ]
            )

        def make_ml_performance_nav():
        
            return dbc.Card(
                [
                    dbc.CardHeader(
                        
                        html.Button(
                            "ML PERFORMANCE MODELS",
                            id="group-2-toggle",
                            style=style_button
                        )
                        
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            [   
                                dbc.Nav(
                                    [
                                        dbc.NavLink(
                                            "".join([word.capitalize() + " " for word in id_model.split("_")]),
                                            id=f'{id_model}-button',
                                            href=f'/{id_model}'
                                        )
                                        for id_model in self.ml_perf_models
                                    ],
                                    vertical=True,
                                    pills=True
                                )
                            ]
                        ),
                        id=f"collapse-2",
                    ),
                ]
            )

        def make_ml_performance_specs_nav():
            
            return dbc.Card(
                [
                    dbc.CardHeader(
                        html.Button(
                            "MODELS SPECS",
                            id="group-3-toggle",
                            style=style_button
                        )
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            [   
                                dbc.Nav(
                                    [
                                        dbc.NavLink(
                                            "".join([word.capitalize() + " " for word in id_model.split("_")]),
                                            id=f'specs-{id_model}-button',
                                            href=f'/specs_{id_model}'
                                        )
                                        for id_model in self.ml_perf_models
                                    ],
                                    vertical=True,
                                    pills=True
                                )
                            ]
                        ),
                        id=f"collapse-3",
                    ),
                ]
            )


        accordion = dbc.Card(
            
                [
                    make_comparison_nav(),
                    make_ml_performance_nav(),
                    make_ml_performance_specs_nav()
                ]
        )


        @app.callback(
            [Output(f"collapse-{i}", "is_open") for i in range(1, 4)],
            [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 4)],
            [State(f"collapse-{i}", "is_open") for i in range(1, 4)],
        )
        def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return True, True, True
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "group-1-toggle" and n1:
                return not is_open1, is_open2, is_open3
            elif button_id == "group-2-toggle" and n2:
                return is_open1, not is_open2, is_open3
            elif button_id == "group-3-toggle" and n3:
                return is_open1, is_open2, not is_open3
            return is_open1, is_open2, is_open3

        return accordion


    def get_mlperf_specs(self, id_model):
            return html.Div(
                children= \
                dbc.CardColumns(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "HYPER PARAMETERS"
                                ),
                                dbc.CardBody(
                                    [html.P(f"{key}: {value}") for key, value in self.ml_perf_models.get(id_model).model.hyper_parameters.__dict__.items()]
                                )
                            ],
                            color="secondary", 
                            inverse=True
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "FIT PARAMETERS"
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(f"{key}: {value}") 
                                        for key, value in self.ml_perf_models.get(id_model).fit_parameters.__dict__.items()
                                    ] + [html.P(f"group_by: {self.ml_perf_models.get(id_model).group_by}")]
                                )
                            ],
                            color="secondary", 
                            inverse=True
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "MEASURES PARAMETERS"
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [html.H5(key)] + [
                                                html.P(f"{name}: {value}")
                                                    
                                                for name, value in perf_measure.__dict__.items() 
                                                if (not(name in ["result", "variables", "group_by"]) 
                                                    and not isinstance(value, dict) 
                                                    and not isinstance(value, pd.DataFrame) 
                                                    and not isinstance(value, dd.DiscreteDistribution))
                                            ]
                                        )
                                        for key, perf_measure in self.ml_perf_models.get(id_model).measures.items()
                                    ]
                                )
                            ],
                            color="secondary", 
                            inverse=True
                        )     
                    ]
                )
            )

    def run_app(self, data, **kwargs):

        for _, ml_perf_mod in self.ml_perf_models.items():
            ml_perf_mod.run(data)

        app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
        server = app.server
        
        #app.layout = self.get_dash_layout(app)
        app.layout = html.Div(
            [
                dcc.Location(id='url', refresh=False),
                dbc.Row(
                    [
                        dbc.Col(self.make_nav(app), width=2),
                        dbc.Col(html.Div(id='measure-test-content'))
                    ]
                )
            ]
        )
        
        #On run tous les sous-layout pour que les callbacks soient bien appelés
        self.get_dash_layout(app)
        self.ml_perf_models.get(list(self.ml_perf_models.keys())[0]).get_dash_layout(app)

        @app.callback(Output('measure-test-content', 'children'),
              [Input('url', 'pathname')])
        def display_page(pathname):
            if pathname == '/success':
                return self.get_dash_layout(app)
            elif pathname in [f"/{id_model}" for id_model in self.ml_perf_models.keys()]:
                return self.ml_perf_models.get(pathname[1:]).get_dash_layout(app)
            elif pathname in [f"/specs_{id_model}" for id_model in self.ml_perf_models.keys()]:
                return self.get_mlperf_specs(pathname[7:])
            else:
                return '404'

        
        app.run_server(**kwargs)