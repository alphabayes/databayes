# -*- coding: utf-8 -*-
import pandas as pd
import pydantic
import typing

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb
if 'plotly' in installed_pkg:
    import plotly
    import plotly.figure_factory as ff

if 'dash' in installed_pkg:
    import dash
    import dash_table
    import dash_core_components as dcc
    import dash_html_components as html
    import plotly.express as px
    from dash.dependencies import Input, Output

if 'dash-bootstrap-components' in installed_pkg:
    import dash_bootstrap_components as dbc


class DashboardParameters(pydantic.BaseModel):
    pass


# source : https://dash.plotly.com/datatable/interactivity
class DataExplorer(pydantic.BaseModel):

    data: pd.DataFrame = pydantic.Field(
        pd.DataFrame(), description="Data to be explored")

    dashboard_parameters: DashboardParameters = pydantic.Field(
        DashboardParameters(), description="Dashboard parameters")

    class Config:
        arbitrary_types_allowed = True

    def data_table_comp(self, id="data_table", **params):
        data_table_comp = \
            dash_table.DataTable(
                id=id,
                columns=[{"name": i, "id": i}
                         for i in self.data.columns],
                data=self.data.to_dict('records'),
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                page_size=25,
            )

        return data_table_comp

    def plot_2d_ctrl_comp(self,
                          id="plot_2d_ctrl",
                          **params):

        control_layout = []

        control_layout.append(dbc.Row([
            dbc.Col(
                html.Label("x-axis data"),
                width=3),
            dbc.Col(
                dcc.Dropdown(
                    value=list(self.data.columns)[0],
                    options=[{'label': val, 'value': val}
                             for val in self.data.columns],
                    multi=False,
                    id=id + "_filter_x"
                ), width=5),
            dbc.Col(
                dcc.Checklist(
                    options=[
                        {'label': 'Reversed', 'value': 'reversed'},
                    ],
                    value=[],
                    id=id + "_reversed_x",
                ), width=3),
        ]))

        control_layout.append(dbc.Row([
            dbc.Col(
                html.Label("y-axis data"),
                width=3),
            dbc.Col(
                dcc.Dropdown(
                    value=list(self.data.columns)[-1],
                    options=[{'label': val, 'value': val}
                             for val in self.data.columns],
                    multi=False,
                    id=id + "_filter_y"
                ), width=5),
            dbc.Col(
                dcc.Checklist(
                    options=[
                        {'label': 'Reversed', 'value': 'reversed'},
                    ],
                    value=[],
                    id=id + "_reversed_y",
                ), width=3),
        ]))

        control_layout.append(dbc.Row([
            dbc.Col(
                html.Label("Group"),
                width=3),
            dbc.Col(
                dcc.Dropdown(
                    value=None,
                    options=[{'label': val, 'value': val}
                             for val in self.data.columns],
                    multi=False,
                    id=id + "_group"
                ), width=5),
        ]))

        control_layout.append(dbc.Row([
            dbc.Col(
                html.Label("Plot type"),
                width=3),
            dbc.Col(
                dcc.RadioItems(
                    options=[
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Line', 'value': 'line'},
                    ],
                    value='scatter',
                    labelStyle={'display': 'inline-block'},
                    id=id + "_plot_type"
                ), width=5),
        ]))

        return control_layout

    def plot_pairs_ctrl_comp(self,
                             id="plot_pairs_ctrl",
                             **params):

        control_layout = []

        control_layout.append(dbc.Row([
            dbc.Col(
                html.Label("Dimensions"),
                width=3),
            dbc.Col(
                dcc.Dropdown(
                    value=list(self.data.columns),
                    options=[{'label': val, 'value': val}
                             for val in self.data.columns],
                    multi=True,
                    id=id + "_dimensions"
                ), width=9),
        ]))

        control_layout.append(dbc.Row([
            dbc.Col(
                html.Label("Group"),
                width=3),
            dbc.Col(
                dcc.Dropdown(
                    value=None,
                    options=[{'label': val, 'value': val}
                             for val in self.data.columns],
                    multi=False,
                    id=id + "_group"
                ), width=9),
        ]))

        return control_layout

# @ app.callback(Output('output', 'children'), [Input('dropdown', 'value')])
# def display_output(value):
#     return str(value)

    def dashboard_layout(self, **params):
        return html.Div(dcc.Tabs([
            dcc.Tab(label='Pairs plot',
                    children=dbc.Row([
                        dbc.Col(self.plot_pairs_ctrl_comp(
                            id="plot_pairs_ctrl",
                            **params), width=3),
                        dbc.Col(dcc.Graph(id="plot_pairs_graph"),
                                width=9)
                    ])),

            dcc.Tab(label='Plot 2D',
                    children=dbc.Row([
                        dbc.Col(self.plot_2d_ctrl_comp(
                            id="plot_2d_ctrl",
                            **params), width=3),
                        dbc.Col(dcc.Graph(id="plot_2d_graph"),
                                width=9)
                    ])),

            dcc.Tab(label='Data',
                    children=self.data_table_comp(**params))
        ]))

    def create_cb_plot_2d(self, app):

        # Define callbacks
        @app.callback(Output('plot_2d_graph', 'figure'),
                      [
                          Input('plot_2d_ctrl_filter_x', 'value'),
                          Input('plot_2d_ctrl_filter_y', 'value'),
                          Input('plot_2d_ctrl_group', 'value'),
                          Input('plot_2d_ctrl_plot_type', 'value'),
                          Input('plot_2d_ctrl_reversed_x', 'value'),
                          Input('plot_2d_ctrl_reversed_y', 'value'),
        ])
        def update_plot_2d(filter_x,
                           filter_y,
                           group,
                           plot_type,
                           reversed_x,
                           reversed_y):

            plot_method = getattr(px, plot_type, None)
            if callable(plot_method):
                fig = plot_method(self.data,
                                  x=filter_x,
                                  y=filter_y,
                                  color=group,
                                  hover_name=group)
            else:
                raise ValueError(
                    f"Plot type '{plot_type}' not supported")

            if len(reversed_x) == 1:
                fig.update_layout(xaxis={"autorange": "reversed"})
            if len(reversed_y) == 1:
                fig.update_layout(yaxis={"autorange": "reversed"})

            return fig

    def create_cb_plot_pairs(self, app):

        # Define callbacks
        @app.callback(Output('plot_pairs_graph', 'figure'),
                      [
                          Input('plot_pairs_ctrl_dimensions', 'value'),
                          Input('plot_pairs_ctrl_group', 'value'),
        ])
        def update_plot_pairs(dimensions,
                              group):

            fig = px.scatter_matrix(self.data,
                                    dimensions=dimensions,
                                    color=group,
                                    hover_name=group)
            return fig

    def create_dashboard_app(self, **params):
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.create_cb_plot_2d(app)
        self.create_cb_plot_pairs(app)

        return app

    def run_dashboard(self, **params):
        app = self.create_dashboard_app(**params)

        app.layout = self.dashboard_layout(**params)

        app.run_server(**params)
