# -*- coding: utf-8 -*-

import logging
import pydantic
import typing
import pandas as pd
from .indicators import OHLCVIndicatorBase
import pkg_resources


installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb

if 'plotly' in installed_pkg:
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as pof

if 'dash' in installed_pkg:
    import dash
    import dash_table
    import dash_core_components as dcc
    import dash_html_components as html
    import plotly.express as px
    from dash.dependencies import Input, Output

if 'dash-bootstrap-components' in installed_pkg:
    import dash_bootstrap_components as dbc


class ohlcvDataAnalyser(pydantic.BaseModel):
    target_time_horizon: typing.List[int] = pydantic.Field(
        [], description="Target prediction horizon given"
        " in data sampling time unit")

    target_var_filter: typing.Pattern = pydantic.Field(
        ".", description="Target variable filtering pattern")
    # discretization: dict = pydantic.Field(
    #     {}, description="Discretization specifications")

    indicators: typing.Dict[str, OHLCVIndicatorBase] = pydantic.Field(
        {}, description="Dictionary of indicators")

    target_df: pd.DataFrame = pydantic.Field(
        pd.DataFrame(),
        description="Target data")

    indic_df: pd.DataFrame = pydantic.Field(
        pd.DataFrame(),
        description="Indicators data")

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator('indicators', pre=True)
    def validate_indicators(cls, indicators):

        indicator_classes_d = {cls.__name__: cls
                               for cls in OHLCVIndicatorBase.__subclasses__()}

        for indicator_name, indicator_specs in indicators.items():

            indicator_class_name_token = \
                indicator_specs.pop("class", indicator_name).split("_")

            indicator_class_name = \
                "".join([word.capitalize() if not(word.isupper()) else word
                         for word in indicator_class_name_token]) \
                if len(indicator_class_name_token) >= 2 \
                else indicator_class_name_token[0]

            if not(indicator_class_name.endswith("Indicator")):
                indicator_class_name += "Indicator"

            if not(indicator_class_name in indicator_classes_d.keys()):
                raise ValueError(
                    f"Indicator {indicator_name}"
                    f"(or class {indicator_class_name}) not supported")
            if indicator_specs is None:
                indicator_specs = {}

            indic_cls = indicator_classes_d.get(indicator_class_name)
            indicators[indicator_name] = \
                indic_cls(name=indicator_name,
                          **indicator_specs)

        return indicators

    def prepare_data(self, data_ohlcv_df, logging=logging):

        self.compute_targets(data_ohlcv_df,
                             logging=logging)

        self.compute_indicators(data_ohlcv_df,
                                logging=logging)

    def to_csv(self, filename, dropna=False, **to_csv_params):

        data_all_df = pd.concat([self.indic_df, self.target_df],
                                axis=1)

        if dropna:
            data_all_df.dropna(inplace=True)

        data_all_df.to_csv(filename, **to_csv_params)

    def compute_indicators(self, data_ohlcv_df, logging=logging):
        logging.debug("> Compute indicators")

        data_indic_df_list = []
        for indicator_name, indicator_specs in self.indicators.items():

            indic_values = \
                indicator_specs.compute(data_ohlcv_df, logging=logging)

            data_indic_df_list.append(indic_values)

            # data_indic_df_d[indicator_name] = \
            #     indicator_specs.compute(data_ohlcv_df, logging=logging)

        self.indic_df = pd.concat(data_indic_df_list, axis=1)

        return self.indic_df

    def compute_targets(self, data_ohlcv_df,
                        agg_specs={"high": "max",
                                   "low": "min",
                                   "volume": "sum"},
                        logging=logging):
        logging.debug("> Extract target variables")

        data_target_df_list = []
        for t_hrz in self.target_time_horizon:

            data_ohlcv_target_cur_df = data_ohlcv_df.rolling(t_hrz)\
                .agg(agg_specs)\
                .shift(-t_hrz)

            data_ohlcv_target_cur_df["close"] = \
                data_ohlcv_df["close"].shift(-t_hrz)

            data_ohlcv_target_cur_df.columns = \
                [var + "_t" + str(t_hrz)
                 for var in data_ohlcv_target_cur_df.columns]

            # Compute high, low returns
            ret_var = ["low", "high", "close"]
            ret_cur_var = [var + "_t" + str(t_hrz)
                           for var in ret_var]
            data_ret_target_cur_df = \
                data_ohlcv_target_cur_df[ret_cur_var]\
                .div(data_ohlcv_df["close"],
                     axis=0) - 1

            data_ret_target_cur_df.columns = \
                ["ret_" + var
                 for var in data_ret_target_cur_df.columns]

            data_target_cur_df = pd.concat([data_ohlcv_target_cur_df,
                                            data_ret_target_cur_df],
                                           axis=1)

            var_match = [var for var in data_target_cur_df.columns
                         if self.target_var_filter.search(var)]

            data_target_df_list.append(data_target_cur_df[var_match])

        self.target_df = pd.concat(data_target_df_list, axis=1)

        return self.target_df

    def build_ml_data(self, logging=logging):

        data_ohlcv_df = self.load_data_raw(logging)

        logging.info("> Build ML ready dataframe")

        self.data_ml_df = self.prepare_data(data_ohlcv_df,
                                            logging=logging)

        return self.data_ml_df

    def get_var_indic_factors(self):
        return [indic.indic_factor.name for name, indic in self.indicators.items()
                if not(indic.indic_factor is None)]

    def get_ret_var(self, var="close"):
        return ["ret_" + var + "_t" + str(t)
                for t in self.target_time_horizon]


# class ohlcvData(pydantic.BaseModel):

#     epoch: float = pydantic.Field(
#         None, description="Epoch of the data")
#     epoch_start: float = pydantic.Field(
#         time.time(), description="Epoch start ??? of the data")
#     open: float = pydantic.Field(None, description="Open price")
#     high: float = pydantic.Field(None, description="High price")
#     low: float = pydantic.Field(None, description="Low price")
#     close: float = pydantic.Field(None, description="Close price")
#     volume: float = pydantic.Field(None, description="Volume price")
#     current: float = pydantic.Field(None, description="Current price")
#     average: float = pydantic.Field(None, description="Average price")

#     def __str__(self):
#         return f"Start time: {self.epoch_start}, Open: {self.open}, "
#         f"High: {self.high}, Low: {self.low}, Close: {self.close}, "
#         f"Current: {self.current}"

#     def tick(self, price, timeframe="1h", logging=logging):

#         self.currentPrice = float(price)

#         if self.datetime is None:
#             self.datetime = time.time()

#         if (self.open is None):
#             self.open = self.currentPrice

#         if (self.high is None) or (self.currentPrice > self.high):
#             self.high = self.currentPrice

#         if (self.low is None) or (self.currentPrice < self.low):
#             self.low = self.currentPrice

#         ipdb.set_trace()
#         timedelta = utils.parseTimedelta(shared.strategy['timeframe'])
#         if time.time() >= (self.epoch_start + timedelta):
#             self.close = self.currentPrice
#             self.priceAverage = (self.high + self.low + self.close) / float(3)

#         logging.debug(str(self))

#     def isClosed(self):
#         return self.close is not None


# class ohlcvDataConnector(ohlcvDataAnalyser):
#     pass


# class ohlcvDataML(pydantic.BaseModel):

#     data_name: str = pydantic.Field(
#         "data_ml", description="ML data identifier")
#     data_ohlcv_dir: str = pydantic.Field(
#         ".", description="Directory where data OHLCV is stored")
#     data_ohlcv_filename_pattern: str = pydantic.Field(
#         r".*", description="Data OHLCV filename pattern")
#     target_time_horizon: typing.List[int] = pydantic.Field(
#         [], description="Target prediction horizon given"
#         " in data sampling time unit")
#     rebuild_ml_data: bool = pydantic.Field(
#         False, description="Force rebuilding of ML dataframe")
#     data_ml_filename: str = pydantic.Field(
#         None, description="Filename where to store ML data")
#     discretization: dict = pydantic.Field(
#         {}, description="Discretization specifications")
#     indicators: typing.Dict[str, OHLCVIndicatorBase] = pydantic.Field(
#         {}, description="Dictionary of indicators")

#     # var_features: typing.List[str] = \
#     #     pydantic.Field([], description="Computed variable features")
#     data_ml_df: pd.DataFrame = pydantic.Field(
#         pd.DataFrame(), description="Data to be explored")

#     class Config:
#         arbitrary_types_allowed = True

#     @ pydantic.validator('indicators', pre=True)
#     def validate_indicators(cls, indicators):

#         indicator_classes_d = {cls.__name__: cls
#                                for cls in OHLCVIndicatorBase.__subclasses__()}

#         for indicator_name, indicator_specs in indicators.items():

#             indicator_class_name_token = \
#                 indicator_specs.pop("class", indicator_name).split("_")

#             indicator_class_name = \
#                 "".join([word.capitalize() if not(word.isupper()) else word
#                          for word in indicator_class_name_token])\
#                 if len(indicator_class_name_token) >= 2\
#                 else indicator_class_name_token[0]

#             if not(indicator_class_name.endswith("Indicator")):
#                 indicator_class_name += "Indicator"

#             if not(indicator_class_name in indicator_classes_d.keys()):
#                 raise ValueError(
#                     f"Indicator {indicator_name}"
#                     f"(or class {indicator_class_name}) not supported")
#             if indicator_specs is None:
#                 indicator_specs = {}

#             indic_cls = indicator_classes_d.get(indicator_class_name)
#             indicators[indicator_name] = \
#                 indic_cls(name=indicator_name,
#                           **indicator_specs)

#         return indicators

#     def prepare_data_ml(self, logging=logging):
#         if self.data_ml_filename is None:
#             self.data_ml_filename = self.data_name + ".csv"

#         if os.path.exists(self.data_ml_filename) and \
#            not(self.rebuild_ml_data):
#             logging.info(
#                 f"Load ML data from filename {self.data_ml_filename}")

#             self.data_ml_df = pd.read_csv(self.data_ml_filename,
#                                           sep=",",
#                                           parse_dates=True,
#                                           index_col="datetime")
#         else:
#             logging.info("Build ML data")
#             self.build_ml_data(logging=logging)

#             logging.info(f"Save ML data in filename {self.data_ml_filename}")
#             data_ml_dirname = os.path.dirname(self.data_ml_filename)
#             if not(os.path.exists(data_ml_dirname)):
#                 os.mkdir(data_ml_dirname)
#             self.data_ml_df.to_csv(self.data_ml_filename,
#                                    sep=",",
#                                    index=True)

#         if self.discretization.get("var_specs_only", True) and \
#            (len(self.discretization.get("var_specs", {})) > 0):
#             logging.info("> Discretize data")

#             self.data_ml_df = etl.discretize(self.data_ml_df,
#                                              logging=logging,
#                                              **self.discretization)
#         else:
#             logging.info("> No discretization required")

#         logging.info("> ML data ready. Enjoy !")

#         return self.data_ml_df

#     def load_data_raw(self, logging=logging):
#         logging.info("> Load OHLCV data")

#         data_filenames_list = \
#             sorted([os.path.join(self.data_ohlcv_dir, f)
#                     for f in os.listdir(self.data_ohlcv_dir)
#                     if re.search(self.data_ohlcv_filename_pattern, f)])

#         if len(data_filenames_list) == 0:
#             logging.error("No data to load with pattern: "
#                           f"{self.data_ohlcv_filename_pattern}")
#             sys.exit(1)

#         data_ohlcv_list = []
#         if not(logging is None):
#             logging.info(f"Collect {len(data_filenames_list)} data files: ")

#         for data_filename in tqdm.tqdm(data_filenames_list,
#                                        desc="OHLV files"):
#             data_ohlcv_cur_df = pd.read_csv(data_filename,
#                                             parse_dates=True,
#                                             index_col='datetime')

#             data_ohlcv_list.append(data_ohlcv_cur_df)

#         data_ohlcv_df = pd.concat(data_ohlcv_list, axis=0).sort_index()

#         # Control Duplicated indexes
#         idx_dup = data_ohlcv_df.index.duplicated(keep=False)
#         if idx_dup.sum() > 0:
#             raise("Duplicated index not permitted: "
#                   f"{data_ohlcv_df.loc[idx_dup]}")

#         return data_ohlcv_df

#     def compute_indicators(self, data_ohlcv_df, logging=logging):
#         logging.debug("> Compute indicators")

#         data_indic_df_d = {}
#         for indicator_name, indicator_specs in self.indicators.items():

#             indic_num, indic_factor = indicator_specs.compute(
#                 data_ohlcv_df, logging=logging)

#             if not(indic_num is None):
#                 data_indic_df_d[indic_num.name] = indic_num
#             if not(indic_factor is None):
#                 data_indic_df_d[indic_factor.name] = indic_factor

#             # data_indic_df_d[indicator_name] = \
#             #     indicator_specs.compute(data_ohlcv_df, logging=logging)

#         data_indic_df = pd.DataFrame(data_indic_df_d,
#                                      index=data_ohlcv_df.index)
#         return data_indic_df

#     def extract_target_variables(self, data_ohlcv_df, logging=logging):
#         logging.info("> Extract target variables")
#         agg_specs = {
#             "high": "max",
#             "low": "min",
#             "volume": "sum",
#         }

#         data_target_df_list = []
#         for t_hrz in self.target_time_horizon:

#             data_ohlcv_target_cur_df = data_ohlcv_df.rolling(t_hrz + 1)\
#                 .agg(agg_specs)\
#                 .shift(-(t_hrz + 1))

#             data_ohlcv_target_cur_df["close"] = \
#                 data_ohlcv_df["close"].shift(-t_hrz)

#             data_ohlcv_target_cur_df.columns = \
#                 [var + "_t" + str(t_hrz)
#                  for var in data_ohlcv_target_cur_df.columns]

#             # Compute high, low returns
#             ret_var = ["low", "high", "close"]
#             ret_cur_var = [var + "_t" + str(t_hrz)
#                            for var in ret_var]
#             data_ret_target_cur_df = \
#                 data_ohlcv_target_cur_df[ret_cur_var]\
#                 .div(data_ohlcv_df["close"],
#                      axis=0) - 1

#             data_ret_target_cur_df.columns = \
#                 ["ret_" + var
#                  for var in data_ret_target_cur_df.columns]

#             data_target_cur_df = pd.concat([data_ohlcv_target_cur_df,
#                                             data_ret_target_cur_df],
#                                            axis=1)
#             data_target_df_list.append(data_target_cur_df)

#         data_target_df = pd.concat(data_target_df_list, axis=1)

#         return data_target_df

#     def build_ml_data(self, logging=logging):

#         data_ohlcv_df = self.load_data_raw(logging)

#         data_indic_df = self.compute_indicators(data_ohlcv_df,
#                                                 logging)

#         data_target_df = self.extract_target_variables(data_ohlcv_df,
#                                                        logging)

#         logging.info("> Build ML ready dataframe")

#         self.data_ml_df = pd.concat([data_ohlcv_df,
#                                      data_indic_df,
#                                      data_target_df],
#                                     axis=1).dropna()

#         return self.data_ml_df

#     def get_var_indic_factors(self):
#         return [indic.indic_factor.name for name, indic in self.indicators.items()
#                 if not(indic.indic_factor is None)]

#     def get_ret_var(self, var="close"):
#         return ["ret_" + var + "_t" + str(t)
#                 for t in self.target_time_horizon]


# class ohlcvDataMLDashboard(ohlcvDataML):

#     dashboard: dict = pydantic.Field(
#         {},
#         description="Dashboard parameters")

#     app: typing.Any = pydantic.Field(
#         None,
#         description="Dashboard parameters")

#     cache: dict = pydantic.Field(
#         {},
#         description="App cache")

#     def __init__(self, logging=logging, **data: typing.Any):
#         super().__init__(**data)

#         self.prepare_data_ml(logging=logging)
#         # Default dashboard parameters
#         default_parameters_filename = \
#             os.path.join(os.path.dirname(__file__),
#                          "ohlcv_dashboard_parameters_default.yaml")
#         with open(default_parameters_filename, 'r', encoding="utf-8") \
#                 as yaml_file:
#             try:
#                 self.dashboard = yaml.load(yaml_file,
#                                            Loader=yaml.SafeLoader)
#             except yaml.YAMLError as exc:
#                 logging.error(exc)

#         self.app = dash.Dash(__name__,
#                              external_stylesheets=[dbc.themes.BOOTSTRAP])

#     def get_future_str_index(self):
#         return ["t + " + str(t) for t in self.target_time_horizon]

#     # Dashboard components
#     # --------------------
#     def dashboard_layout(self, **params):

#         tabs = []
#         for tab_id, tab_specs in self.dashboard.get("tabs").items():
#             create_tab_method = \
#                 getattr(self, "create_tab_" + tab_id, None)
#             if callable(create_tab_method):
#                 tabs.append(create_tab_method(**tab_specs))
#             else:
#                 raise ValueError(
#                     f"Tab {tab_id} has not creation method")

#         return html.Div(dcc.Tabs(tabs))

#     def data_table_comp(self, id="data_table", **params):
#         data_table_comp = dash_table.DataTable(
#             id=id,
#             columns=[{"name": i, "id": i}
#                      for i in self.data_ml_df.columns],
#             data=self.data_ml_df.to_dict('records'),
#             filter_action="native",
#             sort_action="native",
#             sort_mode="multi",
#             column_selectable="single",
#             row_selectable="multi",
#             page_size=25,
#         )

#         return data_table_comp

#     def create_tab_rfa(self, **tab_specs):
#         tab_id = "rfa"

#         # Init figures
#         graph_list = []
#         for value_name in tab_specs.get("values").keys():
#             graph_list.append(dcc.Graph(
#                 figure=self.create_rfa_figure(value_name, **tab_specs),
#                 id=f"{tab_id}_graph_{value_name}")
#             )

#         # hidden signal value for cache management
#         hidden_update_div = \
#             html.Div(id=tab_id + '_update_signal', style={'display': 'none'})
#         # Create cache
#         self.cache["rfa"] = dict(
#             data_sel_df=pd.DataFrame()
#         )

#         tab = dcc.Tab(label=tab_specs.get("title", tab_id),
#                       children=dbc.Row([
#                           dbc.Col(self.rfa_ctrl_comp(
#                               id=f"{tab_id}_ctrl",
#                               **tab_specs), width=2),
#                           dbc.Col(graph_list),
#                           hidden_update_div,
#                       ]))

#         # Create callbacks
#         self.tab_rfa_cb(**tab_specs)

#         return tab

#     def compute_ret_data_future_stats(self, data_df, value_name):
#         ret_var = self.get_ret_var(var=value_name)

#         ret_data_df = \
#             data_df[ret_var].describe(percentiles=[0.1, 0.3, 0.5, 0.7, 0.9])\
#                             .transpose()

#         ret_data_df.index = self.get_future_str_index()

#         return ret_data_df

#     def create_rfa_figure(self, value_name, **tab_specs):

#         fig = self.dashboard.setdefault("tabs", {})\
#                             .setdefault("rfa", {})\
#                             .setdefault("figures", {})\
#                             .setdefault(value_name, go.Figure())

#         dash_values_specs = tab_specs.get("values")
#         stats_specs = dash_values_specs[value_name]

#         for stat_name, stat_specs in stats_specs.items():
#             ret_data_df = \
#                 self.compute_ret_data_future_stats(self.data_ml_df,
#                                                    value_name)
#             fig.add_trace(
#                 go.Scatter(
#                     x=self.get_future_str_index(),
#                     y=ret_data_df[stat_name],
#                     name=f"{stat_name} - ref",
#                     mode="lines+markers",
#                     **stat_specs["ref"]
#                 ),
#             )

#         fig.update_layout(
#             title={
#                 'text': value_name.capitalize() + " value evolution",
#                 'y': 0.85,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'middle'},
#             xaxis_title="Time",
#             yaxis_title="returns",
#             yaxis_tickformat=".2%",

#         )

#         return fig

#     def rfa_ctrl_comp(self,
#                       id="rfa_ctrl",
#                       **params):

#         control_layout = []

#         # Extract factor indicators
#         for indic_name, indic in self.indicators.items():

#             if indic.indic_factor is None:
#                 continue

#             factor_cats = indic.indic_factor.cat.categories.to_list()

#             control_layout.append(
#                 dbc.Row([
#                     dbc.Col(
#                         width=1),
#                     dbc.Col(
#                         html.Label(indic.name_label))
#                 ]))
#             control_layout.append(
#                 dbc.Row([
#                     dbc.Col(
#                         width=1),
#                     dbc.Col(
#                         dcc.Dropdown(
#                             value=[],
#                             options=[{'label': val, 'value': val}
#                                      for val in factor_cats],
#                             multi=True,
#                             id=id + "_" + indic.indic_factor.name))
#                 ]))

#         control_layout.append(
#             dbc.Row([
#                 dbc.Col(
#                     width=1),
#                 dbc.Col(
#                     html.Div(id=id + "_nb_history_data"))
#             ]))

#         return control_layout

#     def create_tab_rfa_figure_cb(self, value_name, **tab_specs):

#         @ self.app.callback(
#             Output(f'rfa_graph_{value_name}', 'figure'),
#             [Input('rfa'+'_update_signal', 'children')])
#         def cb_fun(value):

#             fig = self.dashboard.get("tabs", {})\
#                                 .get("rfa", {})\
#                                 .get("figures", {})\
#                                 .get(value_name, go.Figure())

#             data_sel_df = self.cache['rfa']['data_sel_df']
#             if len(data_sel_df) > 0:
#                 ret_data_df = \
#                     self.compute_ret_data_future_stats(
#                         data_sel_df, value_name)

#                 dash_values_specs = tab_specs.get("values")
#                 stats_specs = dash_values_specs[value_name]

#                 for stat_name, stat_specs in stats_specs.items():
#                     trace_name = f"{stat_name} - sel"

#                     # Find index of trace if exists
#                     trace_idx = [i for i, trace in enumerate(fig.data)
#                                  if trace.name == trace_name]
#                     if len(trace_idx) == 1:
#                         fig.data[trace_idx[0]].y = ret_data_df[stat_name]
#                     else:
#                         fig.add_trace(
#                             go.Scatter(
#                                 x=self.get_future_str_index(),
#                                 y=ret_data_df[stat_name],
#                                 name=trace_name,
#                                 mode="lines+markers",
#                                 **stat_specs["sel"]
#                             ),
#                         )

#             else:
#                 fig.data = [trace for trace in fig.data
#                             if not(trace.name.endswith(" - sel"))]

#             return fig

#         return cb_fun

#     def tab_rfa_cb(self, **tab_specs):

#         indicator_factors_list = self.get_var_indic_factors()
#         input_cb = [Input("rfa_ctrl_" + factor, "value")
#                     for factor in indicator_factors_list]

#         @ self.app.callback([Output('rfa' + '_update_signal',
#                                     'children'),
#                              Output('rfa_ctrl_nb_history_data',
#                                     'children')],
#                             input_cb)
#         def update_rfa_data_sel(*ctrl):

#             if all([len(v) == 0 for v in ctrl]):
#                 self.cache["rfa"]["data_sel_df"] = pd.DataFrame()
#                 return None, f"# data selected: {len(self.data_ml_df)}"

#             # Get factor selected data
#             index_sel = tuple([slice(None) if len(v) == 0 else tuple(v)
#                                for v in ctrl])
#             # NOTE Pandas seems that .loc(axis=0)[(slice(None),)] does not work
#             # but .loc(axis=0)[slice(None)] is OK ?!?
#             if len(index_sel) == 1:
#                 index_sel = index_sel[0]

#             data_sel_df = self.data_ml_df.set_index(
#                 indicator_factors_list).sort_index()\
#                 .loc(axis=0)[index_sel]

#             self.cache["rfa"]["data_sel_df"] = \
#                 data_sel_df

#             nb_history_data = len(self.cache["rfa"]["data_sel_df"])

#             return None, f"# data selected: {nb_history_data}"

#         # Create graph refreshing callback
#         for value_name in tab_specs.get("values", {}).keys():
#             self.create_tab_rfa_figure_cb(value_name, **tab_specs)

#     def create_tab_candles(self, **tab_specs):
#         tab_id = "candles"

#         # hidden signal value for cache management
#         hidden_update_div = \
#             html.Div(id=tab_id + '_update_signal', style={'display': 'none'})

#         # Create cache
#         self.cache["candles"] = dict()

#         tab = dcc.Tab(label=tab_specs.get("title", tab_id),
#                       children=dbc.Row([
#                           dbc.Col(self.candles_ctrl_comp(
#                               id=f"{tab_id}_ctrl",
#                               **tab_specs), width=2),
#                           dbc.Col(dcc.Graph(
#                               # figure=self.create_candles_figure(**tab_specs),
#                               id=f"{tab_id}_graph",
#                               style={'height': '100%'})
#                           ),
#                           hidden_update_div,
#                       ], style={'height': "800px"}))

#         # Create callbacks
#         self.tab_candles_cb(**tab_specs)

#         return tab

#     def candles_ctrl_comp(self,
#                           id="candles_ctrl",
#                           **params):

#         control_layout = []

#         control_layout.append(
#             dbc.Row([
#                 dbc.Col(
#                     width=1),
#                 dbc.Col(
#                     html.Label("Date range selection"))
#             ]))

#         data_date_min = \
#             self.data_ml_df.index.min().to_pydatetime().date()
#         data_date_max = datetime.now().date()
#         data_date_end_cur = self.data_ml_df.index[-1000].to_pydatetime().date()
#         control_layout.append(
#             dbc.Row([
#                 dbc.Col(
#                     width=1),
#                 dbc.Col(
#                     dcc.DatePickerRange(
#                         id=id + "_date_range",
#                         min_date_allowed=data_date_min,
#                         max_date_allowed=data_date_max,
#                         initial_visible_month=data_date_max,
#                         start_date=data_date_end_cur,
#                         end_date=data_date_max
#                     )
#                 ),
#             ]))

#         control_layout.append(
#             dbc.Row([
#                 dbc.Col(
#                     width=1),
#                 dbc.Col(
#                     html.Div(id=id + "_nb_history_data"))
#             ]))

#         control_layout.append(
#             dbc.Row([
#                 dbc.Col(
#                     width=1),
#                 dbc.Col(
#                     dcc.Loading(id=id + "_loading",
#                                 type="default",
#                                 fullscreen=True,
#                                 children=html.Div(id=id + "_loading_output"),
#                                 style={'backgroundColor': 'transparent'})
#                 )
#             ]))

#         return control_layout

#     def tab_candles_cb(self, **tab_specs):
#         """This method aims to create callbacks for candles tab."""
#         input_cb = [Input("candles_ctrl_date_range", "start_date"),
#                     Input("candles_ctrl_date_range", "end_date")
#                     ]

#         # Callback reacting at paramters changes

#         @ self.app.callback([Output('candles_update_signal',
#                                     'children'),
#                              Output('candles_ctrl_nb_history_data',
#                                     'children')],
#                             input_cb)
#         def update_candles_data_sel(start_date, end_date):
#             datetime_start = datetime.strptime(start_date, "%Y-%m-%d")
#             datetime_end = datetime.strptime(end_date, "%Y-%m-%d") + \
#                 timedelta(days=1) - timedelta(minutes=1)

#             # data_ml_sel_df =
#             self.cache["candles"]["data_ml_sel_df"] = \
#                 self.data_ml_df.loc[datetime_start:datetime_end]
#             nb_history_data = len(self.cache["candles"]["data_ml_sel_df"])
#             return None, f"# data selected: {nb_history_data}"

#         # Create callback to update figure
#         self.create_candles_figure_cb(**tab_specs)

#     def create_candles_figure_cb(self, **specs):
#         """Method to create the callback to update candles figure."""
#         @ self.app.callback(
#             [Output('candles_graph', 'figure'),
#              Output('candles_ctrl_loading_output', 'children')],
#             [Input('candles_update_signal', 'children')])
#         def create_candles_figure(value):

#             INCREASING_COLOR = '#14A76C'
#             DECREASING_COLOR = '#FF652F'

#             data_ml_sel_df = self.cache["candles"]["data_ml_sel_df"]

#             data_sel_idx = data_ml_sel_df.index

#             # Add data
#             data = [dict(
#                 type='candlestick',
#                 open=data_ml_sel_df["open"],
#                 high=data_ml_sel_df["high"],
#                 low=data_ml_sel_df["low"],
#                 close=data_ml_sel_df["close"],
#                 x=data_ml_sel_df.index,
#                 yaxis='y2',
#                 name=self.data_name,
#                 increasing=dict(line=dict(color=INCREASING_COLOR)),
#                 decreasing=dict(line=dict(color=DECREASING_COLOR)),
#             )]

#             layout = dict()

#             fig = dict(data=data, layout=layout)

#             # Adjust layout
#             fig['layout'] = dict()
#             fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
#             fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True))
#             fig['layout']['yaxis'] = dict(
#                 domain=[0, 0.2], showticklabels=False)
#             fig['layout']['yaxis2'] = dict(domain=[0.2, 0.8])
#             fig['layout']['legend'] = dict(
#                 orientation='h', y=0.9, x=0.3, yanchor='bottom')
#             fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

#             # Add range selector
#             rangeselector = dict(
#                 visible=True,
#                 x=0, y=0.9,
#                 bgcolor='rgba(150, 200, 250, 0.4)',
#                 font=dict(size=13),
#                 buttons=list([
#                     dict(count=1,
#                          label='reset',
#                          step='all'),
#                     dict(count=1,
#                          label='1yr',
#                          step='year',
#                          stepmode='backward'),
#                     dict(count=3,
#                          label='3 mo',
#                          step='month',
#                          stepmode='backward'),
#                     dict(count=1,
#                          label='1 mo',
#                          step='month',
#                          stepmode='backward'),
#                     dict(count=1,
#                          label='1 day',
#                          step='day',
#                          stepmode='backward'),
#                     dict(count=6,
#                          label='6 hours',
#                          step='hour',
#                          stepmode='backward'),
#                     dict(count=1,
#                          label='1 hour',
#                          step='hour',
#                          stepmode='backward'),
#                     dict(step='all')
#                 ]))

#             fig['layout']['xaxis']['rangeselector'] = rangeselector

#             # Set volume bar chart colors
#             colors = []

#             for i in range(len(data_ml_sel_df.close)):
#                 if i != 0:
#                     if data_ml_sel_df.close[i] > data_ml_sel_df.close[i-1]:
#                         colors.append(INCREASING_COLOR)
#                     else:
#                         colors.append(DECREASING_COLOR)
#                 else:
#                     colors.append(DECREASING_COLOR)

#             fig['data'].append(dict(x=data_ml_sel_df.index,
#                                     y=data_ml_sel_df.volume,
#                                     marker=dict(color=colors),
#                                     type='bar', yaxis='y', name='Volume'))

#             # Add indicators
#             for indic_idx, (indic_name, indic) in enumerate(self.indicators.items()):

#                 if not(indic.indic_num is None):
#                     fig['data'].append(dict(x=data_sel_idx,
#                                             y=indic.indic_num.loc[data_sel_idx],
#                                             yaxis=f'y{indic_idx+3}',
#                                             name=indic.name_label + " (num)",
#                                             legendgroup=indic.name_label,
#                                             showlegend=True,
#                                             visible="legendonly",
#                                             **indic.plot_styling.get("indic_num", {})
#                                             ))
#                     fig['layout'][f'yaxis{indic_idx+3}'] = dict(domain=[0.2, 0.8],
#                                                                 showticklabels=False,
#                                                                 overlaying="y2",
#                                                                 side="right")

#                 if not(indic.indic_factor is None):

#                     indic_colormap = indic.plot_styling.get("indic_factor", {})\
#                         .pop("colormap", None)

#                     if indic_colormap:
#                         indic_marker_color = indic.indic_factor.loc[data_sel_idx].map(
#                             indic_colormap)
#                         indic.plot_styling["indic_factor"]["marker"]["color"] = indic_marker_color.tolist(
#                         )

#                     fig['data'].append(dict(x=data_sel_idx,
#                                             y=indic.indic_num.loc[data_sel_idx],
#                                             yaxis=f'y{indic_idx+3}',
#                                             name=indic.name_label +
#                                             " (factor)",
#                                             legendgroup=indic.name_label,
#                                             showlegend=True,
#                                             visible="legendonly",
#                                             **indic.plot_styling.get("indic_factor", {})
#                                             ))
#                     fig['layout'][f'yaxis{indic_idx+3}'] = dict(domain=[0.2, 0.8],
#                                                                 showticklabels=False,
#                                                                 overlaying="y2",
#                                                                 side="right")

#             return go.Figure(fig), value

#     def run_dashboard(self, **params):
#         # app = self.create_dashboard_app(**params)

#         self.app.layout = self.dashboard_layout(**params)

#         self.app.run_server(**params)
