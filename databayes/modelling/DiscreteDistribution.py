""" Class modelling discrete and finite distribution 
    extending pandas DataFrame."""

# Imported libraries
import pkg_resources

# For computations on data
import numpy as np
import pandas as pd

from .DiscreteVariable import DiscreteVariable
from ..utils import ddomain_equals, pdInterval_series_from_string

# For graph plot

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb
if "plotly" in installed_pkg:
    import plotly.io as pio
    import plotly.offline as pof
    import plotly.graph_objects as go
    import plotly.express as px


# Classe utilisée pour geler les attributions directe
class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(
                "No new Attributes can be added to the Discretedistribution")
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


# ===============================================#
#    Definition de l'objet DiscreteDistribution  #
# ===============================================#

# The discrete distribution class aims to represent probability distribution for DiscreteVariable object.
# Therefore DiscreteDistribution inherits from pandas.DataFrame for probabilities storage and has a DiscreteVariable
# attribute to characterise the underlying discrete and finite random variable.
class DiscreteDistribution(FrozenClass, pd.DataFrame):

    # Constructeur de classe
    def __init__(self, probs=None, name=None,
                 domain=[], bins=[], unit=None,
                 include_lowest=False,
                 **df_specs):

        self.variable = DiscreteVariable(name=name,
                                         domain=domain,
                                         bins=bins,
                                         unit=unit,
                                         include_lowest=include_lowest)

        # Remove domain_type from df_specs if specified to avoid
        # error in the following __init__ call
        df_specs.pop("domain_type", None)

        # Pour les intervals : On convertit la liste de bins définissant le domain en liste d'intervalles pandas .
        if self.variable.domain_type == "interval":
            domain = pd.IntervalIndex.from_breaks(self.variable.bins)
        else:
            domain = self.variable.domain

        if probs is None:
            probs = np.zeros((len(df_specs.get("index", [])),
                              len(domain)))

        super(DiscreteDistribution, self).__init__(
            probs, columns=domain, **df_specs)

        self._freeze()

    @classmethod
    def read_csv(cls, filename, **kwargs):

        df = pd.read_csv(filename, **kwargs)

        dd = cls(probs=df.to_numpy(),
                 domain=list(df.columns),
                 index=df.index)

        return dd

    def checksum(self, atol=1e-9):
        return (1 - self.sum(axis=1)).abs() > atol

    def normalize(self):
        self.values[:] = self.values/self.values.sum(axis=1, keepdims=True)
    # Service pour calculer la probabilité que les variables suivant les distributions soit égales à une valeur donnée
    # Soit le calcul de  p(X=value)

    def cdf(self):
        return self.cumsum(axis=1)

    def get_prob_from_value(self, value, interval_zero_prob=True):

        # Verification que toutes les distributions somment à 1
        # self.checksum()

        if self.variable.domain_type in ["numeric", "label"]:
            if not(isinstance(value, list)):
                value = [value]

            value_idx = [idx for idx, val in enumerate(
                self.variable.domain) if val in value]
            # try:
            #     value_idx = [self.variable.domain.index(value)]
            # except ValueError:
            #     value_idx = []
        elif self.variable.domain_type == 'interval':
            if interval_zero_prob:
                value_idx = []
            else:
                value_idx = self.columns.contains(value).nonzero()[0]
        else:
            raise ValueError(
                f"Domain {self.variable.domain_type} not supported")

        if len(value_idx) > 0:
            probs = self.iloc[:, value_idx].sum(axis=1)
        elif len(value_idx) == 0:
            probs = pd.Series(0, index=self.index)
        else:
            raise ValueError(
                f"Distribution domain should have distinct modalities {self.variable.domain}")

        probs.name = f"P({value})"

        return probs

    # Service pour calculer la probabilité que les variables suivant les distributions appartiennent à un
    # intervalle donné. On calcule donc p(X in [bmin,bmax]).
    # Si il est renseigné, le paramètre user_extreme_bound est utilisé pour se substituer aux +inf et - inf pour les
    # calculs faisant intervenir les intervalles ouvert sur l'infini

    def get_prob_from_interval(self, bmin, bmax,
                               lower_bound=-float("inf"),
                               upper_bound=float("inf")):

        # Verification que toutes les distributions somment à 1
        # self.checksum()

        # ipdb.set_trace()
        probs_name = f"P([{bmin}, {bmax}])"
        if self.variable.domain_type == "numeric":
            probs = self.loc[:, (bmin <= self.columns) & (
                bmax >= self.columns)].sum(axis=1)
            probs.name = probs_name
            return probs

        # On suppose dans le cas ou le domain est de type interval, que localement a chaque intervalle la distribution
        # est uniforme
        elif self.variable.domain_type == "interval":

            b_interval = pd.Interval(bmin, bmax)

            is_left_included = self.columns.left >= b_interval.left
            is_right_included = self.columns.right <= b_interval.right
            is_included = is_left_included & is_right_included

            probs = self.loc[:, is_included].sum(axis=1)
            probs.name = probs_name

            is_overlap = self.columns.overlaps(b_interval)
            # Left overlap
            left_overlap = is_overlap & ~is_left_included
            if left_overlap.any():
                left_interval_overlap = self.columns[left_overlap]
                overlap_right_bound = min(
                    b_interval.right, left_interval_overlap.right)
                overlap_left_bound = max(
                    b_interval.left, left_interval_overlap.left)
                interval_overlap_length = max(left_interval_overlap.left, lower_bound) - \
                    min(left_interval_overlap.right, upper_bound)
                overlap_factor = \
                    (overlap_left_bound - overlap_right_bound) / \
                    interval_overlap_length
                probs_left_overlap = overlap_factor*self.loc[:, left_overlap]
                probs += probs_left_overlap.iloc[:, 0]

            right_overlap = is_overlap & ~is_right_included & ~left_overlap
            if right_overlap.any():
                right_interval_overlap = self.columns[right_overlap]
                overlap_right_bound = min(
                    b_interval.right, right_interval_overlap.right)
                overlap_left_bound = max(
                    b_interval.left, right_interval_overlap.left)
                interval_overlap_length = max(right_interval_overlap.left, lower_bound) - \
                    min(right_interval_overlap.right, upper_bound)
                overlap_factor = \
                    (overlap_left_bound - overlap_right_bound) / \
                    interval_overlap_length
                probs_right_overlap = overlap_factor*self.loc[:, right_overlap]
                probs += probs_right_overlap.iloc[:, 0]

            return probs
        else:
            raise ValueError(
                f"Domain {self.variable.domain_type} not supported")

    def get_map(self, nlargest=1, map_fmt="map_{i}"):

        nlargest = min(nlargest, len(self.columns))

        order = np.argsort(-self.values, axis=1)[:, :nlargest]
        # NOTE: the to_numpy() is meant to avoid a Pandas deprecation Warning
        np_domain = np.array(self.variable.domain)
        map_df = pd.DataFrame(np_domain[order],
                              columns=[map_fmt.format(i=i)
                                       for i in range(1, nlargest+1)],
                              index=self.index)

        cat_type = \
            pd.api.types.CategoricalDtype(categories=self.variable.domain,
                                          ordered=self.variable.domain_type != "label")

        return map_df.astype(cat_type)

    # Fonction affichant une représentation graphique de l'ensemble des distributions contenues

    # Renvoie l'espérance de l'ensemble des distributions

    def E(self,
          ensure_finite=True,
          lower_bound=-float("inf"),
          upper_bound=float("inf")):

        self.checksum()

        if self.variable.domain_type == "numeric":
            expect = self @ self.variable.domain
        elif self.variable.domain_type == "interval":
            domain_lb = self.columns[0].left
            domain_ub = self.columns[-1].right

            if ensure_finite and (domain_lb == -float("inf")) \
               and (lower_bound == -float("inf")):
                lower_bound = self.columns[0].right
            if ensure_finite and (domain_ub == float("inf")) \
               and (upper_bound == float("inf")):
                upper_bound = self.columns[-1].left

            it_mid = [pd.Interval(max(lower_bound, it.left),
                                  min(upper_bound, it.right)).mid
                      for it in self.columns]

            expect = self @ it_mid
        else:
            raise ValueError(
                f"The mean is not defined for domain of type {self.variable.domain_type}")
        expect.name = "Expectancy"
        return expect.astype(float)

    # Renvoie la variance de l'ensemble des distributions
    def sigma2(self):
        if self.variable.domain_type == "numeric":
            return (self @ [i ** 2 for i in self.variable.domain]) - self.E.pow(2)
        elif self.variable.domain_type == "interval":
            return (self @ [i.mid ** 2 for i in self.variable.domain]) - self.E.pow(2)
        else:
            raise ValueError(
                f"The variance is not defined for domain of type {self.variable.domain_type}")

    # TODO
    def quantile(self, q=0.5):
        """Quantile computations"""

        cdf = self.cdf().values
        if q <= 0:
            quant_idx = [0]*len(self)
        # elif q >= 1:
        #     quant_idx = [dom_size]*len(self)
        else:
            quant_idx = \
                (cdf <= q).cumsum(axis=1).max(axis=1)

        if self.variable.domain_type == "interval":

            if q >= 1:
                quant = [self.columns[-1].right]*len(self)
            else:
                quant = []
                for pdf_idx in range(len(self)):

                    dom_idx = quant_idx[pdf_idx]
                    if dom_idx == 0:
                        cdf_left = 0
                    else:
                        cdf_left = cdf[pdf_idx, dom_idx - 1]

                    cdf_right = cdf[pdf_idx, dom_idx]

                    alpha = (q - cdf_left)/(cdf_right - cdf_left)

                    it_left = self.columns[dom_idx].left
                    it_right = self.columns[dom_idx].right

                    if (it_left == -np.inf) or (it_right == np.inf):
                        quant_val = it_left
                    elif (it_right == np.inf):
                        quant_val = it_right
                    else:
                        quant_val = it_left + alpha*(it_right - it_left)

                    # ipdb.set_trace()

                    quant.append(quant_val)

        else:

            domains = self.columns.insert(0, np.nan)
            quant = domains[quant_idx]

        return pd.Series(quant, index=self.index, name=f"Q({q})")

    def sigma(self):
        return self.sigma2.pow(0.5)

    def plot_pdf(self, renderer="plotly", plot_type="all", **specs):

        plot_method = \
            getattr(self, "plot_pdf_" + renderer, None)
        if callable(plot_method):
            plot_method(plot_type=plot_type, **specs)
        else:
            raise ValueError(
                f"Plot rendered {renderer} not supported")

    def plot_pdf_plotly(self, plot_type="all", **specs):
        """Show plotly discrete distribution."""

        if plot_type == "all":
            fig_dict = self.get_plotly_dd_all_specs(**specs)
        elif plot_type == "frames":
            fig_dict = self.get_plotly_dd_frames_specs(**specs)
        else:
            raise ValueError(f"PDF plot type {plot_type} unsupported")

        # Indicate here the parameters not to be passed in
        # plotly plot function
        specs_to_remove = ["index_filter", "data_index", "mode"]
        plot_specs = {k: v for k, v in specs.items()
                      if not(k in specs_to_remove)}

        pof.plot(fig_dict, **plot_specs)

    def get_plotly_dd_frames_specs(self, index_filter=None,
                                   data_index=None, **specs):
        """Create plotly plot specs for discrete distribution."""
        dd_ori = self.copy(deep=True)

        if self.variable.domain_type == 'interval':
            dd_ori.columns = dd_ori.columns.astype(str)

        if not(index_filter is None):
            dd_data_stacked = dd_ori.loc[index_filter].stack()
        else:
            dd_data_stacked = dd_ori.stack()

        dd_data_stacked.name = "prob"
        dd_data_stacked.index.set_names(self.variable.name, 1, inplace=True)

        if not(data_index is None):
            data_index = data_index if index_filter is None \
                else data_index.loc[index_filter]

            dd_data_stacked.index.set_levels(levels=data_index,
                                             level=0, inplace=True)
            dd_data_stacked.index.set_names(names=data_index.name,
                                            level=0, inplace=True)

        # print(dd_data_stacked)
        dd_data_index_name = "level_0" if dd_data_stacked.index.names[0] is None \
            else dd_data_stacked.index.names[0]
        dd_data = dd_data_stacked.reset_index()

        ymax = min(dd_data["prob"].quantile(0.95)*1.01, 1.0)

        # That should not be possible
        variable_name = "level_1" if not(self.variable.name)\
            else self.variable.name

        fig = px.bar(dd_data, x=variable_name, y="prob",
                     animation_frame=dd_data_index_name, range_y=[0, ymax])

        return fig.to_dict()

    def get_plotly_dd_all_specs(self, index_filter=None,
                                data_index=None, **specs):
        """Create plotly plot specs for discrete distribution."""
        dd_ori = self.copy(deep=True)
        if self.variable.domain_type == 'interval':
            dd_ori.columns = dd_ori.columns.astype(str)

        # ipdb.set_trace()
        if not(index_filter is None):
            dd_data_sel = dd_ori.loc[index_filter]
        else:
            dd_data_sel = dd_ori.loc[:]

        dd_data_index_name = dd_data_sel.index.names[0]
        if not(data_index is None):
            data_index = data_index if index_filter is None \
                else data_index.loc[index_filter]

            dd_data_sel.index = data_index
            dd_data_index_name = dd_data_sel.index.names[0]

        dd_data_plot = dd_data_sel.transpose().to_numpy()

        fig = px.imshow(dd_data_plot,
                        zmax=0.1,
                        labels=dict(x=dd_data_index_name, y=self.variable.name,
                                    color="Probability"),
                        x=list(dd_data_sel.index),
                        y=list(dd_ori.columns),
                        aspect="auto",
                        color_continuous_scale='mint'
                        )

        return fig.to_dict()

    @staticmethod
    def apply_condition_gt(dist):
        """ Compute conditional PDF P(X|X>y) for numeric domain X and y. """
        cond_value = dist.name
        dist_cond_idx = dist.index.get_loc(cond_value)

        dist_shifted = dist.shift(-dist_cond_idx).fillna(0)

        if 'inf' in dist.index[-1]:
            # Deal with the case of the upport bound is an open interval
            nb_val_p_inf = dist_cond_idx + 1
            dist_shifted.iloc[-nb_val_p_inf:] = \
                dist.iloc[-1]/nb_val_p_inf

        dist_cond = dist_shifted/dist_shifted.sum()
        return dist_cond

    def condition_gt(self, data_cond):

        self_copy_df = self.copy(deep=True)

        # Check if domain of data_cond is identical to current variable
        # If not try to discretize it with same schema
        if not(ddomain_equals(data_cond, self.variable.domain)):
            # Note : bfill is to change the first NaN introduced
            # by the discretization of first value.
            # ==> This is ugly
            data_cond = \
                pd.cut(data_cond,
                       bins=self.variable.bins,
                       include_lowest=self.variable.include_lowest).astype(str)
            data_cond[data_cond == "nan"] = np.nan
            data_cond.fillna(method="bfill", inplace=True)

        self_copy_df.index = data_cond
        self_copy_df.columns = self_copy_df.columns.astype(str)
        # ALERT: HUGE BOTTLENECK HERE !
        # TODO: FIND A WAY TO OPTIMIZE THIS !
        scores_cond_df = self_copy_df.apply(
            self.apply_condition_gt, axis=1)

        new_dd = DiscreteDistribution(
            probs=scores_cond_df.values, **self.variable.dict())
        # Do not forget to reindex dd as origin
        new_dd.index = self.index

        return new_dd
