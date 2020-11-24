""" Class modelling discrete and finite distribution 
    extending pandas DataFrame."""

# Imported libraries
import pkg_resources

# For computations on data
import numpy as np
import pandas as pd

from .DiscreteVariable import DiscreteVariable

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

    # Service vérifiant si la somme des distributions vaut 1, leve une exception si c'est le cas

    def checksum(self, atol=1e-9):
        return (1 - self.sum(axis=1)).abs() > atol

    # Service pour calculer la probabilité que les variables suivant les distributions soit égales à une valeur donnée
    # Soit le calcul de  p(X=value)
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

    def E(self, lower_bound=-float("inf"), upper_bound=float("inf")):
        self.checksum()

        if self.variable.domain_type == "numeric":
            expect = self @ self.variable.domain
        elif self.variable.domain_type == "interval":
            variable_domain = [pd.Interval(max(lower_bound, it.left),
                                           min(upper_bound, it.right))
                               for it in self.columns]
            expect = self @ [it.mid for it in variable_domain]
        else:
            raise ValueError(
                f"The mean is not defined for domain of type {self.variable.domain_type}")
        expect.name = "Expectancy"
        return expect

    # Renvoie la variance de l'ensemble des distributions
    def sigma2(self):
        if self.variable.domain_type == "numeric":
            return (self @ [i ** 2 for i in self.variable.domain]) - self.E.pow(2)
        elif self.variable.domain_type == "interval":
            return (self @ [i.mid ** 2 for i in self.variable.domain]) - self.E.pow(2)
        else:
            raise ValueError(
                f"The variance is not defined for domain of type {self.variable.domain_type}")

    # Renvoie l ecart type de l'ensemble des distributions
    def sigma(self):
        return self.sigma2.pow(0.5)

    # def plot_distribution(self):

    #     # Init de la fenetre de plotting
    #     fig = plt.figure()
    #     fig.show()
    #     ax = fig.add_subplot()

    #     # on boucle sur les distributions existantes
    #     for i in range(0, len(self)):
    #         ax.plot(self.variable.domain, self.iloc[i, :].values, marker="o", ls='--', label=self.index[i],
    #                 fillstyle='none')

    #     ax.set(xlabel="Distribution Domain", ylabel="Probability",
    #            title=self.variable.name)
    #     plt.legend()

    def plot(self, renderer="plotly", **specs):

        plot_method = \
            getattr(self, "plot_" + renderer, None)
        if callable(plot_method):
            plot_method(**specs)
        else:
            raise ValueError(
                f"Plot rendered {renderer} not supported")

    def plot_plotly(self, **specs):
        """Show plotly discrete distribution."""

        fig_dict = self.get_plotly_dd_frames_specs(**specs)

        pof.plot(fig_dict, **specs)
        # pio.show(fig_dict)

    def get_plotly_dd_frames_specs(self, index_filter=None,
                                   data_index=None, **specs):
        """Create plotly plot specs for discrete distribution."""
        dd_ori = self.copy(deep=True)
        # ipdb.set_trace()
        #dd_ori = dd_ori.iloc[:2]
        if self.variable.domain_type == 'interval':
            dd_ori.columns = dd_ori.columns.astype(str)
            #dd_ori.columns = [it.left for it in dd_ori.columns]

        # TODO: Problem with index_filter a priori
        #index_filter = None
        # ipdb.set_trace()
        # print(data_index)
        #data_index = None
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
        dd_data_index_name = "index" if dd_data_stacked.index.names[0] is None \
            else dd_data_stacked.index.names[0]
        dd_data = dd_data_stacked.reset_index()

        ymax = min(dd_data["prob"].quantile(0.95)*1.01, 1.0)

        #print(f"COucou L = {len(dd_data)}")
        # ipdb.set_trace()
        fig = px.bar(dd_data, x=self.variable.name, y="prob",
                     animation_frame=dd_data_index_name, range_y=[0, ymax])

        #print("Ouf !")

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
        # dd_data_stacked = self.loc[index_filter].stack()
        # dd_data_stacked.name = "prob"
        # dd_data_stacked.index.set_names(self.variable.name, 1, inplace=True)

        # dd_data = dd_data_stacked.reset_index()

        # ymax = min(dd_data["prob"].max()*1.01, 1.0)

        fig = px.imshow(dd_data_plot,
                        zmax=0.1,
                        labels=dict(x=dd_data_index_name, y=self.variable.name,
                                    color="Probability"),
                        x=list(dd_data_sel.index),
                        y=list(dd_ori.columns),
                        aspect="auto",
                        color_continuous_scale='mint'
                        )

        # fig.update_xaxes(side="top")
        # fig = px.density_heatmap(dd_data,
        #                          x="index",
        #                          y=self.variable.name)
        return fig.to_dict()
