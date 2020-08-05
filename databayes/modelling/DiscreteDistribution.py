# ===================================================================================================#
# Classe modelisant des ensembles de distributions discretes en utilisant des dataFrames pandas      #
# josquin.foulliaron@edgemind.net                                                                    #
# ===================================================================================================#

# Imported libraries
import pkg_resources
import re
import numbers

# For computations on data
import numpy as np
import pandas as pd

import databayes.modelling.DiscreteVariable as dv

# For graph plot

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb
if "matplotlib" in installed_pkg:
    import matplotlib.pyplot as plt
if "plotly" in installed_pkg:
    import plotly.io as pio
    import plotly.offline as pof
    import plotly.graph_objects as go

# ===========================================#
#     Fonctions utilitaires pour la classe   #
# ===========================================#

# Fonction verifiant si l'interval left est entièrement inclus dans l'interval_right.


def is_included(interval_left, interval_right):
    return (interval_left.left >= interval_right.left) and (interval_left.right <= interval_right.right)


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
    def __init__(self, probs=None, name=None, domain=[], bins=[], unit=None, **df_specs):

        self.variable = dv.DiscreteVariable(name=name,
                                            domain=domain,
                                            bins=bins,
                                            unit=unit)

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

        self.index.name = name

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

        fig_dict = self.specs_plotly(**specs)

        pof.plot(fig_dict, **specs)
        # pio.show(fig_dict)

    def specs_plotly(self, **specs):
        """Create plotly plot specs for discrete distribution."""
        fig_dict = {}

        frames = [{"data": [{'x': self.columns,
                             'y': dist.to_list(),
                             'width': 0.5,
                             'type': 'bar'}],
                   "name": str(idx)}
                  for idx, dist in self.iterrows()]

        fig_dict["data"] = frames[0]["data"]

        layout_title_text = (self.variable.name + " "
                             if not(self.variable.name is None)
                             else "") + "Distribution"

        fig_dict["layout"] = {
            'xaxis_type': 'category',
            'showlegend': False,
            'hovermode': "closest",
            'font': {
                'family': "sans-serif",
                'size': 18,
                'color': "#4f4f4f"
            },
            'title': {
                'x': 0.5,
                'text': layout_title_text,
                'xanchor': 'center',
                'yanchor': 'top',
            },
            'xaxis': {
                'title': {
                    'text': "Domain",
                },
                'showline': True,
                'linewidth': 2,
                'linecolor': 'black',
            },
            'yaxis': {
                'title': {
                    'text': "Probability",
                },
                'nticks': 20,
                'showline': True,
                'linewidth': 0.5,
                'linecolor': 'black',
            }
        }

        index_name = self.index.name + ": " \
            if not(self.index.name is None) else ""
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": index_name,
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }

        fig_dict["frames"] = []

        for idx, dist in self.iterrows():
            frame = {"data": [{'x': self.columns,
                               'y': dist.to_list(),
                               'width': 0.5,
                               'type': 'bar'}],
                     "name": str(idx)}

            fig_dict["frames"].append(frame)

            slider_step = \
                {"args": [[idx],
                          {"frame": {"duration": 300, "redraw": False},
                           "mode": "immediate",
                           "transition": {"duration": 1000}}],
                 "label": idx,
                 "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]

        return fig_dict
