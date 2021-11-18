import pydantic
from .Variable import DFVariable
import pandas as pd
import typing
import pkg_resources
import math
import json
import numpy as np
import itertools

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb

# TODO :
# Extends pd.Series for DFPotential (in fact, there's nothing to add, just possibly some methods...
# Use index name to identify variable name and index as domain
# Really no difference between Potential and Series, except Potential must be on real values (at least on domain that supports sum and product)

# Extends pd.Series for CPDs too
# -> Slicing CPDs on conditioning var gives discrete distribution
# -> Slicing CPDs on normalized var gives potential
# -> Maintain cond var and norm var_childre
# -> Maintain norm domain normalized !
# -> Use DF to pretty printting columns as normalize variables

# Source for extending Pandas : https://pandas.pydata.org/docs/development/extending.html


PandasSeries = typing.TypeVar('pandas.core.frame.Series')


class DFPotential(pydantic.BaseModel):

    name: str = pydantic.Field(
        None, description="Potential name")
    var: typing.List[DFVariable] = pydantic.Field(
        [], description="Potential variables")
    fun: PandasSeries = pydantic.Field(
        None, description="Array of potential values")

    @pydantic.root_validator
    def check_obj(cls, obj):

        if obj['fun'] is None:
            obj['fun'] = pd.Series([], dtype=float, name=obj['name'])
        else:
            obj['fun'] = pd.Series(obj['fun'], dtype=float, name=obj['name'])

        return obj

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        self.update_shape()

    def _dom_equals(self, other):
        return (len(self.var) == len(other.var)) and \
            all([s_var == o_var for s_var, o_var in zip(self.var, other.var)])

    def __eq__(self, other):

        return self._dom_equals(other) and \
            self.name == other.name and \
            (self.fun == other.fun).all()

    @classmethod
    def init_from_dataframe(cls, data):
        var_list = []
        for var in data.columns:
            if data[var].dtype.name == "category":
                dfvar = DFVariable(
                    name=var,
                    domain=data[var].cat.categories.to_list(),
                    ordered=data[var].cat.ordered)

            elif data[var].dtype.name == "object":
                dfvar = DFVariable(
                    name=var,
                    domain=sorted(data[var].unique().tolist()))
            else:
                raise ValueError(
                    f"Variable {var} is not of categorical or object type")

            var_list.append(dfvar)

        return cls(name=data.index.name,
                   var=var_list)

    def _multiindex_codes_from_var(self):
        return list(zip(*itertools.product(*[range(len(var.domain))
                                             for var in self.var])))

    def _multiindex_from_var(self):
        codes = self._multiindex_codes_from_var()
        midx_levels = []
        for var in self.var:
            if var.domain_type == "interval":
                midx_levels.append(
                    pd.IntervalIndex(data=var.domain))
            else:
                midx_levels.append(
                    pd.CategoricalIndex(data=var.domain,
                                        categories=var.domain,
                                        ordered=var.ordered,
                                        name=var.name))

        return pd.MultiIndex(levels=midx_levels,
                             codes=codes,
                             names=[var.name for var in self.var])

    def to_cpd(self, var_norm):

        return DFCPD(name=self.name,
                     fun=self.fun,
                     var=self.var,
                     var_norm=var_norm)

    def get_var(self, var_name):
        for var in self.var:
            if var.name == var_name:
                return var

        raise ValueError(
            f"Variable {var_name} is not in {type(self).__name__} domain")

    def var_names(self):
        return [var.name for var in self.var]

    def dict(self, **kwrds):

        kwrds.update(exclude={'fun'})
        obj = super().dict(**kwrds)
        obj.update(fun=self.fun.to_list())
        return obj

    def save(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.dict(), outfile)

    @classmethod
    def load(cls, filename):
        return cls.parse_file(filename)

    def get_nb_conf(self):
        return math.prod([len(var.domain)
                          for var in self.var]) \
            if len(self.var) > 0 else 0

    def update_shape(self):

        if len(self.var) == 0:
            self.fun = pd.Series([], dtype=float, name=self.name)
        else:

            # pot_index = pd.MultiIndex.from_product(
            #     [var.domain for var in self.var],
            #     names=[var.name for var in self.var])
            pot_index = self._multiindex_from_var()

            nb_val_cur = len(self.fun.values)
            nb_var_target = len(pot_index)
            nb_rep = nb_var_target // nb_val_cur if nb_val_cur > 0 else 0
            nb_shape_comp = nb_var_target % nb_val_cur == 0 \
                if nb_val_cur > 0 else False

            self.fun = pd.Series(self.fun.values.repeat(nb_rep)
                                 if nb_shape_comp else 0,
                                 dtype=float,
                                 name=self.name,
                                 index=pot_index).fillna(0)
            #     _fun_values = 0

            # else:
            #     _fun_values = self.fun.values

            # self.fun = self.fun.reindex(pot_index)
            # self.fun.values[:] = _fun_values

    def update_var(self, var: typing.List[DFVariable]):

        for var_new in var:
            for j, var_old in enumerate(self.var):
                if var_new.name == var_old.name:
                    self.var[j] = var_new
                    break

            self.var.append(var_new)

        self.update_shape()

    def _adapt_var_order(self, var_target=None):
        if var_target is None:
            var_target = self.var

        pot_res_fun_var = self.fun.index.names
        var_res_names_tmp = [var.name for var in var_target]
        var_perm = [var_res_names_tmp.index(var)
                    for var in pot_res_fun_var]
        self.var = [var_target[i] for i in var_perm]

    def __mul__(self, other):

        pot_res = self.__class__()
        pot_res.name = f"{self.name}*{other.name}"

        pot_res.fun = self.fun.__mul__(other.fun)

        # Reorder var attribute to be consistent with fun names
        var_res = list(set(self.var) | set(other.var))
        pot_res._adapt_var_order(var_res)

        return pot_res

    # def __imul__(self, other):
    #     ipdb.set_trace()

    #     self.fun.__imul__(other.fun)

    #     # Reorder var attribute to be consistent with fun names
    #     pot_res_fun_var = self.fun.index.names
    #     var_res_tmp = list(set(self.var) | set(other.var))
    #     var_res_names_tmp = [var.name for var in var_res_tmp]
    #     var_perm = [var_res_names_tmp.index(var)
    #                 for var in pot_res_fun_var]
    #     self.var = [var_res_tmp[i] for i in var_perm]

    def marg(self, var: typing.List[str] = []):

        if len(var) == 0:
            return self.fun.sum()

        pot_res = self.__class__()
        pot_res.name = f"{self.name}"
        pot_res.fun = self.fun.groupby(level=var).sum()

        # Reorder var attribute to be consistent with fun names
        pot_res._adapt_var_order(self.var)

        return pot_res

    def sum(self, var: typing.List[str] = []):

        var_names = [self_var.name for self_var in self.var]
        var_to_marg = [v for v in var_names if not(v in var)]

        return self.marg(var=var_to_marg)


class DFCPD(DFPotential):

    var_norm: typing.List[str] = pydantic.Field(
        [], description="Normalized variable names")

    counts: PandasSeries = pydantic.Field(
        None, description="Array of configuration counts")

    @pydantic.root_validator
    def check_obj(cls, obj):

        obj = super().check_obj(obj)

        if obj['counts'] is None:
            obj['counts'] = pd.Series([], dtype=float, name=obj['name'])
        else:
            obj['counts'] = pd.Series(
                obj['counts'], dtype=float, name=obj['name'])

        return obj

    @classmethod
    def init_from_dataframe(cls, data, var_norm):

        cpd = super().init_from_dataframe(data)

        if isinstance(var_norm, str):
            var_norm = [var_norm]

        cpd.set_var_norm(var_norm)

        return cpd

    def __str__(self):
        return self.fun_to_df().to_string()

    def print_counts(self):
        print(self.counts_to_df().to_string())

    def __eq__(self, other):
        return super().__eq__(other) and \
            (self.counts == other.counts).all()

    def fun_to_df(self):
        if self.var_norm == self.fun.index.names:
            return self.fun.to_frame().transpose()
        else:
            return self.fun.unstack(self.var_norm)

    def counts_to_df(self):
        if self.var_norm == self.fun.index.names:
            return self.counts.to_frame().transpose()
        else:
            return self.counts.unstack(self.var_norm)

    def dict(self, **kwrds):

        kwrds.update(exclude={'counts'})
        obj = super().dict(**kwrds)
        obj.update(counts=self.counts.to_list())
        return obj

    def set_var_norm(self, var: typing.List[str]):

        if len(var) == 0:
            raise ValueError("Normalization variable should be specified")

        var_names = [self_var.name for self_var in self.var]

        var_undef = [v for v in var if not(v in var_names)]
        if len(var_undef) > 0:
            raise ValueError(f"Unrecognized variables {', '.join(var_undef)}")

        self.var_norm = var

        self._normalize()

    def update_shape(self):

        super().update_shape()

        if len(self.counts) == len(self.fun):
            self.counts.index = self.fun.index
        else:
            self.counts = pd.Series(0, dtype=float, name=self.name,
                                    index=self.fun.index)

        if len(self.var_norm) == 0:
            self.var_norm = [var.name for var in self.var]

        self._normalize()

    def get_nb_norm_conf(self):
        var_norm_dim = [len(var.domain)
                        for var in self.var if var.name in self.var_norm]
        return math.prod(var_norm_dim)

    def _adapt_var_order(self, var_target=None):

        super()._adapt_var_order(var_target)

        # Reorder counts to be consistent
        self.counts.index = self.fun.index

    def _normalize(self, from_counts=False):

        if len(self.var_norm) == 0:
            return

        var_names = [var.name for var in self.var]
        var_cond = [var for var in var_names if not(var in self.var_norm)]

        # Div by zeros management
        na_val = 1/self.get_nb_norm_conf()

        # Copy idx to maintain index consistency after normalization
        #fun_idx = self.fun.index

        # IDEA: align common index at the beginning
        if from_counts:
            norm_fact = self.counts.sum() if len(var_cond) == 0 \
                else self.counts.groupby(level=var_cond).sum()
            self.fun = self.counts.div(norm_fact).fillna(na_val)

        else:
            norm_fact = self.fun.sum() if len(var_cond) == 0 \
                else self.fun.groupby(level=var_cond).sum()
            self.fun = self.fun.div(norm_fact).fillna(na_val)

        # Recopy index if index changes during normlization

        # if (self.fun.index != fun_idx).any():
        #     self.fun.index = fun_idx

        # self.fun = self.fun.div(norm_fact, level=self.var_norm).fillna(na_val)
        self._adapt_var_order()

    def adapt_data(self, data):
        """Utility method to ensure series has well formatted categorical data, i.e. string labels.
        """

        # parents_var = self.parents.get(var_name, [])
        # var_dim = [var_name] + parents_var
        data_new = data.copy()

        # Check if input dataframe has consistent catagorical variables
        for var in self.var:

            if data_new[var.name].dtype.name != "category":

                if var.domain_type == "interval":

                    data_new[var.name] = pd.cut(data_new[var.name],
                                                bins=var.get_bins())

                else:

                    cat_type = pd.CategoricalDtype(categories=var.domain,
                                                   ordered=var.domain_type != "label")

                    data_new[var.name] = data_new[var.name].astype(cat_type)

            elif var.domain != data_new[var.name].cat.categories.tolist():
                err_msg = f"Domain of variable {var.name}: {var.domain}\n"
                err_msg += f"Series categories: {data_new[var.name].cat.categories}\n"
                err_msg += f"Inconsistency detected"
                raise ValueError(err_msg)

        return data_new

    def check_data_consistency(self, data):
        """Utility method to check data consistency with CPD.
        """

        # Check if input dataframe has consistent catagorical variables
        for var in self.var:

            if data[var.name].dtype.name != "category":
                return False

            if var.domain != data[var.name].cat.categories.tolist():
                return False

        return True

    def fit(self, data_df,
            update_fit=False,
            update_decay=0,
            logger=None,
            **kwrds):
        """
        This function aims to compute the joint counts associated to CPT parameters from a Pandas
        dataframe.

        Parameters
        - data_df: a Pandas DataFrame consisting only of categorical variables.
        - update_fit: indicates if current joint counts has to be updated with new observation.
        - update_decay: decay coef in [0,1] to reduce importance of current count comparared to new fitted data. 0 means that old data is as same weight than new data. Otherwise count_update = count_new_fit + (1-decay)*count_old. Note that decay coef == 1 is equivalent to set update_fit == False, i.e. consider only new data in the fitting process.
        """

        # parents_var = self.parents.get(var_name, [])
        # var_dim = [var_name] + parents_var
        if not(self.check_data_consistency(data_df)):
            data_df = self.adapt_data(data_df)

        data_counts = data_df[self.var_names()].value_counts()

        data_counts.index = data_counts.index\
            .reorder_levels(self.counts.index.names)

        if update_fit and update_decay < 1:
            self.counts *= 1 - update_decay

        self.counts.loc[data_counts.index] += data_counts

        self._normalize(from_counts=True)

    def predict(self, data_df,
                **kwrds):
        """
        TODO:
        """

        if not(self.check_data_consistency(data_df)):
            data_df = self.adapt_data(data_df)

        var_cond = [var for var in self.fun.index.names
                    if not(var in self.var_norm)]

        cpd_cond_idx = pd.Index(data_df[var_cond])

        cpd_fun_pred = self.fun.unstack(self.var_norm).loc[cpd_cond_idx]
        cpd_fun_pred.index = data_df.index
        if cpd_fun_pred.index.name is None:
            cpd_fun_pred.index.name = "index"

        dfvar_norm = [var for var in self.var
                      if var.name in self.var_norm]
        pred_var = [DFVariable(name=data_df.index.name,
                               domain=data_df.index.tolist())]
        pred_var.extend(dfvar_norm)

        cpd_fun_pred = cpd_fun_pred.stack()
        # ipdb.set_trace()
        return DFCPD(name=self.name,
                     fun=cpd_fun_pred,
                     var=pred_var,
                     var_norm=self.var_norm)

    def cdf(self):
        return self.fun_to_df().cumsum(axis=1)

    def get_var_norm(self):
        return [var for var in self.var
                if var.name in self.var_norm]

    def get_var_cond(self):
        return [var for var in self.var
                if not(var.name in self.var_norm)]

    def set_prob(self, probs, cond=None):

        var_names = [var.name for var in self.var]

        cpd = self.fun_to_df().stack()

        if cond is None:
            # ipdb.set_trace()
            cpd.values[:] = probs
        else:
            var_cond_names = [var.name for var in self.get_var_cond()]
            if isinstance(cond, dict):
                cond = tuple([cond.get(var, slice(None))
                              for var in var_cond_names])

            cpd.loc[cond] = probs

        cpd = cpd.reorder_levels(var_names)

        self.fun.values[:] = cpd.values[:]

        self._normalize()

    def get_prob(self, value=None,
                 value_min=-float("inf"),
                 value_max=float("inf"),
                 interval_zero_prob=True,
                 lower_bound=-float("inf"),
                 upper_bound=float("inf")):

        if not(value is None):
            return self.get_prob_from_value(
                value=value,
                interval_zero_prob=interval_zero_prob)
        else:
            return self.get_prob_from_interval(
                value_min=value_min,
                value_max=value_max,
                lower_bound=-float("inf"),
                upper_bound=float("inf"))

    def get_prob_from_interval(self,
                               value_min=-float("inf"),
                               value_max=float("inf"),
                               lower_bound=-float("inf"),
                               upper_bound=float("inf")):

        if len(self.var_norm) != 1:
            raise ValueError(
                "get_prob method only work on single"
                "normalized domain for now")

        cpd = self.fun_to_df()
        dfvar_norm = self.get_var(self.var_norm[0])

        # ipdb.set_trace()
        probs_name = f"P([{value_min}, { value_max}])"
        if dfvar_norm.domain_type == "numeric":
            probs = cpd.loc[:, (value_min <= cpd.columns.astype(float)) & (
                value_max >= cpd.columns.astype(float))].sum(axis=1)
            probs.name = probs_name
            return probs

        # Hypothèse : Cas domain intervalle
        # La distribution intra-intervalle est uniforme
        elif dfvar_norm.domain_type == "interval":

            b_interval = pd.Interval(value_min,  value_max)

            is_left_included = cpd.columns.left >= b_interval.left
            is_right_included = cpd.columns.right <= b_interval.right
            is_included = is_left_included & is_right_included

            probs = cpd.loc[:, is_included].sum(axis=1)
            probs.name = probs_name

            is_overlap = cpd.columns.overlaps(b_interval)
            # Left overlap
            left_overlap = is_overlap & ~is_left_included
            if left_overlap.any():
                left_interval_overlap = cpd.columns[left_overlap]
                overlap_right_bound = min(
                    b_interval.right, left_interval_overlap.right)
                overlap_left_bound = max(
                    b_interval.left, left_interval_overlap.left)
                interval_overlap_length = max(left_interval_overlap.left, lower_bound) - \
                    min(left_interval_overlap.right, upper_bound)
                overlap_factor = (overlap_left_bound - overlap_right_bound) / \
                    interval_overlap_length
                probs_left_overlap = overlap_factor*cpd.loc[:, left_overlap]
                probs += probs_left_overlap.iloc[:, 0]

            right_overlap = is_overlap & ~is_right_included & ~left_overlap
            if right_overlap.any():
                right_interval_overlap = cpd.columns[right_overlap]
                overlap_right_bound = min(
                    b_interval.right, right_interval_overlap.right)
                overlap_left_bound = max(
                    b_interval.left, right_interval_overlap.left)
                interval_overlap_length = max(right_interval_overlap.left, lower_bound) - \
                    min(right_interval_overlap.right, upper_bound)
                overlap_factor = (overlap_left_bound - overlap_right_bound) / \
                    interval_overlap_length
                probs_right_overlap = overlap_factor*cpd.loc[:, right_overlap]
                probs += probs_right_overlap.iloc[:, 0]

            return probs
        else:
            raise ValueError(
                f"Domain {dfvar_norm.domain_type} not supported")

    def get_prob_from_value(self, value, interval_zero_prob=True):

        if len(self.var_norm) != 1:
            raise ValueError(
                "get_prob method only work on single"
                "normalized domain for now")

        cpd = self.fun_to_df()
        dfvar_norm = self.get_var(self.var_norm[0])
        if dfvar_norm.domain_type in ["numeric", "label"]:
            if not(isinstance(value, list)):
                value = [value]

            value_idx = [idx for idx, val in enumerate(
                dfvar_norm.domain) if val in value]
        elif dfvar_norm.domain_type == 'interval':
            if interval_zero_prob:
                value_idx = []
            else:
                # ipdb.set_trace()
                value_idx = cpd.columns.contains(value).nonzero()[0]
        else:
            raise ValueError(
                f"Domain {dfvar_norm.domain_type} not supported")

        if len(value_idx) == 0:
            probs = pd.Series(0, index=cpd.index)
        else:
            probs = cpd.iloc[:, value_idx].sum(axis=1)

        probs.name = f"P({value})"

        return probs

    def argmax(self, nlargest=1, force_rank_index=False):

        cpd = self.fun_to_df()

        nlargest = min(nlargest, len(cpd.columns))

        order = np.argsort(-cpd.values, axis=1)[:, :nlargest]

        argmax_vals = [cpd.columns.to_frame().iloc[order[:, i]]
                       for i in range(order.shape[1])]

        argmax_df = pd.concat(argmax_vals, axis=0, ignore_index=True)

        dfvar_cond = self.get_var_cond()
        dfvar_norm = self.get_var_norm()

        if nlargest > 1 or force_rank_index:
            argmax_df_index = pd.MultiIndex.from_product(
                [range(1, nlargest+1)] + [var.domain for var in dfvar_cond],
                names=["rank"] + [var.name for var in dfvar_cond])
        else:
            argmax_df_index = cpd.index

        argmax_df.index = argmax_df_index

        for var in dfvar_norm:

            var_dtype = pd.CategoricalDtype(categories=var.domain,
                                            ordered=var.domain_type != "label")

            argmax_df[var.name] = argmax_df[var.name].astype(var_dtype)

        return argmax_df

    def expectancy(self,
                   ensure_finite=True,
                   lower_bound=-float("inf"),
                   upper_bound=float("inf")):

        if len(self.var_norm) != 1:
            raise ValueError(
                "get_prob method only work on single"
                "normalized domain for now")

        cpd = self.fun_to_df()
        dfvar_norm = self.get_var(self.var_norm[0])

        if dfvar_norm.domain_type == "numeric":
            expect = cpd @ dfvar_norm.domain
        elif dfvar_norm.domain_type == "interval":
            domain_lb = cpd.columns[0].left
            domain_ub = cpd.columns[-1].right

            if ensure_finite and (domain_lb == -float("inf")) \
               and (lower_bound == -float("inf")):
                lower_bound = cpd.columns[0].right
            if ensure_finite and (domain_ub == float("inf")) \
               and (upper_bound == float("inf")):
                upper_bound = cpd.columns[-1].left

            it_mid = [pd.Interval(max(lower_bound, it.left),
                                  min(upper_bound, it.right)).mid
                      for it in cpd.columns]

            expect = cpd @ it_mid
        else:
            raise ValueError(
                f"The mean is not defined for domain of type {dfvar_norm.domain_type}")

        expect.name = "Expectancy"

        return expect.astype(float)

    # TOBE TESTED
    def variance(self):

        if len(self.var_norm) != 1:
            raise ValueError(
                "get_prob method only work on single"
                "normalized domain for now")

        cpd = self.fun_to_df()
        dfvar_norm = self.get_var(self.var_norm[0])

        if dfvar_norm.domain_type == "numeric":
            return (cpd @ [i ** 2 for i in dfvar_norm.domain]) -\
                cpd.expectancy().pow(2)
        elif dfvar_norm.domain_type == "interval":
            return (cpd @ [i.mid ** 2 for i in dfvar_norm.domain]) -\
                cpd.expectancy().pow(2)
        else:
            raise ValueError(
                f"The variance is not defined for domain of type {dfvar_norm.domain_type}")

    def quantile(self, q=0.5):
        """Quantile computations"""

        if len(self.var_norm) != 1:
            raise ValueError(
                "get_prob method only work on single"
                "normalized domain for now")

        cpd = self.fun_to_df()
        dfvar_norm = self.get_var(self.var_norm[0])

        cdf = self.cdf().values

        if q <= 0:
            quant_idx = [0]*len(cpd)
        # elif q >= 1:
        #     quant_idx = [dom_size]*len(self)
        else:
            quant_idx = (cdf <= q).cumsum(axis=1).max(axis=1)

        if dfvar_norm.domain_type == "interval":

            if q >= 1:
                quant = [cpd.columns[-1].right]*len(cpd)
            else:
                quant = []
                for pdf_idx in range(len(cpd)):

                    dom_idx = quant_idx[pdf_idx]
                    if dom_idx == 0:
                        cdf_left = 0
                    else:
                        cdf_left = cdf[pdf_idx, dom_idx - 1]

                    cdf_right = cdf[pdf_idx, dom_idx]

                    alpha = (q - cdf_left)/(cdf_right - cdf_left)

                    it_left = cpd.columns[dom_idx].left
                    it_right = cpd.columns[dom_idx].right

                    if (it_left == -np.inf) or (it_right == np.inf):
                        quant_val = it_left
                    elif (it_right == np.inf):
                        quant_val = it_right
                    else:
                        quant_val = it_left + alpha*(it_right - it_left)

                    # ipdb.set_trace()

                    quant.append(quant_val)

        else:

            domains = cpd.columns.insert(0, np.nan)
            quant = domains[quant_idx]

        return pd.Series(quant, index=cpd.index, name=f"Q({q})")
