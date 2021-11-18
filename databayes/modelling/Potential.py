import pydantic
from .Variable import Variable, DFVariable
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


NumpyNDArray = typing.TypeVar('np.core.ndarray')


class Potential(pydantic.BaseModel):

    name: str = pydantic.Field(
        None, description="Potential name")
    var: typing.List[Variable] = pydantic.Field(
        [], description="Potential variables")
    fun: typing.Any = pydantic.Field(
        None, description="Potential function")

    def get_var(self, var_name):
        for var in self.var:
            if var.name == var_name:
                return var

        raise ValueError(
            f"Variable {var_name} is not in {type(self).__name__} domain")

    def get_var_names(self):
        """Get the list of variables names."""
        return [var.name for var in self.var]

    def var2dim(self, var_query: typing.List[str]):
        """Returns the dimension of a tuple of variables in the potential.

        Syntax
        ------
        dim = p.var2dim(var_query)

        Parameter(s)
        ------------
        var_query : tuple of str
        Sequence of variable names to get corresponding dimension in the potential.

        Returned value(s)
        -----------------
        dim : a tuple of non negative integers
        dim[ i ] corresponds to the dimension (or axis) of var_query[ i ] in the potential.

        See Also
        --------

        Examples
        --------
        >>> var_v1 = DFVariable(name="v1", domain=['ok', 'degraded', 'failure'])
        >>> var_x = DFVariable(name="x", domain=[1, 2])
        >>> var_T = DFVariable(name="T", domain=['nothing'])
        >>>
        >>> pot = DFPotential(name="P1", var=[var_v1, var_x, var_T])
        >>> pot.var2dim(["T", "v1"]) == (2, 0)
        >>> pot.var2dim(["x", "v1", "T"]) == (1, 0, 2)
        """
        return tuple(self.get_var_names().index(v) for v in var_query)

    def dim2var(self, dim: typing.List[int]):
        """Returns the tuple of variables corresponding to some given dimension indices in the potential.

        Syntax
        ------
        var = p.dim2var(dim)

        Parameter(s)
        ------------
        dim : tuple of non-negative integer between 0 and len( self.var ) - 1
        Sequence of dimension indices.

        Returned value(s)
        -----------------
        var : a tuple of variables
        var[ i ] corresponds to the variable identify by self.var[ dim[ i ] ].

        See Also
        --------
        var2dim

        Examples
        --------
        >>> var_X = DFVariable(name="X", domain=['x0', 'x1', 'x2'])
        >>> var_Y = DFVariable(name="Y", domain=['y0', 'y1', 'y2'])
        >>> var_Z = DFVariable(name="Z", domain=['z0'])
        >>>
        >>> pot = DFPotential(name="P1", var=[var_X, var_Y, var_Z])
        >>>
        >>> pot.dim2var((2,)) == ('Z',)
        >>> pot.dim2var((2, 0, 1)) == ('Z', 'X', 'Y')
        """
        var_names = self.get_var_names()
        return tuple(var_names[i] for i in dim)


class DFPotential(Potential):

    var: typing.List[DFVariable] = pydantic.Field(
        [], description="Potential variables")
    fun: NumpyNDArray = pydantic.Field(
        None, description="Array of potential values")

    @pydantic.root_validator
    def check_obj(cls, obj):
        if obj['fun'] is None:
            obj['fun'] = np.ndarray(shape=(0,), dtype=float)
        else:
            obj['fun'] = np.array(obj['fun'], dtype=float)

        return obj

    def __repr__(self):
        return self.to_series().to_string()

    def get_var_sizes(self):
        return [len(var.domain) for var in self.var]

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

    def set_fun(self, fun):
        """Assignment method for attribute fun."""

        self.fun = np.array(fun, dtype='float')
        self.update_shape()

    def __setitem__(self, key, value):
        pot_s = self.to_series()
        pot_s.loc[key] = value
        self.fun[:] = pot_s.values[:]

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

    def to_series(self):
        pot_index = self._multiindex_from_var()

        return pd.Series(self.fun.flatten(),
                         dtype=float,
                         name=self.name,
                         index=pot_index).fillna(0)

    def to_cpd(self, var_norm):

        return DFCPD(name=self.name,
                     fun=self.fun,
                     var=self.var,
                     var_norm=var_norm)

    def dict(self, **kwrds):

        kwrds.update(exclude={'fun'})
        obj = super().dict(**kwrds)
        obj.update(fun=self.fun.tolist())
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
            self.fun = np.ndarray(shape=(0,), dtype=float)
        else:

            # ipdb.set_trace()

            fun_values = self.fun.flatten()
            nb_val_cur = len(fun_values)
            nb_val_target = self.get_nb_conf()
            nb_rep = nb_val_target // nb_val_cur if nb_val_cur > 0 else 0
            nb_shape_comp = nb_val_target % nb_val_cur == 0 \
                if nb_val_cur > 0 else False

            self.fun = np.ndarray(shape=(nb_val_target,), dtype=float)
            self.fun[:] = fun_values.repeat(nb_rep) if nb_shape_comp else 0
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

    @staticmethod
    def compute_trans_to_op(var_p1, var_p2):
        """Tool function used to compute the reshaping needed to accomplish arithmetic operations between two potentials (e.g. element-wise addition or product).

        Syntax
        ------
        var_res, perm_p1, resh_p1, perm_p2, resh_p2 = DFPotential.compute_trans_to_op( var_p1, var_p2 )

        Parameters
        ----------
        var_p1 : tuple of DFVariable
            Discrete and finite variable defining the domain of discrete and finite potential p1.

        var_p2 : tuple of DFVariable
            Discrete and finite variable defining the domain of discrete and finite potential p2.

        Returned values
        ---------------
        var_res : tuple of n DFVariable
            Discrete and finite variable defining the domain of the resulting discrete and finite potential resulting from the operation between p1 and p2. Practically, var_res is the union of var_p1 and var_p2.

        perm_p1 : tuple of n non negative integers
            Tuple corresponding to the permutation to apply to p1 to perform the operation with p2.

        resh_p1 : tuple of n non negative integers
            Tuple corresponding to the axis reshaping transformation to apply to p1 to perform the operation with p2.

        perm_p2 : tuple of n non negative integers
            Tuple corresponding to the permutation to apply to p2 to perform the operation with p1.

        resh_p2 : tuple of n non negative integers
            Tuple corresponding to the axis reshaping transformation to apply to p2 to perform the operation with p1.

        See also
        --------
        Arithmetic operation methods, i.e. __mul__, __add__ etc.

        Example
        -------

        """

        # Compute the resulting potential variable sequence
        var_p1_list = [var.name for var in var_p1]
        shape_p1 = [len(var.domain) for var in var_p1]
        sz_p1 = np.prod(shape_p1)

        var_p2_list = [var.name for var in var_p2]
        shape_p2 = [len(var.domain) for var in var_p2]
        sz_p2 = np.prod(shape_p2)

        # Compare the sizes of each potential
        if sz_p1 >= sz_p2:
            # If p1 is larger than p2
            # No permutation is needed for p1
            perm_p1 = None
            # Reshaping is done to prepare p1 with respect to variables in p2 not in p1
            v2add = tuple(v2 for v2 in var_p2_list if v2 not in var_p1_list)
            resh_p1 = shape_p1.copy() + [1]*len(v2add)

            # Build the resulting variable dict by adding new p2 variables to those in p1
            var_res = var_p1.copy()
            [var_res.append(var)
             for var in var_p2
             if not(var.name in var_p1_list)]
            # var_res = list(var_p1.copy()
            # var_res.update(var_p2)

            perm_p2 = []
            resh_p2 = []
            for var in var_res:
                # Reshaping transformation and permutation
                # For p2
                v_dom_sz = len(var.domain)
                if var.name in var_p2_list:
                    resh_p2.append(v_dom_sz)
                    perm_p2.append(var_p2_list.index(var.name))
                else:
                    resh_p2.append(1)

        else:  # TODO
            # If p2 is larger than p1
            # No permutation is needed for p2
            perm_p2 = None
            # Reshaping is done to prepare p1 with respect to variables in p2 not in p1
            v1add = tuple(v1 for v1 in var_p1_list if v1 not in var_p2_list)
            resh_p2 = shape_p2.copy() + [1]*len(v1add)

            # Build the resulting variable dict by adding new p2 variables to those in p1
            var_res = var_p2.copy()
            [var_res.append(var)
             for var in var_p1
             if not(var.name in var_p2_list)]

            # var_res = var_p2.copy()
            # var_res.update(var_p1)

            perm_p1 = []
            resh_p1 = []
            for var in var_res:
                # Reshaping transformation and permutation
                # For p1
                v_dom_sz = len(var.domain)
                if var.name in var_p1_list:
                    resh_p1.append(v_dom_sz)
                    perm_p1.append(var_p1_list.index(var.name))
                else:
                    resh_p1.append(1)

        return var_res, perm_p1, resh_p1, perm_p2, resh_p2

    def prepare_fun_to_op(self, perm, resh):
        """Tool function used to apply the convenient permutation and reshaping to prepare a discrete and finite potential function (i.e. an ndarray) for arithmetic operation with another potential. The permutation and the reshaping are previously compute by DFPotential.compute_trans_to_op method.

        Syntax
        ------
        fun_new = DFPotential.prepare_fun_to_op( fun, perm, resh )

        Parameters
        ----------
        fun : ndarray of n dimensions
            The potential function to prepare.

        perm : tuple of n non negative integers
            Tuple corresponding to the permutation to apply.

        resh : tuple of m >= n non negative integers
            Tuple corresponding to the axis reshaping transformation to apply.

        Returned value
        --------------
        fun_new : ndarray of m dimensions
            New function with the same elements than in the original fun array but rearrange in order to perform an arithmetic operation with another potential.

        See also
        --------
        compute_trans_to_op
        Arithmetic operation methods, i.e. __mul__, __add__ etc.

        Example
        -------
        """
        # Permute the dimensions appropriately
        fun_new = self.fun.reshape(self.get_var_sizes())
        if not(perm is None):
            fun_new = fun_new.transpose(perm)

        # Reshape the array
        fun_new = fun_new.reshape(resh)

        return fun_new

    def __mul__(self, other):
        """Multiplication between two discrete and finite potentials. Let p1 and p2 be two discrete and finite potentials defined over multivariate domain X and Y. Let also define domains X' = X \\ Y (difference between X and Y), Y' = Y \\ X and Z = X inter Y. The result of p1*p2 is a potential defined over domain (X union Y)* = X' x Z x Y' (cartesian product) such that for all (x', z, y') in X' x Z x Y' :
        p_res(x', z, y') = (p1*p2)(x', z, y') = p1(x', z) * p2(z, y').

        In other words, this operation is a generalisation of an element-wise multiplication to array defined over different domains.

        Syntax
        ------
        p_res = p1*p2

        Parameters
        ----------
        p1 : DFPotential (self)
        The first operand of the multiplication.

        p2 : DFPotential
        The second operand of the multiplication.

        Returned value
        --------------
        p_res : DFPotential
        The result of the multiplication between p1 and p2. More precisely, p_res is defined over X x Y' if |X| >= |Y| and X' x Y otherwise, where |X| is the cardinality of set X.
        """
        var_res, perm_self, resh_self, perm_other, resh_other = \
            DFPotential.compute_trans_to_op(self.var, other.var)

        f1 = self.prepare_fun_to_op(perm_self, resh_self)
        f2 = other.prepare_fun_to_op(perm_other, resh_other)

        fun_res = f1*f2

        pot_res_name = f"{self.name}*{other.name}"\
            if not(self.name is None) and not(other.name is None)\
            else None
        return DFPotential(name=pot_res_name,
                           var=var_res,
                           fun=fun_res.flatten())

    def __imul__(self, other):
        """In-place multiplication between two discrete and finite potentials (see __mul__ method for details)

        Syntax
        ------
        p1*=p2
        """
        var_res, perm_self, resh_self, perm_other, resh_other = \
            DFPotential.compute_trans_to_op(self.var, other.var)

        f2 = other.prepare_fun_to_op(perm_other, resh_other)

        self.fun = (self.prepare_fun_to_op(perm_self, resh_self)*f2).flatten()
        self.var = var_res

        return self

    def __truediv__(self, other):
        """Division between two discrete and finite potentials. Let p1 and p2 be two discrete and finite potentials defined over multivariate domain X and Y. Let also define domains X' = X \\ Y (difference between X and Y), Y' = Y \\ X and Z = X inter Y. The result of p1*p2 is a potential defined over domain (X union Y)* = X' x Z x Y' (cartesian product) such that for all (x', z, y') in X' x Z x Y' :
        p_res(x', z, y') = (p1*p2)(x', z, y') = p1(x', z) / p2(z, y').

        In other words, this operation is a generalisation of an element-wise division to array defined over different domains.

        Syntax
        ------
        p_res = p1/p2

        Parameters
        ----------
        p1 : DFPotential (self)
        The dividend.

        p2 : DFPotential
        The divisor.

        Returned value
        --------------
        p_res : DFPotential
        The result of the division of p1 by p2. More precisely, p_res is defined over X x Y' if |X| >= |Y| and X' x Y otherwise, where |X| is the cardinality of set X.
        """
        var_res, perm_self, resh_self, perm_other, resh_other = \
            DFPotential.compute_trans_to_op(self.var, other.var)

        f1 = self.prepare_fun_to_op(perm_self, resh_self)
        f2 = other.prepare_fun_to_op(perm_other, resh_other)

        fun_res = f1/f2

        pot_res_name = f"{self.name}/{other.name}"\
            if not(self.name is None) and not(other.name is None)\
            else None
        return DFPotential(name=pot_res_name,
                           var=var_res,
                           fun=fun_res.flatten())

    def __itruediv__(self, other):
        """In-place division between two discrete and finite potentials (see __mul__ method for details)

        Syntax
        ------
        p1 /= p2
        """
        var_res, perm_self, resh_self, perm_other, resh_other = \
            DFPotential.compute_trans_to_op(self.var, other.var)

        f2 = other.prepare_fun_to_op(perm_other, resh_other)

        self.fun = (self.prepare_fun_to_op(perm_self, resh_self)/f2).flatten()
        self.var = var_res

        return self

    def sum(self, var: typing.List[str] = []):
        """This method sums some variables of the potential out.

        Syntax
        ------
        pot_new = p.sum(var)

        Parameters
        ----------
        var : sequence of variable names to sum.
        The sequence of variable to be summed out or over.

        Returned value
        --------------
        pot_new : DFPotential
        The new potential resulting of the summing operation.
        """

        if sorted(var) == sorted(self.get_var_names()):
            return self.fun.sum()

        fun_new = self.fun.reshape(self.get_var_sizes())\
                          .sum(axis=self.var2dim(var))

        pot_res_name = self.name
        pot_res_var = [pvar for pvar in self.var if not(pvar.name in var)]
        pot_res = DFPotential(name=pot_res_name,
                              var=pot_res_var,
                              fun=fun_new.flatten())

        return pot_res

    def marg(self, var: typing.List[str] = []):

        var_to_sum = [v for v in self.get_var_names()
                      if not(v in var)]

        return self.sum(var_to_sum)


class DFCPD(DFPotential):

    var_norm: typing.List[str] = pydantic.Field(
        [], description="Normalized variable names")

    counts: NumpyNDArray = pydantic.Field(
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
        obj.update(counts=self.counts.tolist())
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

        data_counts = data_df[self.get_var_names()].value_counts()

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
