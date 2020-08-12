import pyAgrum.lib.pretty_print as gum_pp
import pyAgrum as gum
import databayes.modelling.DiscreteDistribution as dd
import databayes.modelling.DiscreteVariable as dv
import databayes.modelling.core as dcm
from termcolor import colored
import numpy as np
import pandas as pd
import tqdm
import typing_extensions
import pydantic
import typing
import copy
import re
import pkg_resources
import warnings

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


_CPT_HTML_TEMPLATE = """
<div>
Variable {var_name}
</div>
<div style="float:left; margin:10px">
Conditional count table
{cct}
</div>
<div style="float:left; margin:10px">
Conditional probability table
{cpt}
</div>
"""

_BN_HTML_TEMPLATE = """
<div>
Model {name}
</div>
<div>
{cpt}
</div>
"""


class BayesianNetwork(pydantic.BaseModel):
    __slots__ = ('bn',)

    name: str = pydantic.Field("", description="Bayes net title")
    variables: typing.Dict[str, typing.Optional[dv.DiscreteVariable]] = pydantic.Field(
        {}, description="Discrete variable specification")
    parents: typing.Dict[str, typing.List[str]] = pydantic.Field(
        {}, description="Dictionary giving for each variable the list of their parents")
    cct: typing.Dict[str, list] = pydantic.Field(
        {}, description="Conditional counts for each variable according their parents")
    backend: typing.Optional[typing_extensions.Literal["pyagrum"]] = pydantic.Field(
        "pyagrum", description="Bayesian network backend")

    @pydantic.validator("variables")
    def normalize_variables(cls, variables):

        for var_name, var in variables.items():
            if var is None:
                variables[var_name] = dv.DiscreteVariable()
            variables[var_name].name = var_name
        return variables

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        # Init BN backend
        backend_bn_method = \
            getattr(self, self.backend + "_init_bn", None)
        if callable(backend_bn_method):
            backend_bn_method()
        else:
            raise ValueError(
                f"Bayesian network backend '{self.backend}' not supported")

        self.init_backend()

    def init_backend(self):
        # Add variables to backend
        for var_name, var in self.variables.items():
            self.bn_update_variable(var_name)

        # Add parents to backend
        for var_name, var in self.parents.items():
            self.bn_update_parents(var_name)

        # Update CCT
        for var_name in self.variables.keys():
            self.update_cct(var_name)

    def __str__(self):

        cct_cpt_sep = colored("     |     ", "blue",
                              attrs=["bold"])

        var_strlist = []
        for var_name, var in self.variables.items():

            var_header_str = colored(f"Variable: {var_name}", "blue",
                                     attrs=['bold'])

            cct_str = self.get_cct(var_name, transpose=True).to_string()
            cpt_str = self.get_cpt(var_name, transpose=True).to_string()

            cct_strlist = cct_str.split("\n")
            cpt_strlist = cpt_str.split("\n")

            cct_cpt_strlist = [cct + cct_cpt_sep + cpt
                               for cct, cpt in zip(cct_strlist, cpt_strlist)]

            cct_cpt_str = "\n".join(cct_cpt_strlist)

            var_str = "\n".join([var_header_str,
                                 cct_cpt_str])

            var_strlist.append(var_str)

        bn_str = "\n\n".join(var_strlist)

        return bn_str

    def to_html(self, transpose=True, filename=None):

        cpt_html_list = []

        for var_name in self.variables.keys():
            cpt_html_list.append(self.cpt_to_html(
                var_name, transpose=transpose))

        cpt_html = "\n".join(cpt_html_list)
        bn_html = _BN_HTML_TEMPLATE.format(
            name=self.name,
            cpt=cpt_html)

        if not(filename is None):
            with open(filename, "w") as f:
                f.write(bn_html)

        return bn_html

    def cpt_to_html(self, var_name, transpose=True, filename=None):

        cct = self.get_cct(var_name, transpose=transpose)
        cpt = self.get_cpt(var_name, transpose=transpose)

        cpt_html = _CPT_HTML_TEMPLATE.format(
            var_name=var_name,
            cct=cct.to_html(),
            cpt=cpt.to_html())

        if not(filename is None):
            with open(filename, "w") as f:
                f.write(cpt_html)

        return cpt_html

    # Required when using slot attributes
    def __setattr__(self, attr, value):
        if attr in self.__slots__:
            object.__setattr__(self, attr, value)
        else:
            super(self.__class__, self).__setattr__(attr, value)

    def is_num_equal(self, other, check_cct=True):

        # Check variables
        if list(self.variables.keys()) != list(other.variables.keys()):
            return False

        for var_self, var_other in zip(self.variables.values(),
                                       other.variables.values()):

            if var_self.domain != var_other.domain:
                return False

            if var_self.domain_type != var_other.domain_type:
                return False

            if (self.get_cpt(var_self.name) != other.get_cpt(var_other.name)).any(None):
                return False

            if (self.get_cct(var_self.name) != other.get_cct(var_other.name)).any(None):
                return False

            if not(self.backend_is_num_equal(other)):
                return False

        return True

    def backend_is_num_equal(self, other):

        if self.backend != other.backend:
            return False

        backend_is_num_equal_method = \
            getattr(self, self.backend + "_is_num_equal", None)
        if callable(backend_is_num_equal_method):
            return backend_is_num_equal_method(other)
        else:
            raise ValueError(
                f"Bayesian network backend '{self.backend}' not supported")

    def pyagrum_is_num_equal(self, other):

        if self.bn.names() != other.bn.names():
            return False

        for var in self.bn.names():

            if (self.bn.cpt(var)[:] != other.bn.cpt(var)[:]).any():
                return False

        return True

    def pyagrum_init_bn(self):
        self.bn = gum.BayesNet(self.name)

    def add_variable(self, **var_specs):
        self.update_variable(**var_specs)

    def bn_update_variable(self, var_name):
        backend_bn_method = \
            getattr(self, self.backend + "_update_variable", None)
        if callable(backend_bn_method):
            backend_bn_method(var_name)
        else:
            raise ValueError(
                f"Bayesian network backend '{self.backend}' not supported")

    def pyagrum_update_variable(self, var_name):
        bn_var = self.variables[var_name].pyagrum_init_var()
        if var_name in self.bn.names():
            self.bn.erase(var_name)

        # Pyagrum does not support empty domain variables
        if bn_var.domainSize() > 0:
            self.bn.add(bn_var)

    def remove_parent(self, var_name):
        self.add_parents(var_name)

    def add_parents(self, var_name, parents=[]):

        # TODO: We have a problem here !
        if var_name in self.variables.keys():

            if all([pa in self.variables.keys()
                    for pa in parents]):

                self.parents[var_name] = parents
                # Update parents info in backend
                self.bn_update_parents(var_name)
                # Update variables numerical specs
                self.update_cct(var_name)

            else:
                raise ValueError(
                    "Parent variables must be part of BN variables")

        else:
            raise ValueError("Variable {var_name} is not part of BN variables")

    def update_variable(self, **var_specs):
        """ Update variables specs."""
        new_var = dv.DiscreteVariable(**var_specs)

        # Just add variable
        self.variables[new_var.name] = new_var

        self.bn_update_variable(new_var.name)
        self.update_cct(new_var.name)

        # Update case: the variable already exists and has children
        var_children = [v for v in self.variables.keys()
                        if new_var.name in self.parents.get(v, [])]

        for vc in var_children:
            self.bn_update_variable(vc)
            self.update_cct(vc)

    def bn_update_parents(self, var_name):
        """ Update parents specs in backend."""

        backend_bn_method = \
            getattr(self, self.backend + "_update_parents", None)
        if callable(backend_bn_method):
            backend_bn_method(var_name)
        else:
            raise ValueError(
                f"Bayesian network backend '{self.backend}' not supported")

    def pyagrum_update_parents(self, var_name):
        for parent in self.parents.get(var_name, []):

            if (var_name in self.bn.names()) and \
               (parent in self.bn.names()):
                self.bn.addArc(parent, var_name)

    def update_cct(self, var_name):

        parents_var = self.parents.get(var_name, [])
        var_domain = [var_name] + parents_var

        var_domain_labels = \
            pd.MultiIndex.from_product([self.variables[v].domain for v in var_domain],
                                       names=var_domain)

        # Init CCT as a DataFrame
        cct_df = pd.Series(0, index=var_domain_labels, name="count")

        # Store CCT in cct attribute
        self.cct[var_name] = \
            cct_df.to_frame()\
            .reset_index()\
            .to_dict("records")

        # Update backend parameters after learning process
        self.update_cpt_params(var_name)

    def init_from_dataframe(self, df, add_data_var=False):

        for var_name, dfs in df.items():
            # Do not add variable from data columns not defined in
            # variables dictionary
            if not(add_data_var) and not(var_name in self.variables.keys()):
                continue

            if dfs.dtype.name == "category":
                if dfs.cat.categories.dtype.name == "interval":
                    var_domain_type = "interval"
                    var_domain = [str(it)
                                  for it in dfs.cat.categories.to_list()]
                else:
                    var_domain_type = "label"
                    var_domain = dfs.cat.categories.to_list()
            else:
                # Default behaviour: force categorical data
                var_domain_type = "label"
                var_domain = dfs.astype(str).astype(
                    "category").cat.categories.to_list()

            var = dv.DiscreteVariable(name=var_name,
                                      domain=var_domain,
                                      domain_type=var_domain_type)

            self.variables[var_name] = var

        self.init_backend()

    def adapt_data(self, data):
        """Utility method to ensure series has well formatted categorical data, i.e. string labels.
        """

        # parents_var = self.parents.get(var_name, [])
        # var_dim = [var_name] + parents_var
        data_new = {}

        data_var_list = [var for var in self.variables.keys()
                         if var in data.columns]
        # Check if input dataframe has consistent catagorical variables
        for var_name, data_var_s in data[data_var_list].items():

            data_var_s = data[var_name]

            if data_var_s.dtype.name != "category":
                # Try to transform it
                cat_type = \
                    pd.api.types.CategoricalDtype(categories=self.variables[var_name].domain,
                                                  ordered=self.variables[var_name].domain_type != "label")

                data_var_s = data_var_s.astype(str).astype(cat_type)

            if data_var_s.cat.categories.dtype.name == "interval":
                series_lab = [str(it)
                              for it in data_var_s.cat.categories.to_list()]
            else:
                series_lab = data_var_s.cat.categories.to_list()

            if self.variables[var_name].domain != series_lab:
                err_msg = f"Domain of variable {var_name}: {self.variables[var_name].domain}\n"
                err_msg += f"Series categories: : {series_lab}\n"
                err_msg += f"Inconsistency detected"
                raise ValueError(err_msg)

            if data_var_s.cat.categories.dtype.name == "interval":

                data_var_s.cat.rename_categories(
                    self.variables[var_name].domain, inplace=True)

            data_new[var_name] = data_var_s

        return pd.DataFrame(data_new, index=data.index)

    def fit(self, data,
            update_fit=False,
            update_decay=0,
            logger=None):

        for var_name in self.variables.keys():
            var_domain = [var_name] + self.parents.get(var_name, [])

            var_not_in_data = \
                [var for var in var_domain if not(var in data.columns)]
            if len(var_not_in_data) > 0:
                warnings.warn(f"Skip variable {var_name} fitting: domain [{var_not_in_data}] not in data",
                              RuntimeWarning)
                continue
            df_cur = data[var_domain]

            self.fit_cpt(data=df_cur,
                         var_name=var_name,
                         update_fit=update_fit,
                         update_decay=update_decay,
                         logger=logger)

    def fit_cpt(self, data, var_name,
                update_fit=False,
                update_decay=0,
                logger=None):
        """
        This function aims to compute the joint counts associated to CPT parameters from a Pandas
        dataframe.

        Parameters
        - data: a Pandas DataFrame consisting only of categorical variables.
        - var_name: the variable name associated to the CPT to be fitted.
        - update_fit: indicates if current joint counts has to be updated with new observation.
        - update_decay: decay coef in [0,1] to reduce importance of current count comparared to new fitted data. 0 means that old data is as same weight than new data. Otherwise count_update = count_new_fit + (1-decay)*count_old. Note that decay coef == 1 is equivalent to set update_fit == False.

        Note: this method is an adaptation of codes found at http://www-desir.lip6.fr/~phw/aGrUM/officiel/notebooks/
        """
        if not(logger is None):
            logger.debug("- Learn CPT {0}\n".format(var_name))

        parents_var = self.parents.get(var_name, [])
        var_dim = [var_name] + parents_var

        data = self.adapt_data(data)

        # # Check if input dataframe has consistent catagorical variables
        # for var in data[var_domain].columns:
        #     # data_var_type = self.find_consistent_series_type(data[var])
        #     # data[var] = data[var].astype(str).astype(data_var_type)
        #     self.adapt_series_categories(data[var], inplace=True)

        # ipdb.set_trace()

        # Compute counts from input data
        index_name = data.index.name if not(data.index.name is None) \
            else "index"
        cct_cur_df = data[var_dim].reset_index()\
            .groupby(var_dim)[index_name]\
            .count()\
            .rename("count")

        if len(self.cct.get(var_name, [])) > 0 and update_fit:
            cct_df = cct_cur_df + \
                (1 - update_decay)*self.get_cct(var_name, flatten=True)

            self.cct[var_name] = \
                cct_df.to_frame()\
                      .reset_index()\
                      .to_dict("records")

        else:
            # Erase cpt counts with new counts
            self.cct[var_name] = \
                cct_cur_df.to_frame()\
                          .reset_index()\
                          .to_dict("records")

        # Update backend parameters after learning process
        self.update_cpt_params(var_name)

    def get_cct(self, var_name, transpose=False, flatten=False):
        """
        This method returns the variable count table as a dataframe or a serie if flatten is demanded.
        """
        parents_var = self.parents.get(var_name, [])
        var_domain = [var_name] + parents_var

        cct_df = pd.DataFrame(self.cct[var_name])
        # Set categorical labels for counts columns with respect to
        # variable labels
        for var in cct_df[var_domain].columns:
            cat_type = \
                pd.api.types.CategoricalDtype(categories=self.variables[var].domain,
                                              ordered=self.variables[var].domain_type != "label")

            cct_df[var] = cct_df[var].astype(cat_type)

        cct_df = pd.pivot_table(data=cct_df,
                                index=var_name,
                                columns=parents_var,
                                values="count")

        if len(parents_var) == 0:
            # To have unified output change unique columns name as "" since
            # there is no parents
            cct_df.rename(columns={"count": ""}, inplace=True)

        if flatten:
            # When no parent, cpt_df.index is a CategoricalIndex
            # which cannot be reorder_levels
            if len(parents_var) > 0:
                cct_df = cct_df.stack(parents_var).reorder_levels(var_domain)
            else:
                cct_df = cct_df[""]

            cct_df.name = "count"

        if transpose:
            cct_df = cct_df.transpose()

        return cct_df

    def get_cpt(self, var_name,
                nan_management="uniform",
                flatten=False,
                transpose=False):
        """
        apriori_coef: Parameter representing the apriori weight during the fitting process compared to data. if apriori_coef is a non negative real number :
          - the higher it is, the closer to the apriori distribution the resulting configuration distribution will be.  
          - the lower it is, the closer to the distribution fitted by data the resulting configuration distribution will be.  
          User can also pass a string associated to an apriori coefficient strategy. Possible values are
          - smart: in this case, the apriori coefficient is set equal to 1/nb_data_conf if nb_data_conf > 0 else 1 where nb_data_conf is the number of data observed for a given configuration.

        apriori_dist: shape of the apriori distribution. Possible values are: 'uniform'. Passing None to this parameter disables apriori consideration in the fitting process.

        apriori_data_threshold: apply apriori for a conditional distribution if the number of oberserved corresponding configurations is lower or equal than this parameter.

        Notes:
        - To avoid numerical problems like joint probabilities at 0 during inference process, it is recommanded to 
          ensure non-zeros probabilities for each modalities of each variable | parents.
          => TODO: implement the smart a priori system to do that
        """

        cct_df = self.get_cct(var_name)

        parents_var = self.parents.get(var_name, [])
        var_domain = [var_name] + parents_var

        var_norm_size = len(self.variables[var_name].domain)

        # if len(parents_var) == 0:
        #     # To have unified output change unique columns name as "" since
        #     # there is no parents
        #     cpt_df = cct_df/cct_df.sum()
        #     cpt_df.name = "prob"
        # else:
        # Normalization
        cpt_df = cct_df.div(cct_df.sum(axis=0), axis=1)

        if flatten:
            # When no parent, cpt_df.index is a CategoricalIndex
            # which cannot be reorder_levels
            if len(parents_var) > 0:
                cpt_df = cpt_df.stack(parents_var, dropna=False)

                # NOTE: Theoretically reorder level is useless
                # but we keep it to be sure
                cpt_df = cpt_df.reorder_levels(var_domain)
            else:
                cpt_df = cpt_df[""]

            cpt_df.name = "prob"

        # Manage NaN
        if nan_management == "uniform":
            cpt_df.fillna(1/var_norm_size,
                          inplace=True)
        else:
            cpt_df.fillna(0, inplace=True)

        if transpose:
            cpt_df = cpt_df.transpose()

        return cpt_df

    # # A priori management
    # if not(apriori_dist is None):
    #     # Select apriori distribution
    #     apriori_joint_arr = pd.np.zeros(joint_counts.shape)
    #     if apriori_dist == "uniform":
    #         apriori_joint_arr[:] = 1/apriori_joint_arr.shape[0]
    #     else:
    #         err_msg = "apriori distribution {0} is not supported. Possible values are : 'uniform'\n".format(apriori_dist)
    #         raise ValueError(err_msg)

    #     # Build the apriori coefficient array
    #     apriori_coef_arr = pd.np.ones(cond_counts.shape)
    #     if isinstance(apriori_coef, str):
    #         if apriori_coef == "smart":
    #             if len(cond_counts.shape) == 0:
    #                 apriori_coef_arr = 1/cond_counts if cond_counts > 0 else 1.0
    #             else:
    #                 idx_cond_count_sup_0 = pd.np.where(cond_counts > 0)
    #                 apriori_coef_arr = pd.np.ones(cond_counts.shape)
    #                 apriori_coef_arr[idx_cond_count_sup_0] = 1/cond_counts[idx_cond_count_sup_0]
    #         else:
    #             err_msg = "apriori coef {0} is not supported. Possible values are : 'smart' or non negative value\n".format(apriori_coef)
    #             raise ValueError(err_msg)
    #     else:
    #         if len(cond_counts.shape) == 0:
    #             apriori_coef_arr = abs(apriori_coef)
    #         else:
    #             apriori_coef_arr[:] = abs(apriori_coef)

    #     # Check coordinate that need apriori
    #     if len(cond_counts.shape) == 0:
    #         if cond_counts > apriori_data_threshold: apriori_coef_arr = 0.
    #     else:
    #         apriori_counts_idx = cond_counts <= apriori_data_threshold
    #         apriori_coef_arr[~apriori_counts_idx] = 0.

    #     # Update joint and cond counts
    #     joint_counts += apriori_joint_arr*apriori_coef_arr
    #     cond_counts = joint_counts.sum(axis=0)

    # # Normalization of counts to get a consistent CPT
    # # Note: np.nan_to_num is used only in the case where no apriori is requested to force nan value to 0
    # #       => this is of course highly unsafe to work in this situation as CPTs may not sum to 1 for all configurations
    # bn.cpt(var_name)[:] = pd.np.nan_to_num((joint_counts/cond_counts).transpose().reshape(*domains)

    def update_cpt_params(self, var_name):
        update_cpt_params_method = \
            getattr(self, self.backend + "_update_cpt_params", None)
        if callable(update_cpt_params_method):
            return update_cpt_params_method(var_name)
        else:
            raise ValueError(
                f"Bayesian network backend '{self.backend}' not supported")

    def pyagrum_update_cpt_params(self, var_name):

        if not(var_name in self.bn.names()):
            return

        cpt = self.get_cpt(var_name, transpose=True)

        # Reshape
        parents_var = self.parents.get(var_name, [])
        var_domain = parents_var + [var_name]
        cpt_domain_size = [len(self.variables[var].domain)
                           for var in var_domain]

        cpt_np = cpt.to_numpy()\
                    .reshape(cpt_domain_size)

        # Then transpose axis if needed
        cpt_nb_dim = len(var_domain)
        if cpt_nb_dim > 1:
            cpt_transpose = list(
                reversed(range(0, cpt_nb_dim - 1))) + [cpt_nb_dim - 1]
            cpt_np = cpt_np.transpose(cpt_transpose)

        self.bn.cpt(var_name)[:] = cpt_np

    def predict(self, data, var_targets, map_k=0, probs=True, progress_mode=False, logger=None):
        """
        This function is used to predict the value of a target variable from observations 
        using a bayesian network model. 

        Inputs:
        - data: the data containing the observations used to predict the target variable 
          as a =pandas.DataFrame= object
        - var_targets: the name of the target variable as a =str= object
        - probs: indicate if posterior probabilities are returned as a DiscreteDistribution object
        - map_k: indicate if the k most probable labels are returned. (default map_k == 0)

        Returns:
        - a numpy.array containing the predictions of the target variables maximising the 
          maximum a posteriori criterion 
        - a numpy.array containing the posterior probability distribution of the target 
          variable given each observation in data.
        """

        if isinstance(data, pd.core.frame.DataFrame):
            data_pred = self.adapt_data(data)
        else:
            raise ValueError(f"Input data must be a Pandas DataFrame")

        predict_method = \
            getattr(self, self.backend + "_predict", None)
        if callable(predict_method):
            return predict_method(data=data_pred,
                                  var_targets=var_targets,
                                  map_k=map_k,
                                  probs=probs,
                                  progress_mode=progress_mode,
                                  logger=logger)
        else:
            raise ValueError(
                f"Bayesian network backend '{self.backend}' not supported")

    def pyagrum_predict(self, data, var_targets,
                        map_k=0,
                        probs=True,
                        progress_mode=False,
                        logger=None):

        # Initialize the inference engine
        inf_bn = gum.LazyPropagation(self.bn)
        inf_bn.setTargets(set(var_targets))

        # target_size = len(self.variables[var_target].domain)
        # target_dom = self.variables[var_target].domain
        nb_data = len(data)

        # TODO: Use DiscreteDistribution
        pred_res = \
            {tv: {"scores": dd.DiscreteDistribution(index=data.index,
                                                    **self.variables[tv].dict()),
                  "comp_ok": pd.Series(True, index=data.index)}
             for tv in var_targets}
        for data_id, data_cur in tqdm.tqdm(data.iterrows(),
                                           desc="Inference",
                                           unit=" predictions",
                                           disable=not(progress_mode)):
            # Set the evidence
            inf_bn.setEvidence(data_cur.to_dict())
            # Run inference
            inf_bn.makeInference()
            # Compute posterior probability of target variables
            for tv in var_targets:
                try:
                    pred_res[tv]["scores"].loc[data_id] = inf_bn.posterior(
                        tv).toarray()
                except Exception as e:
                    pred_res[tv]["comp_ok"].loc[data_id] = False

            # Erase evidence
            inf_bn.eraseAllEvidence()

        # TODO: Use DiscreteDistribution get map method
        for tv in var_targets:
            if map_k > 0:
                # ipdb.set_trace()
                pred_res[tv]["map"] = \
                    pred_res[tv]["scores"].get_map(map_k)

        return pred_res

    # def update_inference_engine(self):
    #     # TODO: Make this method
    #     # TODO: Adapt to check if variable type is consistent
    #     # with bn structure in the backend

    #     # # CPT dimensions (size)
    #     # # First dimensions correspond to parent, last dimension is for normalized variable
    #     # domains_size = [self.bn.variable(vn).domainSize()
    #     #                 for vn in self.bn.cpt(var_name).var_names]

    #     # Check if df has consist catagorical variables
    #     for var, data in df[[var_name] + parents].items():
    #         if data.dtype.name != "label":
    #             err_msg = f"Variable {var} is not categorical, but of type {data.dtype.name}"
    #             raise TypeError(err_msg)

    #         if data.cat.categories.dtype.name == "interval":
    #             data_lab = [str(it) for it in data.cat.categories.to_list()]
    #         else:
    #             data_lab = data.cat.categories.to_list()

    #         bn_var_lab = list(self.bn.variable(var).domain())
    #         if bn_var_lab != data_lab:
    #             err_msg = f"Domain of variable {var}: {bn_var_lab}\n"
    #             err_msg += f"Data categories: : {data_lab}\n"
    #             err_msg += f"Inconsistency detected"
    #             raise ValueError(err_msg)


class BayesianNetworkFitParameters(dcm.FitParametersBase):
    update_fit: bool = pydantic.Field(
        False, description="Indicates if fitting process will update current CPT parameters during the fitting process (update_fit=True) or erase current CPT parameters with results of the last fitting process (update_fit=False)")
    update_decay: float = pydantic.Field(0, description="Fitting Update decay")


class BayesianNetworkModel(dcm.MLModel):

    type: str = pydantic.Field(
        "BayesianNetworkModel", description="Type of the model")

    model: BayesianNetwork = pydantic.Field(
        BayesianNetwork(), description="Bayesian network object")

    fit_parameters: BayesianNetworkFitParameters = pydantic.Field(BayesianNetworkFitParameters(),
                                                                  description="Bayesian network object")

    @pydantic.validator('type')
    def check_type(cls, val):
        if val != "BayesianNetworkModel":
            raise ValueError("Not BayesianNetworkModel object")
        return val

    def change_var_features(self, removed_variables, inplace=False):
        """
        Delete links and variables
        Returns the BayesianNetworkModel
        """

        new_variables = dict(self.model.variables)
        new_parents = dict(self.model.parents)
        for var_name in removed_variables:
            new_variables.pop(var_name)
            new_parents.pop(var_name)

        for var_name_child, parent_list in self.model.parents.items():
            for var_name in removed_variables:
                if var_name in parent_list:
                    new_parents[var_name_child].remove(var_name)

        model_specs = {'variables': new_variables,
                       'parents': new_parents,
                       'name': self.model.name,
                       'backend': self.model.backend}
        # We must reset the cct, because the model structure changed

        if inplace:
            self.model = BayesianNetwork(**model_specs)
            self.new_features(removed_variables, inplace=True)
            return self
        else:
            new_model = copy.deepcopy(self)
            new_model.model = BayesianNetwork(**model_specs)
            new_model.var_features = self.new_features(removed_variables)
            return new_model
