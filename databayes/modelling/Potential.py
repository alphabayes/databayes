import pydantic
from .DiscreteVariable import DiscreteVariable
import pandas as pd
import typing
import pkg_resources

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


class DFPotential(pd.Series):

    @property
    def _constructor(self):
        return DFPotential


class DFPotential_2(pydantic.BaseModel):
    __slots__ = ('_values',)

    variables: typing.Dict[str, typing.Optional[DiscreteVariable]] = pydantic.Field(
        {}, description="Discrete variable specification")
    values: typing.List[float] = pydantic.Field(
        [], description="Values of potential for each combinaison of domain labels")

    @pydantic.validator("variables")
    def normalize_variables(cls, variables):
        for var_name, var in variables.items():
            if var is None:
                variables[var_name] = DiscreteVariable()
            variables[var_name].name = var_name
        return variables

    def __init__(self, **data: typing.Any):
        super().__init__(**data)

        self.update_values()

    def __setattr__(self, attr, value):
        if attr in self.__slots__:
            object.__setattr__(self, attr, value)
        else:
            super().__setattr__(self, attr, value)

    def add_variable(self, **var_specs):
        self.update_variable(**var_specs)

    def update_variable(self, **var_specs):
        """ Update variables specs."""
        new_var = DiscreteVariable(**var_specs)

        # Just add variable
        self.variables[new_var.name] = new_var

        self.update_values()

    def update_values(self):

        var_domain = list(self.variables.keys())

        if len(var_domain) > 0:
            var_domain_labels = \
                pd.MultiIndex.from_product([self.variables[v].domain for v in var_domain],
                                           names=var_domain)

            # Store CCT in cct attribute
            self._values = pd.Series(0, index=var_domain_labels, name="values")
            self.values = self._values.to_list()
        else:
            self._values = pd.Series(name="values")
            self.values = []

    def get_values(self, index=(slice(None),)):

        return self._values.loc(axis=0)[index]

    def set_values(self, index=(slice(None)), values=[]):
        pass

    def __str__(self):

        return self.get_values().to_frame().to_string()
