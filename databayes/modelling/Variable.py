import pandas as pd
import pyAgrum as gum
import typing_extensions
import pydantic
import typing
import re
import numbers
import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class Variable(pydantic.BaseModel):
    name: str = pydantic.Field(..., description="Variable name")
    domain: list = pydantic.Field(..., description="Variable domain")
    ordered: bool = pydantic.Field(
        None, description="Indicates if domain is ordered")
    unit: str = pydantic.Field(None, description="Unit of the variable")

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):

        return (self.name == other.name) and \
            (self.domain == other.domain) and \
            (self.ordered == other.ordered) and \
            (self.unit == other.unit)


class DFVariable(Variable):
    """Discrete variable model."""

    domain_type: typing.Optional[typing_extensions.Literal["interval", "label",
                                                           "numeric"]] = pydantic.Field(None, description="Variable domain type")
    domain: list = pydantic.Field([], description="Variable domain")

    @pydantic.root_validator
    def check_obj(cls, obj):
        if len(obj["domain"]) > 0:
            if obj["domain_type"] == "interval":
                obj["domain"] = \
                    pd.IntervalIndex.from_breaks(obj["domain"]).to_list()

            if obj["domain_type"] is None:
                obj["domain_type"] = cls.detect_domain_type(obj["domain"])

            if obj["domain_type"] == "numeric":
                obj["domain"].sort()

        else:
            raise TypeError(
                "domain info must be specified to create a DFVariable")

        if obj["ordered"] is None:
            obj["ordered"] = obj["domain_type"] in ["numeric", "interval"]

        return obj

    @staticmethod
    def detect_domain_type(domain):
        if all([isinstance(lab, numbers.Real) for lab in domain]):
            return "numeric"
        elif all([isinstance(lab, pd.Interval)
                  for lab in domain]):
            return "interval"
        elif all([isinstance(lab, str) for lab in domain]):
            return "label"
        else:
            raise ValueError(f"Impossible to detect type of domain: {domain}")

    @classmethod
    def from_bins(cls, **var_specs):
        return cls(domain_type="interval",
                   **var_specs)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):

        return (self.domain_type == other.domain_type) and \
            super().__eq__(other)

    def get_bins(self):
        if self.domain_type == "interval":
            return [self.domain[0].left] + \
                [itv.right for itv in self.domain]
        else:
            return None

    def dict(self, **kwrds):
        if self.domain_type == "interval":
            kwrds.update(exclude={'domain'})
            obj = super().dict(**kwrds)
            obj.update(domain=self.get_bins())
            return obj
        else:
            return super().dict(**kwrds)
