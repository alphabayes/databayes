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


class DiscreteVariable(pydantic.BaseModel):
    """Discrete variable model."""

    name: str = pydantic.Field(None, description="Variable name")
    domain_type: typing.Optional[typing_extensions.Literal["interval", "label",
                                                           "numeric"]] = pydantic.Field("label", description="Variable domain type")
    domain: list = pydantic.Field([], description="Variable domain")
    bins: typing.List[float] = pydantic.Field(
        [], description="Domain bins for numeric variable")
    unit: str = pydantic.Field(None, description="Unit of the variable")

    @pydantic.root_validator
    def check_domain(cls, obj):
        if len(obj["domain"]) > 0:
            obj["domain_type"] = cls.detect_domain_type(obj["domain"])
            if obj["domain_type"] == "interval":
                obj["bins"] = cls.labels_to_bins(obj["domain"])
            elif obj["domain_type"] == "numeric":
                obj["domain"].sort()

        elif len(obj["bins"]) > 0:

            obj["domain_type"] = "interval"
            obj["domain"] = cls.bins_to_labels(obj["bins"])
        # Relax this constraints
        # else:
        #     raise TypeError(f"labels or bins must be specified to create a DiscreteVariable")
        return obj

    @staticmethod
    def detect_domain_type(domain):

        if len(domain) == 0:
            return "label"

        if all([isinstance(lab, numbers.Real) for lab in domain]):
            return "numeric"

        if all([isinstance(lab, str) for lab in domain]):
            # For now the check is very simple as we only test
            # the beginning of the labels to find caracter [ or (
            # Advanced tests could be implemented to check the entire
            # label and take into account specificities like infinte value
            if all([not(re.match("^[\[(]", lab) is None) for lab in domain]):
                return "interval"

            if all([lab.isnumeric() for lab in domain]):
                return "numeric"

            return "label"

        raise ValueError(f"Impossible to detect type of domain: {domain}")

    @staticmethod
    def labels_to_bins(labels):
        bins = [float(labels[0].split(",")[0][1:]),
                float(labels[0].split(",")[1][0:-1])]
        bins += [float(lab.split(",")[1][0:-1])
                 for lab in labels[1:]]
        return bins

    @staticmethod
    def bins_to_labels(bins):
        return pd.IntervalIndex.from_breaks(bins)\
                               .astype(str)\
                               .to_list()

    def pyagrum_init_var(self):

        bn_var = gum.LabelizedVariable(self.name, "", 0)
        [bn_var.addLabel(l) for l in self.domain]

        return bn_var
