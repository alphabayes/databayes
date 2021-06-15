import pydantic
import typing
import logging

from ..utils import get_subclasses

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


def create_db(**specs):

    DBBase_sub_dict = {cls.__name__: cls for cls in get_subclasses(DBBase)}

    db_classname = specs.pop("cls")
    db_cls = DBBase_sub_dict.get(db_classname)

    if db_cls is None:
        raise ValueError(f"{db_classname} is not a subclass of DBBase")

    db = db_cls(**specs)

    return db


class DBConfigBase(pydantic.BaseModel):
    database: str = pydantic.Field(None, description="DB database name")
    version: str = pydantic.Field(default=None,
                                  description="The data backend provider version")
    host: str = pydantic.Field("localhost", description="DB host address")
    port: str = pydantic.Field(default=None, description="DB host port")
    user: str = pydantic.Field(default=None, description="DB user name")
    password: str = pydantic.Field(
        default=None, description="DB user password")


class DBBase(pydantic.BaseModel):
    """Abstract data backend model."""
    name: str = pydantic.Field(None, description="Data backend id/name")
    config: DBConfigBase = pydantic.Field(default=DBConfigBase(),
                                          description="The data backend configuration")
    db: typing.Any = pydantic.Field(default=None,
                                    description="The data backend connector")

    def connect(self, logging=logging, **params):
        """DB connection function."""
        raise NotImplementedError("This function must be implemented in class {}"
                                  .format(self.__class__))

    def count(self, endpoint, filter={}, **params):
        """DB get data size after filtering."""
        raise NotImplementedError("This function must be implemented in class {}"
                                  .format(self.__class__))

    def update(self, endpoint, data=[], **params):
        """DB update data function."""
        raise NotImplementedError("This function must be implemented in class {}"
                                  .format(self.__class__))

    def put(self, endpoint, data={}, **params):
        """DB put data function."""
        raise NotImplementedError("This function must be implemented in class {}"
                                  .format(self.__class__))

    def get(self, endpoint, filter={}, **params):
        """DB get data function."""
        raise NotImplementedError("This function must be implemented in class {}"
                                  .format(self.__class__))

    def delete(self, endpoint, filter={}, **params):
        """DB delete data function."""
        raise NotImplementedError("This function must be implemented in class {}"
                                  .format(self.__class__))
