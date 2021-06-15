# -*- coding: utf-8 -*-
import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}

if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


def get_subclasses(cls, recursive=True):
    """ Enumerates all subclasses of a given class.

    # Arguments
    cls: class. The class to enumerate subclasses for.
    recursive: bool (default: True). If True, recursively finds all sub-classes.

    # Return value
    A list of subclasses of `cls`.
    """
    sub = cls.__subclasses__()
    if recursive:
        for cls in sub:
            sub.extend(get_subclasses(cls, recursive=True))
    return sub
