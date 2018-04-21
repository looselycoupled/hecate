# utils.decorators
# Method and function utilities and custom decorators and descriptors.
#
# Author:  Allen Leis
# Created:
#
# ID: decorators.py [] allen.leis@gmail.com $

"""
Method and function utilities and custom decorators and descriptors.
"""

##########################################################################
## Imports
##########################################################################

import warnings

from functools import wraps


##########################################################################
## Decorators
##########################################################################

def memoized(fget):
    """
    Return a property attribute for new-style classes that only calls its
    getter on the first access. The result is stored and on subsequent
    accesses is returned, preventing the need to call the getter any more.

    Use del to delete the cache and force the property to recompute the cahced
    value by calling fget again.
    """
    attr_name = '_{0}'.format(fget.__name__)

    @wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)

    def fdel(self):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    return property(fget=fget_memoized, fdel=fdel)


def filterwarnings(action="ignore", category=UserWarning):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(action, category=category)
                return func(*args, **kwargs)
        return wrapper
    return decorator
