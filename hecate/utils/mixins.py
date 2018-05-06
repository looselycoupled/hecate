# hecate.utils.mixins
# Mixin classes for convencience and central configuration
#
# Author:   Allen Leis <allen.leis@gmail.com>
# Created:  Sat Aug 05 15:40:46 2017 -0400
#
# Copyright (C) 2017 Allen Leis
# For license information, see LICENSE
#
# ID: mixins.py [] allen.leis@gmail.com $

"""
Mixin classes for convencience and central configuration
"""

##########################################################################
# Imports
##########################################################################

import os
import sys
import logging
import logging.config

try:
    from hecate.config import settings
except ImportError as e:
    sys.path.insert(0, os.getcwd())
    from hecate.config import settings

##########################################################################
# Classes
##########################################################################

class LoggableMixin(object):
    """
    Placeholder logging mixin for eventual configuration or wrapping of global
    logging features.

    Possible TODO features include:
        self.logger.error() should email admins
    """
    def __init__(self, *args, **kwargs):
        logging.config.dictConfigClass(settings.logging.dict_config()).configure()
        self.logger = logging.getLogger('hecate')
        super(LoggableMixin, self).__init__(*args, **kwargs)


##########################################################################
# Execution
##########################################################################

if __name__ == '__main__':
    obj = LoggableMixin()
    obj.logger.debug("Test for debug...")
    obj.logger.info("Test for info...")
    obj.logger.warning("Test for warning...")
    obj.logger.error("Test for error...")
    obj.logger.critical("Test for critical...")
    try:
        1/0
    except Exception as e:
        obj.logger.exception(e)
