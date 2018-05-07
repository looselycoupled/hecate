# hecate.config
# primary configuration module for Anemoi project
#
# See http://confire.readthedocs.io/en/latest/ for usage and documentation
#
# Author:   Allen Leis <allen.leis@gmail.com>
# Created:  Sat Aug 05 14:39:18 2017 -0400
#
# Copyright (C) 2017 Allen Leis
# For license information, see LICENSE
#
# ID: config.py [] allen.leis@gmail.com $

"""
primary configuration module for Anemoi project
"""

##########################################################################
## Imports
##########################################################################

import os
from datetime import datetime

from confire import Configuration
from confire import environ_setting

##########################################################################
## Storage Configuration
##########################################################################

class StorageConfig(object):
    def __init__(self, base, game, network):
        dir_name = "{}-{}-{}".format(game, network.__name__, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.base = os.path.join(base, dir_name)
        self.checkpoint = os.path.join(self.base, "checkpoints")
        self.summary = os.path.join(self.base, "summary")
        self.monitoring = os.path.join(self.base, "monitoring")

        self._create_directories()

    def _create_directories(self):
        for path in [self.checkpoint, self.summary, self.monitoring]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)


##########################################################################
## Logging Configuration
##########################################################################

class LoggingConfiguration(Configuration):
    """
    Specialized configuration system for the Python logging module. After 2.7
    Python now accepts configurations for logging from a dictionary. This
    configuration class exposes that dictionary on demand to the logging
    system. Note that it is complex because of the nested, internal configs.
    """
    version = 1
    disable_existing_loggers = False

    formatters = {
        'simple': {
            # 'format': '[%(asctime)s] %(levelname)s {%(name)s.%(funcName)s:%(lineno)d} %(message)s',
            'format': '[%(asctime)s] %(levelname)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S %z',
        }
    }

    handlers = {
        'null': {
            'level': 'DEBUG',
            'class': 'logging.NullHandler',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'logfile': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/tmp/hecate.log',
            'maxBytes': 67108864,
            'formatter': 'simple',
        }
    }

    loggers =  {
        'hecate': {
            'level': 'DEBUG',
            'handlers': ['console', 'logfile'],
            'propagagte': True,
        },
        'py.warnings': {
            'level': 'DEBUG',
            'handlers': ['console', 'logfile'],
            'propagate': True,
        }
    }

    def dict_config(self):
        """
        Returns the dictionary configuration for use with the Python logging
        module's logging.config.dictConfigClass function.
        """
        return dict(self.options())

##########################################################################
## App Configuration
##########################################################################

class DefaultConfiguration(Configuration):

    CONF_PATHS = [
        'conf/settings.{}.yaml'.format(os.getenv('ANEMOI_ENV', 'development')),
    ]
    logging = LoggingConfiguration()


settings = DefaultConfiguration.load()
