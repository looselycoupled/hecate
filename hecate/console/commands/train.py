# package.module
# module description
#
# Author:   Allen Leis <allen.leis@gmail.com>
# Created:  timestamp
#
# Copyright (C) 2017 Allen Leis
# For license information, see LICENSE
#
# ID: filename.py [] allen.leis@gmail.com $

"""
module description
"""

##########################################################################
# Imports
##########################################################################

import tensorflow as tf
from commis import Command

from hecate.utils.mixins import LoggableMixin
from hecate.agent import Agent
from hecate.utils.timer import Timer


##########################################################################
# Command
##########################################################################

class TrainCommand(LoggableMixin, Command):

    name    = "train"
    help    = "train Hecate on a given game"

    args    = {
        ('--simple'): {
            'action': 'store_true',
            'help': 'small defaults for debugging',
        },

        # Steps for training
        ('-s', '--steps'): {
            'type': int,
            'default': 10000,
            'help': 'steps for training',
        },

        # Game to train on
        '--game': {
            'default': 'Breakout-v0',
            'type': str,
            'choices': ['Breakout-v0', ],
            'help': 'game to play ("Breakout-v0", etc.)',
        }
    }

    def handle(self, args):
        """
        Train on the supplied game
        """
        with Timer() as t:
            self.logger.info("startup (TrainCommand)")

            tf.reset_default_graph()
            with tf.Session() as sess:
                params = {
                    "episodes": 100 if args.simple else 10000,
                    "steps": 10000 if args.simple else 500000,
                    "storage_path": "data",
                    "upate_target_steps": 10000
                }
                params["decay_steps"] = int(params["steps"] * .9)
                params["populate_memory_steps"] = 10000
                agent = Agent(sess, "Breakout-v0", **params)
                agent.train()

        self.logger.info("job finished in {}".format(t))
