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
from hecate import VALID_GAMES

##########################################################################
# Command
##########################################################################

class TrainCommand(LoggableMixin, Command):

    name    = "train"
    help    = "train Hecate on a given game"

    args    = {
        ('-v', '--verbose'): {
            'action': 'store_true',
            'help': 'verbose output',
        },

        ('--simple'): {
            'action': 'store_true',
            'help': 'small defaults for debugging',
        },

        # Steps for training
        ('-s', '--max_steps'): {
            'type': int,
            'default': 500000,
            'help': 'steps for training',
        },

        # Steps for training
        ('-m', '--model_year'): {
            'type': int,
            'default': 2013,
            'choices': (2013, 2015),
            'help': 'year of published model',
        },

        # Game to train on
        '--game': {
            'default': 'Breakout-v0',
            'type': str,
            'choices': VALID_GAMES,
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
                    "max_steps": 40000 if args.simple else args.max_steps,
                    "storage_path": "data",
                    "update_target_steps": 10000,
                    "verbose": args.verbose,
                    "model_year": args.model_year,
                }
                params["decay_steps"] = int(params["max_steps"] * .1)
                params["populate_memory_steps"] = 20000 if args.simple else min(int(args.max_steps * .5), 50000)
                agent = Agent(sess, args.game, **params)
                agent.train()

        self.logger.info("job finished in {}".format(t))
