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

class PlayCommand(LoggableMixin, Command):

    name    = "play"
    help    = "play Hecate on a given game"

    args    = {
        # Steps for play
        'episodes' : {
            'type': int,
            'default': 10,
            'help': 'eps for play',
        },

        # Network to use
        ('-m', '--model_year'): {
            'type': int,
            'default': 2013,
            'choices': (2013, 2015),
            'help': 'year of published model',
        },

        # Game to play
        '--game': {
            'default': 'Breakout-v0',
            'type': str,
            'choices': VALID_GAMES,
            'help': 'game to play ("Breakout-v0", etc.)',
        },

        # Checkpoint to load for play
        '--record_frequency': {
            'type': int,
            'default': 1,
            'help': 'how often to record videos',
        },

    }

    def handle(self, args):
        with Timer() as t:
            self.logger.info("startup (PlayCommand)")

            tf.reset_default_graph()
            with tf.Session() as sess:
                params = {
                    "max_steps": 0,
                    "model_year": args.model_year,
                    'decay_steps': None,
                    'populate_memory_steps': 0,
                    'update_target_steps': 0,
                }
                agent = Agent(sess, args.game, **params)
                agent.play(args.episodes, args.record_frequency)

        self.logger.info("job finished in {}".format(t))
