from commis import Command
from hecate.utils.mixins import LoggableMixin


import numpy as np
import tensorflow as tf

class ExperimentCommand(LoggableMixin, Command):

    name    = "experiment"
    help    = "runs random code for development"

    args    = {
    }

    def handle(self, args):
        """
        Train on the supplied game
        """
        self.logger.info("science!")
        _experiment(self.logger)



def _experiment(logger):
    sess = tf.Session()
