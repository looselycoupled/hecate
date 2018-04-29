import os

import gym
import tensorflow as tf
import numpy as np


from hecate.image import wrangle_image
from hecate.utils.mixins import LoggableMixin


def _get_environment(name="Breakout-v0"):
    env = gym.envs.make(name)
    return env


class StorageConfig(object):
    def __init__(self, base, game):
        self.base = os.path.join(base, game)
        self.checkpoint = os.path.join(self.base, "checkpoints")
        self.summary = os.path.join(self.base, "summary")
        self.monitoring = os.path.join(self.base, "monitoring")

        self._create_directories()

    def _create_directories(self):
        for path in [self.checkpoint, self.summary, self.monitoring]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)



class Model(LoggableMixin):

    def __init__(self):
        pass

    def predict(self):
        pass




class Agent(LoggableMixin):

    def __init__(self,
        session,
        game,
        episodes=1000,
        steps=10000,
        decay_steps=10000,
        storage_path="data",
    ):
        super(Agent, self).__init__()
        self.session = session
        self.game = game
        self.dirs = StorageConfig(storage_path, game)

        self.model = tf.Variable(0, name='foo', trainable=False)
        self.saver = tf.train.Saver()

        self._load_checkpoint()

    def _log_configuration(self):
        self.logger.info("AGENT CONFIG\n" + "="*80)
        self.logger.info("game: {}".format(self.game))
        self.logger.info("action_space: {}".format(self.action_space))
        self.logger.info("checkpoint dir: {}".format(self.dirs.checkpoint))
        self.logger.info("monitoring dir: {}".format(self.dirs.monitoring))
        self.logger.info("summary dir: {}".format(self.dirs.summary))


    def _load_checkpoint(self):
        checkpoint = tf.train.latest_checkpoint(self.dirs.checkpoint)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)

    def _init_env(self):
        self.env = _get_environment(self.game)
        self.action_space = list(range(self.env.action_space.n))

    def train(self):
        self._init_env()
        self._log_configuration()

if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as sess:
        params = {

        }
        agent = Agent(sess, "Breakout-v0", **params)
        agent.train()
