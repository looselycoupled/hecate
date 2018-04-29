##########################################################################
# Imports
##########################################################################

# DEV ONLY
import time

import os
import random

import gym
import tensorflow as tf
import numpy as np
import tqdm
import pandas as pd

# from hecate.image import wrangle_image
from hecate.utils.mixins import LoggableMixin
from hecate.utils.timer import Timer
from hecate.replay_buffer import ReplayBuffer



#########################################################################
# Helpers
##########################################################################

pd.set_option('display.height', 0)
pd.set_option('display.max_rows', 0)
pd.set_option('display.max_columns', 0)
pd.set_option('display.width', 0)

RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
INPUT_SHAPE = [210, 160, 3]
OUTPUT_SHAPE = [84, 84]

##########################################################################
# Helpers
##########################################################################

def _get_environment(name="Breakout-v0"):
    env = gym.envs.make(name)
    return env

def _fix_reward(reward):
    """
    From: Playing atari with deep reinforcement learning (page 6)
        ...we made one change to the reward structure of the games during training
        only. Since the scale of scores varies greatly from game to game, we fixed
        all positive rewards to be 1 and all negative rewards to be −1, leaving 0
        rewards unchanged.
    """
    if reward == 0: return reward
    if reward < 0: return -1
    if reward > 0: return 1

##########################################################################
# Classes
##########################################################################

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
        episodes=None,
        steps=50000,
        decay_steps=10000,
        populate_memory_steps=10000,
        upate_target_steps=10000,
        storage_path="data",
    ):
        super(Agent, self).__init__()
        self.session = session
        self.game = game
        self.episodes = episodes
        self.steps = steps
        self.decay_steps = decay_steps
        self.populate_memory_steps = populate_memory_steps
        self.upate_target_steps = upate_target_steps
        self.dirs = StorageConfig(storage_path, game)

        self.model = tf.Variable(0, name='foo', trainable=False)
        self.saver = tf.train.Saver()
        self.replay_buffer = ReplayBuffer(2 * populate_memory_steps)
        self.env = None
        self.action_size = None
        self.action_space = None

        self._init_image_wrangler()
        self._load_checkpoint()

    def _log_configuration(self):
        self.logger.info("AGENT CONFIG")
        self.logger.info("="*40)
        self.logger.info("game: {}".format(self.game))
        self.logger.info("action_space: {}".format(self.action_space))
        self.logger.info("checkpoint dir: {}".format(self.dirs.checkpoint))
        self.logger.info("monitoring dir: {}".format(self.dirs.monitoring))
        self.logger.info("summary dir: {}".format(self.dirs.summary))
        self.logger.info("")


    def _load_checkpoint(self):
        checkpoint = tf.train.latest_checkpoint(self.dirs.checkpoint)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)

    def _init_env(self):
        self.env = _get_environment(self.game)
        self.action_size = self.env.action_space.n
        self.action_space = list(range(self.action_size))

    def _init_image_wrangler(self):
        self.image_placeholder = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
        transform = tf.image.crop_to_bounding_box(self.image_placeholder, 34, 0, 160, 160)
        transform = tf.image.rgb_to_grayscale(transform)
        transform = tf.image.resize_images(transform, OUTPUT_SHAPE, method=RESIZE_METHOD)
        self.image_wrangler = tf.squeeze(transform)

    def wrangle_image(self, image):
        return self.session.run(self.image_wrangler, {self.image_placeholder: image})

    def _populate_replay_memory(self):
        """

        From: Playing atari with deep reinforcement learning (page 5)
            For the experiments in this paper, the function φ from algorithm 1 applies
            this preprocessing to the last 4 frames of a history and stacks them to
            produce the input to the Q-function.
        """
        self.logger.info("_populate_replay_memory: populating memory replay buffer")
        step_result_fields = ["state", "reward", "game_over", "extras", "previous_state"]

        with Timer() as t:
            previous_state = self.env.reset()

            for step in tqdm.tqdm(range(self.populate_memory_steps)):
                action = random.randrange(self.action_size)
                results = self.env.step(action) + (previous_state,)

                record = dict(zip(step_result_fields, results))
                record["reward"] = _fix_reward(record["reward"])
                record["state"] = self.wrangle_image(record["state"])

                # print(pd.DataFrame(record["state"]))
                self.replay_buffer.append(record)

                if record["game_over"]:
                    previous_state = self.env.reset()
                else:
                    previous_state = record["state"]

        self.logger.info("_populate_replay_memory: populated ({}) in {}".format(len(self.replay_buffer), t))


    def train(self):
        self._init_env()
        self._log_configuration()
        self._populate_replay_memory()













##########################################################################
# Execution
##########################################################################

if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as sess:
        params = {

        }
        agent = Agent(sess, "Breakout-v0", **params)
        agent.train()
