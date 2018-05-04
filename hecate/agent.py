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


# TODO:
# Following previous approaches to playing Atari games, we also use a simple
# frame-skipping technique [3]. More precisely, the agent sees and selects
# actions on every kth frame instead of every frame, and its last action is
# repeated on skipped frames. Since running the emulator forward for one step
# requires much less computation than having the agent select an action, this
# technique allows the agent to play roughly k times more games without
# significantly increasing the runtime. We use k = 4 for all games except
# Space Invaders where we noticed that using k = 4 makes the lasers invisible
# because of the period at which they blink. We used k = 3 to make the lasers
# visible and this change was the only difference in hyperparameter values between
# any of the games.


# TODO:
# In these experiments, we used the RMSProp algorithm with minibatches of size 32.
# The behavior policy during training was ε-greedy with ε annealed linearly from
# 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter. We trained
# for a total of 10 million frames and used a replay memory of one million most
# recent frames.



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

PAPER_PARAMETERS = {
    "EPSILON_DECAY_STEPS": 1e6,
    "EPSILON_START_VALUE": 1,
    "EPSILON_END_VALUE": .1,

    "TOTAL_TRAINING_STEPS": 1e7,

    "REPLAY_BUFFER_SIZE": 1e6,
    "REPLAY_BUFFER_INIT_SIZE": 5e5, # TODO: double check starting size in paper
}

STEP_RESULT_FIELDS = ["state", "reward", "game_over", "extras", "previous_state"]

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



class DeepQNetwork(LoggableMixin):

    def __init__(self, session, action_size):
        self.session = session
        self.action_size = action_size
        self.input = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.uint8, name="input")
        self.labels = tf.placeholder(shape=[None], dtype=tf.uint8, name="labels")
        self.actions = tf.placeholder(shape=[None], dtype=tf.uint8, name="actions")

    def _network(self):
        # https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
        # https://www.tensorflow.org/api_docs/python/tf/layers/dense

        # The first hidden layer convolves 16 8 × 8 filters with stride 4 with
        # the input image and applies a rectifier nonlinearity [10, 18].
        conv_layer_1 = tf.layers.conv2d(
            self.input,
            filters=16,
            kernel_size=(8, 8),
            strides=4,
            data_format="channels_last",
            activation=tf.nn.relu,
            name="conv_layer_1",
        )
        # The second hidden layer convolves 32 4 × 4 filters with stride 2, again
        # followed by a rectifier nonlinearity.
        conv_layer_2 = tf.layers.conv2d(
            conv_layer_1,
            filters=32,
            kernel_size=(4, 4),
            strides=2,
            data_format="channels_last",
            activation=tf.nn.relu,
            name="conv_layer_2",
        )
        # The final hidden layer is fully-connected and consists of 256
        # rectifier units.
        dense_layer_1 = tf.layers.dense(
            conv_layer_2,
            256,
            activation=tf.nn.relu,
            name="dense_layer_1",
        )

        # The output layer is a fully-connected linear layer with a single output
        # for each valid action.
        return tf.layers.dense(dense_layer_1, self.action_size, name="output_layer")

    def _optimizer(self):
        # From paper:
        #   In these experiments, we used the RMSProp algorithm with minibatches of size 32.
        # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
        return tf.train.RMSPropOptimizer(
            0.00025, # learning_rate,
            decay=0.99,
            momentum=0.0,
            epsilon=1e-6,
            name='RMSProp'
        )

    def _setup(self):
        self.output_layer = self._network()

        # https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error
        loss = tf.losses.mean_squared_error(self.labels, self.output_layer)

        self.optimize = self._optimizer().minimize(loss)

    def optimize(self, input, labels, actions):
         train_result, loss = self.sess.run(
            [self.optimize, self.loss],
            { self.input: input, self.labels: labels, self.actions: actions })

        # return loss


    def predict(self, input):
        return sess.run(self.output_layer, { self.input: input })


    def copy(self, source):
        pass


class Agent(LoggableMixin):

    def __init__(self,
        session,
        game,
        episodes=None,
        steps=50000,
        decay_steps=10000,
        populate_memory_steps=1000,
        upate_target_steps=10000,
        storage_path="data",
        verbose=True,
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
        self.step_count = 0
        self.env = None
        self.action_size = None
        self.action_space = None

        self.verbose = verbose

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

    @property
    def epsilon(self):
        if not hasattr(self, "_epsilon_schedule"):
            self._epsilon_schedule = np.linspace(1, .1, self.decay_steps)
        if self.step_count > self.decay_steps:
            return 1
        return self._epsilon_schedule[self.step_count - 1]

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

        with Timer() as t:
            previous_state = self.env.reset()
            for step in tqdm.tqdm(range(self.populate_memory_steps)):
                action = random.randrange(self.action_size)
                results = self.env.step(action) + (previous_state,)

                record = dict(zip(STEP_RESULT_FIELDS, results))
                record["reward"] = _fix_reward(record["reward"])
                record["state"] = self.wrangle_image(record["state"])
                self.replay_buffer.append(record)

                if record["game_over"]:
                    previous_state = self.env.reset()
                else:
                    previous_state = record["state"]

        self.logger.info("_populate_replay_memory: populated ({}) in {}".format(len(self.replay_buffer), t))


    def _choose_action(self, episode_num):
        if random.random() > self.epsilon:
            return random.randrange(self.action_size)

        # for now just return random instead of policy choice
        return random.randrange(self.action_size)

    def train(self):
        self._init_env()
        self._log_configuration()
        self._populate_replay_memory()
        total_reward = 0

        other_network = DeepQNetwork(self.session, self.action_size)
        target_network = DeepQNetwork(self.session, self.action_size)

        # start games
        for episode_num in range(self.episodes):
            with Timer() as episode_timer:

                previous_state = self.env.reset()
                step = 0
                rewards = 0

                while True:
                    self.step_count += 1
                    step +=1

                    action = self._choose_action(episode_num)

                    # TODO: abstract this out (repetative code)
                    results = self.env.step(action) + (previous_state,)
                    record = dict(zip(STEP_RESULT_FIELDS, results))
                    record["reward"] = _fix_reward(record["reward"])
                    record["state"] = self.wrangle_image(record["state"])

                    self.replay_buffer.append(record)
                    previous_state = record["state"]
                    rewards += record["reward"]

                    # TODO: train model on random mini batch


                    # TODO: periodically update target network to keep training stable
                    if not self.step_count % 5000:
                        target_network.copy(other_network)

                    if record["game_over"] or self.step_count >= self.steps:
                        break

            # time.sleep(.005)
            total_reward += rewards
            self.logger.info("Episode {} completed in {}".format(episode_num, episode_timer))
            if self.verbose:
                self.logger.info("Episode Reward: {}".format(rewards))
                self.logger.info("Episode Steps: {}".format(step))
                self.logger.info("Total Steps: {}".format(self.step_count))
                self.logger.info("Total Reward: {}".format(total_reward))
                self.logger.info("")

            if self.step_count >= self.steps:
                break










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
