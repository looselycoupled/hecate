##########################################################################
# Imports
##########################################################################

# DEV ONLY
import time

import os
import random
from collections import Counter
from itertools import count

import gym
from gym.wrappers import Monitor
import tensorflow as tf
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from hecate.utils.mixins import LoggableMixin
from hecate.utils.timer import Timer
from hecate.replay_buffer import ReplayBuffer
from hecate.dqn import DeepQNetwork, DeepQNetwork2015
from hecate.config import StorageConfig


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

DISCOUNT_RATE = .99
HOLD_OUT_BUFFER_SIZE = 320

STEP_RESULT_FIELDS = ["state", "reward", "game_over", "extras", "previous_state", "action"]

##########################################################################
# Helpers
##########################################################################

def _render_state_frames(state):
    for idx in range(state.shape[2]):
        plt.imshow(state[...,idx])
        plt.show()

def _get_network_by_year(year):
    if year == 2013:
        return DeepQNetwork
    if year == 2015:
        return DeepQNetwork2015
    raise Exception("Invalid year for model research: {}".format(year))

def _get_environment(name="Breakout-v0"):
    env = gym.envs.make(name)
    return env

def _get_frame_skip_length(name):
    """
    From Paper: We use k = 4 for all games except Space Invaders
    """
    if name == "SpaceInvaders-v0":
        return 2
    return 3

def _fix_reward(reward):
    """
    From: Playing atari with deep reinforcement learning (page 6)
        ...we made one change to the reward structure of the games during training
        only. Since the scale of scores varies greatly from game to game, we fixed
        all positive rewards to be 1 and all negative rewards to be −1, leaving 0
        rewards unchanged.
    """
    if reward == 0: return reward
    if reward < 0: return -1.0
    if reward > 0: return 1.0

##########################################################################
# Classes
##########################################################################

class Agent(LoggableMixin):

    def __init__(self,
        session,
        game,
        max_steps,
        decay_steps,
        populate_memory_steps,
        update_target_steps,
        storage_path="data",
        model_year=2013,
        verbose=False,
    ):
        # TODO: most of these should really be arguments to the train method
        super(Agent, self).__init__()
        self.session = session
        self.game = game
        self.frame_skip_length = _get_frame_skip_length(game)
        self.max_steps = max_steps
        self.decay_steps = decay_steps
        self.populate_memory_steps = populate_memory_steps
        self.update_target_steps = update_target_steps
        self.stabilized_learning = model_year == 2015
        self.NetworkModel = _get_network_by_year(model_year)
        self.dirs = StorageConfig(storage_path, game, self.NetworkModel)
        self.verbose = verbose

        self.hold_out_buffer = ReplayBuffer(HOLD_OUT_BUFFER_SIZE, "holdout")
        self.replay_buffer = ReplayBuffer(2 * populate_memory_steps, "replay")
        self.step_count = 0
        self.env = None
        self.action_size = None
        self.action_space = None


        self._init_image_wrangler()

    @property
    def saver(self):
        if not hasattr(self, "_saver"):
            self._saver = tf.train.Saver()
        return self._saver

    def _log_configuration(self):
        self._log_heading("AGENT CONFIG")
        self.logger.info("game: {}".format(self.game))
        self.logger.info("action_space: {}".format(self.action_space))
        self.logger.info("network model: {}".format(self.NetworkModel.__name__))
        self.logger.info("stabilize learning: {}".format(self.stabilized_learning))
        self.logger.info("checkpoint dir: {}".format(self.dirs.checkpoint))
        self.logger.info("monitoring dir: {}".format(self.dirs.monitoring))
        self.logger.info("summary dir: {}".format(self.dirs.summary))
        for item in ["max_steps","populate_memory_steps","decay_steps","update_target_steps"]:
            self.logger.info("{}: {}".format(item, getattr(self, item)))

    def _load_checkpoint(self, path=None):
        # https://www.tensorflow.org/programmers_guide/saved_model
        path = path if path is not None else self.dirs.checkpoint
        checkpoint = tf.train.latest_checkpoint(path)
        if checkpoint:
            self.logger.info("loading checkpoint: {}".format(path))
            self.saver.restore(self.session, checkpoint)
            return True
        else:
            self.logger.info("no checkpoint to load")
            return False

    def _save_checkpoint(self):
        # https://www.tensorflow.org/programmers_guide/saved_model
        self.saver.save(tf.get_default_session(), os.path.join(self.dirs.checkpoint, "model.ckpt"))

    def _init_env(self):
        self.env = _get_environment(self.game)
        self.action_size = self.env.action_space.n
        self.action_space = list(range(self.action_size))

    def _add_monitor(self, record_frequency=50):
        self.env = Monitor(
            self.env,
            directory=self.dirs.monitoring,
            video_callable=lambda episode_count: episode_count % record_frequency == 0,
            resume=True
        )

    def _init_image_wrangler(self):
        self.image_placeholder = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
        transform = tf.image.rgb_to_grayscale(self.image_placeholder)
        transform = tf.image.crop_to_bounding_box(transform, 34, 0, 160, 160)
        transform = tf.image.resize_images(transform, OUTPUT_SHAPE, method=RESIZE_METHOD)
        self.image_wrangler = tf.squeeze(transform)

    @property
    def epsilon(self):
        # From Paper:
        # The behavior policy during training was ε-greedy with ε annealed linearly from
        # 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter.
        if not hasattr(self, "_epsilon_schedule"):
            self._epsilon_schedule = np.linspace(1, .1, self.decay_steps)

        if self.step_count > self.decay_steps:
            return .1
        return self._epsilon_schedule[self.step_count - 1]

    def wrangle_image(self, image):
        processed_image =  self.session.run(
            self.image_wrangler,
            {self.image_placeholder: image}
        )
        assert processed_image.shape == (84, 84)
        return processed_image

    def _populate_replay_memory(self, buffer, steps):
        """

        From: Playing atari with deep reinforcement learning (page 5)
            For the experiments in this paper, the function φ from algorithm 1 applies
            this preprocessing to the last 4 frames of a history and stacks them to
            produce the input to the Q-function.
        """
        self.logger.info("populating {} buffer".format(buffer.name))

        with Timer() as t:
            previous_state = self.wrangle_image(self.env.reset())
            previous_state = np.stack([previous_state] * 4, axis=2)

            for step in tqdm.tqdm(range(steps)):
                action = random.randrange(self.action_size)
                record = self.game_step(action, previous_state)
                buffer.append(record)

                if record["game_over"]:
                    previous_state = self.wrangle_image(self.env.reset())
                    previous_state = np.stack([previous_state] * 4, axis=2)
                else:
                    previous_state = record["state"]

        self.logger.info("populated ({}) in {}".format(len(buffer), t))


    def _choose_action(self, state, network, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size), True

        action_values = network.predict([state])[0]
        return np.argmax(action_values), False

    @property
    def summary_writer(self):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = tf.summary.FileWriter(
                self.dirs.summary
            )
        return self._summary_writer

    def write_episodic_summaries(self, episode_rewards, episode_steps, random_actions, avg_max_q, elapsed):
        summary = tf.Summary()
        summary.value.add(simple_value=elapsed, tag="episode/elapsed")
        summary.value.add(simple_value=episode_rewards, tag="episode/reward")
        summary.value.add(simple_value=episode_steps, tag="episode/steps")
        summary.value.add(simple_value=random_actions, tag="episode/random choices")
        summary.value.add(simple_value=random_actions / episode_steps, tag="episode/random choices %")
        summary.value.add(simple_value=avg_max_q, tag="episode/avg_max_q")
        self.summary_writer.add_summary(summary, self.episode_num)
        self.summary_writer.flush()

    def test_hold_out_buffer(self, network):
        # From paper:
        #    a more stable, metric is the policy’s estimated action-value function Q,
        #    which provides an estimate of how much discounted reward the agent can
        #    obtain by following its policy from any given state. We collect a fixed
        #    set of states by running a random policy before training starts and track
        #    the average of the maximum2 predicted Q for these states.
        if not hasattr(self, "_hold_out_states"):
            batch_replays = self.hold_out_buffer.buffer
            self._hold_out_states = np.array([replay["state"] for replay in batch_replays])

        # batch_predicted_action_values = None
        # for batch in np.array_split(self._hold_out_states, 32):
        #     if batch_predicted_action_values is None:
        #         batch_predicted_action_values = network.predict(batch)
        #     else:
        #         batch_predicted_action_values = np.concatenate([batch_predicted_action_values, network.predict(batch)], axis=0)
        #         # batch_predicted_action_values.append(network.predict(batch))

        batch_predicted_action_values = network.predict(self._hold_out_states)
        return np.array(batch_predicted_action_values).max(axis=1).mean()


    def game_step(self, action, previous_state):
        # take step and create dict for action result
        results = list(self.env.step(action) + (previous_state, action))
        record = dict(zip(STEP_RESULT_FIELDS, results))

        # clip reward per 2013 paper
        record["reward"] = _fix_reward(record["reward"])

        # wrangle image to 84x84 and expand so it can be added to history frame
        record["state"] = self.wrangle_image(record["state"])
        record["state"] = np.expand_dims(record["state"], 2)

        # drop first history frame and then append new state to history
        tmp = record["previous_state"][...,1:]
        record["state"] = np.append(tmp, record["state"], axis=2)

        assert record["state"].shape == (84,84,4)
        assert record["previous_state"].shape == (84,84,4)
        return record


    def train_mini_batch(self, q_network, target_q_network):
        # train model on random mini batch
        batch_dicts = self.replay_buffer.sample()
        batch = [
                (item["state"], item["reward"], item["game_over"], item["extras"], item["previous_state"], item["action"])
                for item in batch_dicts
            ]
        batch_states, batch_rewards, batch_game_overs, _, batch_previous_states, batch_actions = map(np.array, zip(*batch))

        # calculate action values of next states
        batch_predicted_action_values = target_q_network.predict(batch_states)
        batch_next_action_values = np.amax(
            batch_predicted_action_values, axis=1
        )

        # calculate values using Bellman equation for training network
        batch_labels = batch_rewards + (
            np.invert(batch_game_overs).astype(np.float32) *
            DISCOUNT_RATE *
            batch_next_action_values
        )

        # record loss over time
        return q_network.train(batch_previous_states, batch_labels, batch_actions)

    def _log_heading(self, msg):
        self.logger.info("")
        self.logger.info(msg)
        self.logger.info("="*40)


    def train(self):
        self._init_env()
        self._log_configuration()

        # training setup
        self._log_heading("TRAINING INITIALIZATION")
        self._populate_replay_memory(self.replay_buffer, self.populate_memory_steps)
        self._populate_replay_memory(self.hold_out_buffer, 1000)
        self._add_monitor()
        total_reward = 0

        q_network = self.NetworkModel(self.session, "q_network", self.action_size)
        if self.stabilized_learning:
            target_q_network = self.NetworkModel(self.session, "target_q_network", self.action_size)
        else:
            target_q_network = q_network

        self._load_checkpoint()
        self.session.run(tf.global_variables_initializer())

        # start games
        self._log_heading("STARTING TRAINING EPISODES")
        for self.episode_num in count(start=1):
            episode_random_actions = 0

            # checkpoint all tf vars
            self._save_checkpoint()

            with Timer() as episode_timer:
                previous_state = self.wrangle_image(self.env.reset())
                previous_state = np.stack([previous_state] * 4, axis=2)
                step = 0
                rewards = 0

                while True:
                    self.step_count += 1
                    step +=1

                    action, is_random = self._choose_action(previous_state, q_network, self.epsilon)
                    episode_random_actions += int(is_random)

                    # take game step and record step results
                    record = self.game_step(action, previous_state)
                    self.replay_buffer.append(record)
                    previous_state = record["state"]
                    rewards += record["reward"]

                    # perform training
                    loss = self.train_mini_batch(q_network, target_q_network)

                    # record loss at every step
                    summary = tf.Summary()
                    summary.value.add(simple_value=loss, tag="step/loss")
                    self.summary_writer.add_summary(summary, self.step_count)
                    self.summary_writer.flush()

                    # periodically update target network to keep training stable (2015 paper)
                    if self.stabilized_learning and self.step_count % self.update_target_steps == 0:
                        self.logger.info("Updating target network weights")
                        target_q_network.copy(q_network)

                    if record["game_over"] or self.step_count >= self.max_steps:
                        break

            total_reward += rewards

            # test holdout replays after episode
            avg_max_q = self.test_hold_out_buffer(target_q_network)

            # record tf summaries for episode
            self.write_episodic_summaries(rewards, step, episode_random_actions, avg_max_q, int(episode_timer.elapsed))

            if self.verbose:
                self.logger.info("Episode {} completed in {}".format(self.episode_num, episode_timer))
                self.logger.info("Episode Reward: {}".format(rewards))
                self.logger.info("Episode Steps: {}".format(step))
                self.logger.info("Total Steps: {}".format(self.step_count))
                self.logger.info("Total Reward: {}".format(total_reward))
                self.logger.info("")
            else:
                report = {"steps": step, "reward": rewards, "time": str(episode_timer), "total_steps": self.step_count}
                self.logger.info("Episode {}: {}".format(self.episode_num, report))

            if self.step_count >= self.max_steps:
                break

        self.logger.info("Tensorboard command: tensorboard --logdir=\"{}\"".format(self.dirs.summary))



    def play(self, episodes_to_play, record_frequency):
        self._init_env()
        self._log_configuration()

        # training setup
        self._log_heading("PLAY INITIALIZATION")
        self._add_monitor(record_frequency=1)
        total_reward = 0
        all_rewards = []
        q_network = self.NetworkModel(self.session, "q_network", self.action_size)
        if self.stabilized_learning:
            target_q_network = self.NetworkModel(self.session, "target_q_network", self.action_size)

        if not self._load_checkpoint():
            raise Exception("No checkpoint found at {}".format(self.dirs.checkpoint))

        self.session.run(tf.global_variables_initializer())

        # start games
        self._log_heading("STARTING GAME PLAY")
        for self.episode_num in range(1, episodes_to_play+1):
            episode_random_actions = 0
            step = 0
            rewards = 0

            with Timer() as episode_timer:
                previous_state = self.wrangle_image(self.env.reset())
                previous_state = np.stack([previous_state] * 4, axis=2)

                # take steps in game
                while True:
                    self.step_count += 1
                    step += 1

                    action, is_random = self._choose_action(previous_state, q_network, 0.05)
                    episode_random_actions += int(is_random)

                    # take game step and record step results
                    record = self.game_step(action, previous_state)
                    previous_state = record["state"]
                    rewards += record["reward"]

                    if record["game_over"]:
                        break

            report = {"steps": step, "reward": rewards, "time": str(episode_timer), "total_steps": self.step_count}
            self.logger.info("Episode {}: {}".format(self.episode_num, report))


            # elapsed = int(episode_timer.elapsed)
            # summary = tf.Summary()
            # summary.value.add(simple_value=elapsed, tag="play/elapsed")
            # summary.value.add(simple_value=rewards, tag="play/reward")
            # summary.value.add(simple_value=step, tag="play/steps")
            # summary.value.add(simple_value=episode_random_actions, tag="play/random choices")
            # summary.value.add(simple_value=episode_random_actions / step, tag="play/random choices %")
            # self.summary_writer.add_summary(summary, self.episode_num)
            # self.summary_writer.flush()


            all_rewards.append(rewards)

        self._log_heading("PLAY RESULTS")
        self.logger.info("max reward: {}".format(max(all_rewards)))
        self.logger.info("avg reward: {}".format(np.mean(all_rewards)))


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
