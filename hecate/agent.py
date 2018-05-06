##########################################################################
# Imports
##########################################################################

# DEV ONLY
import time

import os
import random
from collections import Counter

import gym
from gym.wrappers import Monitor
import tensorflow as tf
import numpy as np
import tqdm
import pandas as pd

from hecate.utils.mixins import LoggableMixin
from hecate.utils.timer import Timer
from hecate.replay_buffer import ReplayBuffer
from hecate.dqn import DeepQNetwork
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

STEP_RESULT_FIELDS = ["state", "reward", "game_over", "extras", "previous_state", "action"]

##########################################################################
# Helpers
##########################################################################

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
    if reward < 0: return -1
    if reward > 0: return 1

##########################################################################
# Classes
##########################################################################

class Agent(LoggableMixin):

    def __init__(self,
        session,
        game,
        episodes=None,
        steps=50000,
        decay_steps=10000,
        populate_memory_steps=1000,
        update_target_steps=5000,
        storage_path="data",
        verbose=True,
    ):
        super(Agent, self).__init__()
        self.session = session
        self.game = game
        self.frame_skip_length = _get_frame_skip_length(game)
        self.episodes = episodes
        self.steps = steps
        self.decay_steps = decay_steps
        self.populate_memory_steps = populate_memory_steps
        self.update_target_steps = update_target_steps
        self.dirs = StorageConfig(storage_path, game)

        self.hold_out_buffer = ReplayBuffer(2000)
        self.replay_buffer = ReplayBuffer(2 * populate_memory_steps)
        self.step_count = 0
        self.env = None
        self.action_size = None
        self.action_space = None

        self.verbose = verbose

        self._init_image_wrangler()

    @property
    def saver(self):
        if not hasattr(self, "_saver"):
            self._saver = tf.train.Saver()
        return self._saver

    def _log_configuration(self):
        self.logger.info("AGENT CONFIG")
        self.logger.info("="*40)
        self.logger.info("game: {}".format(self.game))
        self.logger.info("action_space: {}".format(self.action_space))
        self.logger.info("checkpoint dir: {}".format(self.dirs.checkpoint))
        self.logger.info("monitoring dir: {}".format(self.dirs.monitoring))
        self.logger.info("summary dir: {}".format(self.dirs.summary))
        for item in ["episodes","steps","populate_memory_steps","decay_steps","update_target_steps"]:
            self.logger.info("{}: {}".format(item, getattr(self, item)))


    def _load_checkpoint(self):
        # TODO: Move this to DeepQNetwork class?
        checkpoint = tf.train.latest_checkpoint(self.dirs.checkpoint)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)

    def _save_checkpoint(self):
        # TODO: Move this to DeepQNetwork class?
        self.saver.save(tf.get_default_session(), self.dirs.checkpoint)

    def _init_env(self):
        self.env = _get_environment(self.game)
        self.action_size = self.env.action_space.n
        self.action_space = list(range(self.action_size))

    def _add_monitor(self):
        self.env = Monitor(
            self.env,
            directory=self.dirs.monitoring,
            video_callable=lambda count: self.step_count % 500 == 0,
            resume=True
        )

    def _init_image_wrangler(self):
        self.image_placeholder = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
        transform = tf.image.crop_to_bounding_box(self.image_placeholder, 34, 0, 160, 160)
        transform = tf.image.rgb_to_grayscale(transform)
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
        return  np.stack([processed_image] * 1, axis=2)

    def _populate_replay_memory(self, buffer, steps):
        """

        From: Playing atari with deep reinforcement learning (page 5)
            For the experiments in this paper, the function φ from algorithm 1 applies
            this preprocessing to the last 4 frames of a history and stacks them to
            produce the input to the Q-function.
        """
        self.logger.info("_populate_replay_memory: populating memory replay buffer")

        with Timer() as t:
            previous_state = self.wrangle_image(self.env.reset())
            # previous_state = np.stack([previous_state] * 4, axis=2)

            for step in tqdm.tqdm(range(steps)):
                action = random.randrange(self.action_size)
                results = self.env.step(action) + (previous_state, action)

                record = dict(zip(STEP_RESULT_FIELDS, results))
                record["reward"] = _fix_reward(record["reward"])
                record["state"] = self.wrangle_image(record["state"])
                # record["state"] =np.stack([record["state"]] * 4, axis=2)
                buffer.append(record)

                if record["game_over"]:
                    previous_state = self.wrangle_image(self.env.reset())
                    # previous_state = np.stack([previous_state] * 4, axis=2)
                else:
                    previous_state = record["state"]

        self.logger.info("_populate_replay_memory: populated ({}) in {}".format(len(buffer), t))


    def _choose_action(self, state, network):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size), True

        action_values = network.predict([state])[0]
        return np.argmax(action_values), False

    def _init_summary_writer(self):
        self.summary_writer = tf.summary.FileWriter(
            self.dirs.summary
        )

    def write_episodic_summaries(self, episode_rewards, episode_steps, random_actions, original_rewards, fixed_rewards):

        same_reward = int(original_rewards == fixed_rewards)

        summary = tf.Summary()
        summary.value.add(simple_value=episode_rewards, tag="episode/reward")
        summary.value.add(simple_value=sum(original_rewards), tag="episode/original reward total")
        summary.value.add(simple_value=sum(fixed_rewards), tag="episode/fixed reward total")
        summary.value.add(simple_value=same_reward, tag="episode/same_reward")
        summary.value.add(simple_value=episode_steps, tag="episode/steps")
        summary.value.add(simple_value=self.epsilon, tag="episode/epsilon")
        summary.value.add(simple_value=random_actions, tag="episode/random choices")
        summary.value.add(simple_value=random_actions / episode_steps, tag="episode/random choices %")
        self.summary_writer.add_summary(summary, self.episode_num)
        self.summary_writer.flush()

    def test_hold_out_buffer(self, network):
        if not hasattr(self, "_hold_out_states"):
            batch_replays = self.hold_out_buffer.buffer
            self._hold_out_states = np.array([replay["state"] for replay in batch_replays])

        batch_predicted_action_values = network.predict(self._hold_out_states)
        avg_max_action_value = np.array(batch_predicted_action_values).max(axis=1).mean()

        summary = tf.Summary()
        summary.value.add(simple_value=avg_max_action_value, tag="holdout/episode/avg max Q")
        self.summary_writer.add_summary(summary, self.episode_num)
        self.summary_writer.flush()


    def train(self):
        self._init_env()
        self._log_configuration()
        self._populate_replay_memory(self.replay_buffer, self.populate_memory_steps)
        self._populate_replay_memory(self.hold_out_buffer, 2000)
        self._add_monitor()
        self._init_summary_writer()
        total_reward = 0

        print("CREATING MODELS")
        other_network = DeepQNetwork(self.session, "other_network", self.action_size)
        target_network = DeepQNetwork(self.session, "target_network", self.action_size)

        self._load_checkpoint()
        self.session.run(tf.global_variables_initializer())

        print("STARTING TRAINING EPISODES")
        # start games
        for self.episode_num in range(self.episodes):
            episode_random_actions = 0

            # checkpoint all tf vars every couple of episodes
            if self.episode_num % 5 == 0:
                self._save_checkpoint()

            with Timer() as episode_timer:
                previous_state = self.wrangle_image(self.env.reset())
                step = 0
                rewards = 0
                chosen_actions = []

                original_rewards = []
                fixed_rewards = []

                while True:
                    self.step_count += 1
                    step +=1
                    action, is_random = self._choose_action(previous_state, target_network)

                    episode_random_actions += int(is_random)
                    chosen_actions.append(action)

                    # frame skipping
                    skipped_rewards = 0
                    for _ in range(self.frame_skip_length + 1):
                        results = list(self.env.step(action) + (previous_state, action))
                        skipped_rewards += results[1]
                        results[1] = skipped_rewards
                        results = tuple(results)
                        if results[2] is True:
                            break
                    # results = list(self.env.step(action) + (previous_state, action))


                    # TODO: abstract this out (repetative code)
                    record = dict(zip(STEP_RESULT_FIELDS, results))
                    original_rewards.append(record["reward"])

                    record["reward"] = _fix_reward(record["reward"])
                    fixed_rewards.append(record["reward"])

                    record["state"] = self.wrangle_image(record["state"])
                    # record["state"] = np.stack([record["state"]] * 4, axis=2)

                    self.replay_buffer.append(record)
                    previous_state = record["state"]
                    rewards += record["reward"]

                    # TODO: train model on random mini batch
                    batch_dicts = self.replay_buffer.sample()
                    batch = [
                            (item["state"], item["reward"], item["game_over"], item["extras"], item["previous_state"], item["action"])
                            for item in batch_dicts
                        ]
                    batch_states, batch_rewards, batch_game_overs, _, batch_next_states, batch_actions = list(map(np.array, zip(*batch)))

                    # calculate action values of next states
                    batch_predicted_action_values = target_network.predict(batch_next_states)
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
                    loss = other_network.train(batch_states, batch_labels, batch_actions)

                    if self.step_count % 10 == 0:
                        summary = tf.Summary()
                        summary.value.add(simple_value=loss, tag="loss")
                        self.summary_writer.add_summary(summary, self.step_count)
                        self.summary_writer.flush()


                    # periodically update target network to keep training stable
                    if self.step_count % self.update_target_steps == 0:
                        self.logger.info("Copying model variables to target network")
                        target_network.copy(other_network)

                    if record["game_over"] or self.step_count >= self.steps:
                        break

            total_reward += rewards
            print(fixed_rewards == original_rewards)

            self.write_episodic_summaries(rewards, step, episode_random_actions, original_rewards, fixed_rewards)
            self.test_hold_out_buffer(other_network)

            self.logger.info("Episode {} completed in {}".format(self.episode_num, episode_timer))
            if self.verbose:
                self.logger.info("Episode Reward: {}".format(rewards))
                self.logger.info("Episode Steps: {}".format(step))
                self.logger.info("Total Steps: {}".format(self.step_count))
                self.logger.info("Total Reward: {}".format(total_reward))
                # self.logger.info("Chosen Actions: {}".format(Counter(chosen_actions)))
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
