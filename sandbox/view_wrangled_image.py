"""
Outputs wrangled image to screen as ASCII
"""

##########################################################################
# Imports
##########################################################################

import random

import gym
import tensorflow as tf
import numpy as np
import tqdm
import pandas as pd


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

STEP_RESULT_FIELDS = ["state", "reward", "game_over", "extras", "previous_state"]

tf.reset_default_graph()


##########################################################################
# Helpers
##########################################################################

def _get_environment(name="Breakout-v0"):
    env = gym.envs.make(name)
    return env

##########################################################################
# Classes
##########################################################################


image_placeholder = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
transform = tf.image.crop_to_bounding_box(image_placeholder, 34, 0, 160, 160)
transform = tf.image.rgb_to_grayscale(transform)
transform = tf.image.resize_images(transform, OUTPUT_SHAPE, method=RESIZE_METHOD)
image_wrangler = tf.squeeze(transform)


def wrangle_image(session, image):
    return session.run(image_wrangler, {image_placeholder: image})

def main(session, env, populate_memory_steps=100, action_size=4, ):

    previous_state = env.reset()

    for step in tqdm.tqdm(range(populate_memory_steps)):
        action = random.randrange(action_size)
        results = env.step(action) + (previous_state,)

        record = dict(zip(STEP_RESULT_FIELDS, results))
        record["state"] = wrangle_image(session, record["state"])

        print(pd.DataFrame(record["state"]))

        if record["game_over"]:
            previous_state = env.reset()
        else:
            previous_state = record["state"]


##########################################################################
# Execution
##########################################################################

if __name__ == '__main__':
    GAME = "Seaquest-v0"
    with tf.Session() as sess:
        env = _get_environment(GAME)
        action_size = env.action_space.n

        main(sess, env, action_size=4)
