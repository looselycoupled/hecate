
# Overview

This project contains a Tensorflow implementaion of Mnih et al.'s 2013 work establishing the Deep Q-Network.  Also included, is a version of the 2015 network the authors published in Nature.  

The software is able to train on the following OpenAI Gym games: Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders.

# Setup

In general, consult the OpenAI Gym and Tensoflow documentation to setup your local dependencies.  The following may help get you started.

## Home Laptop (Macbook Pro)

    brew install cmake boost boost-python sdl2 swig wget
    pip install gym[all] tensorflow numpy pandas jupyter matplotlib


## AWS

Experiments have been conducted on AWS EC2 servers using `p2.xlarge` instances:

* AMI: `Deep Learning AMI (Ubuntu) Version 8.0 (ami-dff741a0)`
* Instance Type: `p2.xlarge`

To install dependencies, run the following commands:

    sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

    git clone https://github.com/looselycoupled/hecate.git

    source activate tensorflow_p36

    pip install gym[atari] tqdm

    cd /hecate

    pip install -r requirements.txt

# Execution

To train a network (using nohup) for Breakout-v0:

    nohup python -u driver.py train --model_year=2015  --max_steps=1000000 --game=Breakout-v0  --traceback >log.txt 2>&1 &
