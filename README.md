
# Overview

## Available Games
Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders


# Setup

## Home Laptop

    brew install cmake boost boost-python sdl2 swig wget
    pip install gym[all] tensorflow numpy pandas jupyter matplotlib


## AWS

* AMI: `Deep Learning AMI (Ubuntu) Version 8.0 (ami-dff741a0)`
* Instance Type: `p2.xlarge`

    sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

    source activate tensorflow_p36

    pip install gym[atari] tqdm


    nohup python -u driver.py train >log.txt 2>&1 &
