

# Setup

## PiP

    pip install gym[all] tensorflow numpy pandas jupyter matplotlib

## Brew

    brew install cmake boost boost-python sdl2 swig wget


env.reset()
env.render()

def noop(env, steps=1):
    for i in range(steps):
        env.step(0)

env.reset()
env.render()

for i in range(120):
    env.step(2)
    env.render()
    env.step(1)
    env.render()


    noop(env)
    env.step(1)
    env.render()
    noop(env)
    env.render()
