##########################################################################
## Imports
##########################################################################

from .train import TrainCommand
from .play import PlayCommand
from .experiment import ExperimentCommand

COMMANDS = [
    TrainCommand,
    PlayCommand,
    ExperimentCommand,
]
