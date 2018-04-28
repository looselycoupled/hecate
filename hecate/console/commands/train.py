from commis import Command
from hecate.utils.mixins import LoggableMixin


class TrainCommand(LoggableMixin, Command):

    name    = "train"
    help    = "train Hecate on a given game"

    args    = {
        # Steps for training
        ('-s', '--steps'): {
            'type': int,
            'default': 10000,
            'help': 'steps for training',
        },

        # Game to train on
        '--game': {
            'default': 'Breakout-v0',
            'type': str,
            'choices': ['Breakout-v0', ],
            'help': 'game to play ("Breakout-v0", etc.)',
        }
    }

    def handle(self, args):
        """
        Train on the supplied game
        """
        self.logger.info("startup")
        print("Training {}!".format(args.game))
