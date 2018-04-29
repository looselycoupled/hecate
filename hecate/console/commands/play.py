from commis import Command
from hecate.utils.mixins import LoggableMixin

class PlayCommand(LoggableMixin, Command):

    name    = "play"
    help    = "play Hecate on a given game"

    args    = {
        # Game to play
        '--game': {
            'default': 'Breakout-v0',
            'type': str,
            'choices': ['Breakout-v0', ],
            'help': 'game to play ("Breakout-v0", etc.)',
        },
        '--episodes': {
            'default': 100,
            'type': int,
            'help': 'episodes to play',
        },
        '--steps': {
            'default': 1000,
            'type': int,
            'help': 'steps allowed per episode',
        }
    }

    def handle(self, args):
        """
        Train on the supplied game
        """
        self.logger.info("startup")
        print("Playing {}!".format(args.game))
