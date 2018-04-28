# hecate.console.app
# Describes the Hecate admin.py utility application and commands.
#
# Created: Wed Oct 11 13:15:28 2017 -0400
# Author:  Allen Leis
#
# ID: app.py [] allen.leis@gmail.com $

"""
Describes the Hecate driver.py utility application and commands.
"""

##########################################################################
## Imports
##########################################################################

from commis import color
from commis import ConsoleProgram

from hecate.console.commands import COMMANDS

##########################################################################
## Utility Definition
##########################################################################

DESCRIPTION = "run and manage the Hecate process"
EPILOG = ""


##########################################################################
## The Hecate CLI Utility
##########################################################################

class HecateUtility(ConsoleProgram):

    description = color.format(DESCRIPTION, color.CYAN)
    epilog = color.format(EPILOG, color.MAGENTA)
    version = color.format("(Hecate) v{}", color.CYAN, "0.1")

    @classmethod
    def load(klass, commands=COMMANDS):
        utility = klass()
        for command in commands:
            utility.register(command)
        return utility


##########################################################################
## Run as a module
##########################################################################

def main():
    app = HecateUtility.load()
    app.execute()


if __name__ == '__main__':
    main()
