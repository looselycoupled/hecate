#!/usr/bin/env python3
# admin
# Management script for the hecate process and runner.
#
# Created: Tue Jun 27 15:03:37 2017 -0400
# Author: Allen Leis
#
# ID: admin.py [] allen.leis@gmail.com $

"""
Management script for the hecate process and runner.
"""

##########################################################################
## Imports
##########################################################################

from hecate.console import HecateUtility

##########################################################################
## Load and execute the CLI utility
##########################################################################

if __name__ == '__main__':
    app = HecateUtility.load()
    app.execute()
