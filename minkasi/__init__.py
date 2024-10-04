from . import fitting, mapmaking, maps, tods, tools
from .parallel import *

import sys
from warnings import warn
if sys.version_info < (3, 9):
    warn(f"You are running on {sys.version}, which is EOL. Basic mapmaking has been tested down to 3.7 but not everything is guaranteed to work. Please upgrade python if possible!")
