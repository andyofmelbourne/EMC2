import sys
from context import emc2

from emc2 import utils

config = utils.load_config(sys.argv[1])
print(config['iterations'])
