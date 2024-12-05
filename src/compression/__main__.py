from sys import argv

from dvc.api import params_show

from .compress import compress
from .train import train

if len(argv) != 2:
    msg = "The module main program requires exactly 1 argument"
    raise ValueError(msg)

if argv[1] == "train":
    train(**params_show(stages=[argv[1]]))
if argv[1] == "compress":
    compress(**params_show(stages=[argv[1]]))
