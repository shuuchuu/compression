from importlib import import_module
from sys import argv

from dvc.api import params_show

MODULES = {"prepare": "data", "train": "train", "compress": "compress"}

if len(argv) != 2 or argv[1] not in MODULES:
    msg = f"Usage: python -m model_compression {{{','.join(MODULES)}}}"
    raise SystemExit(msg)

stage = argv[1]
function = getattr(import_module(f"model_compression.{MODULES[stage]}"), stage)
function(**params_show(stages=[stage]))
