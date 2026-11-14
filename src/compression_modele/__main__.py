from importlib import import_module
from sys import argv

from dvc.api import params_show

MODULES = {"prepare": "donnees", "entraine": "entrainement", "compresse": "compression"}

if len(argv) != 2 or argv[1] not in MODULES:
    message = f"Usage : python -m compression_modele {{{','.join(MODULES)}}}"
    raise SystemExit(message)

etape = argv[1]
fonction = getattr(import_module(f"compression_modele.{MODULES[etape]}"), etape)
fonction(**params_show(stages=[etape]))
