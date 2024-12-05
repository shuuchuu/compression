from dvc.api import params_show

from .train import train

train(**params_show(stages=["train"]))
