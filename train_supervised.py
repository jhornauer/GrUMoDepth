from __future__ import absolute_import, division, print_function

from trainer_supervised import Trainer
from extended_options import *


if __name__ == "__main__":
    options = UncertaintyOptions()
    opts = options.parse()
    trainer = Trainer(opts)
    trainer.train()
