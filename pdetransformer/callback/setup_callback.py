from lightning import Callback
from omegaconf import OmegaConf
import os


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
        else:
            pass
