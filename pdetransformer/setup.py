import os
import sys

import torch
import lightning
from lightning import seed_everything, Trainer

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import OmegaConf, DictConfig

from src.callback.callbacks import get_callbacks
from src.utils import instantiate_from_config

import logging
log = logging.getLogger(__name__)

from lightning.fabric.utilities.rank_zero import rank_prefixed_message, _get_rank, rank_zero_info


def set_default_options(config: DictConfig):
    torch.set_float32_matmul_precision('medium')
    sys.path.append(os.getcwd())
    seed_everything(config['runtime']['seed'])



def get_loggers(config: DictConfig):

    log.info(rank_prefixed_message(f"Runtime config: {config.runtime}", _get_rank()))

    wandb_logger = WandbLogger(
        id=config.runtime.id,
        name=config.runtime.name,
        project=config.runtime.project,
        dir=config.runtime.logdir,
        mode=config.runtime.logger_state,
        resume="allow",
        config=OmegaConf.to_container(config, resolve=True)
    )

    loggers = {'wandb': wandb_logger}

    return loggers


def modify_signals(trainer, ckptdir):
    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:

            ckpt_path = os.path.join(ckptdir, "last.ckpt")

            try:
                trainer.save_checkpoint(ckpt_path)
                log.info(rank_prefixed_message("Summoning checkpoint.", _get_rank()))
            except AttributeError as e:
                log.info(rank_prefixed_message("Did not save checkpoint.", _get_rank()))

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    return melk


def set_initial_learning_rate(trainer, cpu, batch_size, base_learning_rate, trainer_config):
    if not cpu:
        ngpu = trainer.num_devices
    else:
        ngpu = 1

    accumulate_grad_batches = trainer_config.get("accumulate_grad_batches", 1)

    if trainer_config.scale_lr:

        learning_rate = accumulate_grad_batches * ngpu * batch_size * base_learning_rate
        rank_zero_info(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * "
            "{} (batchsize) * {:.2e} (base_lr)".format(learning_rate, accumulate_grad_batches,
                                                       ngpu, batch_size, base_learning_rate))
    else:
        learning_rate = base_learning_rate
        rank_zero_info("++++ NOT USING LR SCALING ++++")
        rank_zero_info(f"Setting learning rate to {learning_rate:.2e}")

    return learning_rate


def handle_exception(trainer, e, debug):

    if debug and trainer is not None and trainer.global_rank == 0:
        try:
            import pudb as debugger
        except ImportError:
            import pdb as debugger
        debugger.post_mortem()
        raise

    raise e


def cleanup(trainer, debug, resume, logdir):

    if trainer is not None and trainer.global_rank == 0:

        # move newly created debug project to debug_runs
        if debug and not resume:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

        if trainer.profiler is not None:
            print(trainer.profiler.summary())


def main_setup(config):

    set_default_options(config)

    loggers = get_loggers(config)

    trainer_config: DictConfig = config.pop("trainer", OmegaConf.create())

    trainer = None
    checkpoint_dir = config.runtime.checkpoint_dir

    try:

        callbacks = get_callbacks(config.pop("callbacks", OmegaConf.create()))

        model_config = OmegaConf.to_container(config.model)
        model: lightning.LightningModule = instantiate_from_config(model_config)

        if "fine_tuning" in config.keys():
            from src.fine_tuning import fine_tuning_setup
            fine_tuning_setup(model, config, wandb_logger=loggers["wandb"])

        trainer = Trainer(
            callbacks=callbacks,
            logger=list(loggers.values()),
            use_distributed_sampler=False,
            **trainer_config["params"]
        )

        data: lightning.LightningDataModule = instantiate_from_config(config.data)

        melk_fn = modify_signals(trainer, config['runtime']['checkpoint_dir'])

        # run
        if config.train:

            try:
                main_train(data, model, trainer, trainer_config, checkpoint_dir)
            except Exception:
                melk_fn()
                raise

            # if not config.no_test and not trainer.interrupted:
            #     trainer.test(ckpt_path="best", datamodule=data)

        if config.inference and not trainer.interrupted:
            main_inference(data, model, trainer, config)

    except Exception as e:

        handle_exception(trainer, e, config.debug)

    finally:

        cleanup(trainer, config.runtime.debug,
                config.runtime.resume, config.runtime.logdir)


def main_train(data: lightning.LightningDataModule, model: lightning.LightningModule,
               trainer: lightning.Trainer, config, checkpoint_dir: str):

    cpu = config['params'].get("accelerator", "gpu") == "cpu"

    learning_rate = set_initial_learning_rate(trainer, cpu, batch_size=config.get("batch_size"),
                                              base_learning_rate=config.get("base_learning_rate"),
                                              trainer_config=config)



    model.learning_rate = learning_rate
    trainer.fit(model, data, ckpt_path="last")


def main_inference(data: lightning.LightningDataModule, model: lightning.LightningModule,
                   trainer: lightning.Trainer, config):

    ema_file = config['runtime']['logdir'] + '/checkpoint-ema/last.ckpt'
    # check if ema_file exists
    if os.path.exists(ema_file) and config['ema']:
        ckpt = ema_file
    else:
        ckpt = 'last'

    rank_zero_info(f'Using checkpoint: {ckpt}')

    trainer.test(model, data, ckpt_path=ckpt)
