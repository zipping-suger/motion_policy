from typing import Optional, Any, Dict, List
from pathlib import Path
import sys
import os
from datetime import timedelta
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from termcolor import colored
import argparse
import yaml
import uuid

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataModule


def import_training_policy_net(mode):
    if mode == "finetune" or mode == "finetune_tasks":
        from models.policynet_opt import TrainingPolicyNet
    elif mode == "pretrain":
        from models.policynet import TrainingPolicyNet
    else:
        raise ValueError(f"Unknown training mode: {mode}. Expected 'finetune' or 'pretrain'.")
    return TrainingPolicyNet


def setup_trainer(
    accelerator: str,
    gpu_num: int,
    test: bool,
    should_checkpoint: bool,
    logger: Optional[WandbLogger],
    checkpoint_interval: int,
    checkpoint_dir: str,
    validation_interval: float,
) -> pl.Trainer:
    """
    Creates the Pytorch Lightning trainer object

    :param accelerator str: The accelerator to use ("cpu", "gpu", "tpu", etc.)
    :param gpu_num int: The number of gpu_num (if more than 1, uses DDP)
    :param test bool: Whether to use a test dataset
    :param should_checkpoint bool: Whether to save checkpoints
    :param logger Optional[WandbLogger]: The logger object, set to None if logging is disabled
    :param checkpoint_interval int: The number of minutes between checkpoints
    :param checkpoint_dir str: The directory in which to save checkpoints (a subdirectory will
                               be created according to the experiment ID)
    :param validation_interval float: How often to run the validation step, either as a proportion
                                      of the training epoch or as a number of batches
    :rtype pl.Trainer: The trainer object
    """
    args: Dict[str, Any] = {}

    if test:
        args = {**args, "limit_train_batches": 10, "limit_val_batches": 3}
        validation_interval = 2  # Overwritten to be an appropriate size for test
    if accelerator == "gpu":
        if (isinstance(gpu_num, list) and len(gpu_num) > 1) or (
            isinstance(gpu_num, int) and gpu_num > 1
        ):
            args = {
                **args,
                "strategy": DDPStrategy(find_unused_parameters=False),
            }
        if validation_interval is not None:
            args = {**args, "val_check_interval": validation_interval}
        callbacks: List[Callback] = []
        if logger is not None:
            experiment_id = str(logger.experiment.id)
        else:
            experiment_id = str(uuid.uuid1())
        if should_checkpoint:
            if checkpoint_dir is not None:
                dirpath = Path(checkpoint_dir).resolve() / experiment_id
            else:
                dirpath = PROJECT_ROOT / "checkpoints" / experiment_id
            pl.utilities.rank_zero_info(f"Saving checkpoints to {dirpath}")
            every_n_checkpoint = ModelCheckpoint(
                monitor="avg_target_error",  # <-- changed from "train_loss"
                save_last=True,
                dirpath=dirpath,
                train_time_interval=timedelta(minutes=checkpoint_interval),
            )
            epoch_end_checkpoint = ModelCheckpoint(
                monitor="avg_target_error",  # <-- changed from "train_loss"
                save_last=True,
                dirpath=dirpath,
                save_on_train_epoch_end=True,
            )
            epoch_end_checkpoint.CHECKPOINT_NAME_LAST = "epoch-{epoch}-end"
            callbacks.extend([every_n_checkpoint, epoch_end_checkpoint])

    trainer = pl.Trainer(
        enable_checkpointing=should_checkpoint,
        callbacks=callbacks,
        max_epochs=1 if test else 100,
        gradient_clip_val=1.0,
        accelerator=accelerator,
        precision=16,
        # num_sanity_val_steps=0, # Skip validation sanity check
        logger=False if logger is None else logger,
        **args,
    )
    return trainer


def setup_logger(
    should_log: bool, experiment_name: str, config_values: Dict[str, Any]
) -> Optional[WandbLogger]:
    if not should_log:
        pl.utilities.rank_zero_info("Disabling all logs")
        return None
    logger = WandbLogger(
        name=experiment_name,
        project="learn_to_reach",
        log_model=True,
    )
    logger.log_hyperparams(config_values)
    return logger


def parse_args_and_configuration():
    """
    Checks the command line arguments and merges them with the configuration yaml file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test with only a few batches (disables logging)",
    )
    parser.add_argument(
        "--no-logging", action="store_true", help="Don't log to weights and biases"
    )
    parser.add_argument(
        "--no-checkpointing", action="store_true", help="Don't checkpoint"
    )
    args = parser.parse_args()

    if args.test:
        args.no_logging = True

    with open(args.yaml_config) as f:
        configuration = yaml.safe_load(f)

    return {
        "training_node_name": os.uname().nodename,
        **configuration,
        **vars(args),
    }


def run():
    """
    Runs the training procedure
    """
    config = parse_args_and_configuration()

    color_name = colored(config["experiment_name"], "green")
    pl.utilities.rank_zero_info(f"Experiment name: {color_name}")
    logger = setup_logger(
        not config["no_logging"],
        config["experiment_name"],
        config,
    )

    trainer = setup_trainer(
        config["accelerator"],
        config["gpu_num"],
        config["test"],
        should_checkpoint=not config["no_checkpointing"],
        logger=logger,
        checkpoint_interval=config["checkpoint_interval"],
        checkpoint_dir=config["save_checkpoint_dir"],
        validation_interval=config["validation_interval"],
    )
    dm = DataModule(
        batch_size=config["batch_size"],
        train_mode= config["train_mode"],
        **(config["shared_parameters"] or {}),
        **(config["data_module_parameters"] or {}),
    )
      
    mode = config["train_mode"]
    TrainingPolicyNet = import_training_policy_net(mode)
    
    # Initialize the model
    mdl = TrainingPolicyNet(
        **(config["shared_parameters"] or {}),
        **(config["training_model_parameters"] or {}),
    )
    if config["model_path"] is None:
        print("Training from scratch")
        mdl = TrainingPolicyNet(
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
        )
    else:
        print(f"Loading model from {config['model_path']}")
        mdl = TrainingPolicyNet.load_from_checkpoint(
            config["model_path"],
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
        )
    
    if logger is not None:
        logger.watch(mdl, log="gradients", log_freq=10)
    trainer.fit(model=mdl, datamodule=dm)


if __name__ == "__main__":
    run()