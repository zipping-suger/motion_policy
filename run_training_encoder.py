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
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataModule
from models.encoderent import TrainingEncoderNet  # Import the encoder model

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
    """
    args: Dict[str, Any] = {}
    callbacks: List[Callback] = []

    if test:
        args = {**args, "limit_train_batches": 10, "limit_val_batches": 3}
        validation_interval = 2  # Overwritten for test mode
    
    # Setup logging and checkpointing
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
        
        # Checkpoint callback based on validation loss
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename="encoder-{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Multi-GPU setup
    if accelerator == "gpu":
        if (isinstance(gpu_num, list) and len(gpu_num) > 1) or (
            isinstance(gpu_num, int) and gpu_num > 1
        ):
            args = {**args, "strategy": DDPStrategy(find_unused_parameters=False)}
    
    if validation_interval is not None:
        args = {**args, "val_check_interval": validation_interval}

    trainer = pl.Trainer(
        enable_checkpointing=should_checkpoint,
        callbacks=callbacks,
        max_epochs=1 if test else 100,
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=gpu_num,
        precision="16-mixed" if accelerator == "gpu" else 32,
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
        project="encoder_training",
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
    Runs the training procedure for the encoder model
    """
    config = parse_args_and_configuration()

    color_name = colored(config["experiment_name"], "green")
    pl.utilities.rank_zero_info(f"Training Encoder - Experiment: {color_name}")
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
    
    # Setup data module
    dm = DataModule(
        batch_size=config["batch_size"],
        train_mode= config["train_mode"],
        **(config["shared_parameters"] or {}),
        **(config["data_module_parameters"] or {}),
    )
    
    # Initialize the encoder model
    if config["model_path"] is None:
        pl.utilities.rank_zero_info("Training encoder from scratch")
        mdl = TrainingEncoderNet(
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
        )
    else:
        pl.utilities.rank_zero_info(f"Loading encoder from {config['model_path']}")
        mdl = TrainingEncoderNet.load_from_checkpoint(
            config["model_path"],
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
        )
    
    # Watch gradients if logging is enabled
    if logger is not None:
        logger.watch(mdl, log="gradients", log_freq=100)
    
    # Start training
    trainer.fit(model=mdl, datamodule=dm)

if __name__ == "__main__":
    run()