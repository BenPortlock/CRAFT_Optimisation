"""
Script used to train the CRAFT text detection model
"""

import argparse
import os
import yaml

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader

from callbacks.bbox_logger import BoundingBoxLoggerCallback
from callbacks.grad_monitor import GradientMonitorCallback
from callbacks.nan_detect import DetectNaNCallback
from data.craft_dataset import CraftDataset
from loss.loss import CRAFT_loss, OHEM_loss
from models.architecture.craft import CRAFT
from models.lightning.craft_lightning import CRAFTLightningWrapper
from utils.craft_utils import copyStateDict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the YAML file containing args for model training"
    )
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    experiment_name = config.experiment_name
    checkpoint_dir = config.checkpoint_dir
    log_uri = config.log_results.log_uri if config.log_results.enabled else None
    visualise = config.log_results.visualise.enabled
    epochs = config.epochs
    batch_size = config.batch_size
    data_dir = config.data_dir
    canvas_size = config.canvas_size
    mag_ratio = config.mag_ratio
    device = config.device
    lr = config.lr

    #! LOADING TRAINING MODEL
    logger.info(f"{'-'*8} Loading CRAFT Model {'-'*8}")
    model_path = config.model.path
    if not config.model.state_dict:
        net = torch.load(model_path, map_location=device)
    else:
        net = CRAFT()
        net.load_state_dict(
            copyStateDict(torch.load(model_path, map_location=device))
        )
    net.to(device)

    loss_args = {}

    #! CONFIGURING LOSS PARAMETERS
    if config.loss.CRAFT_loss.enabled:
        loss_args["loss_fn"] = CRAFT_loss
        loss_args["affinity_weight"] = config.loss.CRAFT_loss.affinity_weight
    else:
        loss_args["loss_fn"] = OHEM_loss
        loss_args["neg_ratio"] = config.loss.OHEM_loss.neg_ratio
        loss_args["num_min_neg"] = config.loss.OHEM_loss.num_min_neg

    if config.loss.distillation.enabled:
        logger.info(f"{'-'*8} Loading Teacher Model {'-'*8}")
        loss_args["distillation"] = True
        loss_args["alpha"] = config.loss.distillation.alpha
        loss_args["temperature"] = config.loss.distillation.temperature
        teacher_model_path = config.loss.distillation.teacher.model_path
        if not config.loss.distillation.teacher.state_dict:
            teacher = torch.load(teacher_model_path, map_location=device)
        else:
            teacher = CRAFT()
            net.load_state_dict(
                copyStateDict(
                    torch.load(teacher_model_path, map_location=device)
                )
            )
        teacher.to(device)
        teacher.eval()
        teacher.requires_grad_(False)
        loss_args["teacher"] = teacher

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    #! CREATING DATALOADERS
    train_dataset = CraftDataset(
        data_dir=f"{data_dir}/train",
        canvas_size=canvas_size,
        mag_ratio=mag_ratio)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=-1
    )

    val_dataset = CraftDataset(
        data_dir=f"{data_dir}/val",
        canvas_size=canvas_size,
        mag_ratio=mag_ratio)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=-1
    )

    test_dataloader = None
    if visualise:
        test_dataset = CraftDataset(
            data_dir=f"{data_dir}/test",
            canvas_size=canvas_size,
            mag_ratio=mag_ratio
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.log_results.visualise.batch_size,
            shuffle=False,
            num_workers=-1
        )

    #! CREATING PYTORCH LIGHTNING WRAPPER
    lightning_net = CRAFTLightningWrapper(net, loss_args, test_dataloader)

    #! CONFIGURING PYTORCH LIGHTNING CALLBACKS
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename="craft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
            save_weights_only=False
        )
    )
    callbacks.append(DetectNaNCallback())
    callbacks.append(GradientMonitorCallback())
    if visualise:
        callbacks.append(
            BoundingBoxLoggerCallback(
                text_threshold=config.log_results.visualise.text_threshold,
                link_threshold=config.log_results.visualise.link_threshold,
                low_text=config.log_results.visualise.low_text,
            )
        )

    #! INITIALISING LOGGER
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name, tracking_uri=log_uri
    )

    #! INITIALISING TRAINER
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        logger=mlflow_logger,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    logger.info(f"{'-'*8} Beginning Training {'-'*8}")
    trainer.fit(lightning_net, train_dataloader, val_dataloader)

    try:
        save_file_name = f"{checkpoint_dir}/final.pth"
        torch.save(net.state_dict(), save_file_name)
        if isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(
                local_path=save_file_name,
                artifact_path="epoch_bbox_visualisations"
            )
        os.remove(save_file_name)
    except Exception as e:
        logger.warning(f"Failed to save final model to mlflow: {e}")
