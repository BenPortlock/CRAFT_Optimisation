"""
PyTorch Lightning callback for monitoring the gradient magnitude
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
import torch


class GradientMonitorCallback(Callback):

    def on_after_backward(
        self, trainer: pl.Trainer, net: pl.LightningModule
    ) -> None:
        """
        Given a neural network, logs the L2 norm of the gradient

        Args:
            trainer: The PyTorch lightning trainer orchestrating the model training
            net: The network being trained
        """
        total_norm_sq = 0.0
        for p in net.parameters():
            if p.grad is not None:
                grad = p.grad.detach()
                total_norm_sq += float(grad.norm(2).item()) ** 2
        total_norm = total_norm_sq ** 0.5
        if isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_metric(
                key="grad_norm",
                value=total_norm,
                step=trainer.global_step
            )
