"""
PyTorch Lightning callback used to check whether a network contains NaN or Inf values
"""

from typing import Dict, Tuple

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import Callback
import torch


class DetectNaNCallback(Callback):

    def __init__(self):
        """
        Initialises a DetectNaNCallBack object, which is used to log NaN/Inf values in the activations, weights, and gradients
        """
        super().__init__()
        self.hooks = []
        self.logs = []

    def check_activation_hook(
        self,
        layer: torch.nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
    ) -> None:
        """
        Given a network layer and its outputs, checks whether the activations contain NaN

        Args:
            layer: The layer of the neural network being checked for NaNs
            input: The input to the model layer
            output: The output/activations of the model layer
        """
        if torch.isnan(output).any() or torch.isinf(output).any():
            self.logs.append(
                f"NaN/Inf in the activations of layer: {layer.__class__.__name__}"
            )

    def on_fit_start(
        self, trainer: pl.Trainer, net: pl.LightningModule
    ) -> None:
        """
        Attaches the check_activation_hook to each of the network layers before training commences

        Args:
            trainer: The PyTorch lightning trainer orchestrating the model training
            net: The network being trained
        """
        for layer in net.modules():
            hook = layer.register_forward_hook(self.check_activation_hook)
            self.hooks.append(hook)

    def on_after_backward(
        self, trainer: pl.Trainer,  net: pl.LightningModule
    ) -> None:
        """
        Checks whether a neural network contains any NaN or Inf gradient values after the gradient calculation. If detected, flags the trainer to stop model training

        Args:
            net: The neural network being checked for NaN/Inf
            trainer: The PyTorch lightning trainer orchestrating the model training
        """
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    trainer.should_stop = True
                    self.logs.append(f"NaN/Inf detected in grad: {name}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        net: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Checks whether a neural network contains any NaN or Inf weight values after the parameter updates. If detected, flags the trainer to stop model training.

        Args:
            trainer: The PyTorch lightning trainer orchestrating the model training
            net: The neural network being checked for NaN/Inf
            outputs: The output of the model for the most recent batch
            batch: The most recent batch
            batch_idx: The index of the most recent batch
        """
        for name, param in net.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                trainer.should_stop = True
                self.logs.append(f"NaN/Inf detected in weight: {name}")

    def on_train_end(self, trainer: pl.Trainer, net: pl.LightningModule) -> None:
        """
        If given a compatible logger (i.e MLFlowLogger), logs the NaN/Inf warning messages.

        Args:
            trainer: The PyTorch lightning trainer orchestrating the model training
            net: The neural network being checked for NaN/Inf
        """
        if isinstance(trainer.logger, MLFlowLogger):
            for log in self.logs:
                trainer.logger.experiment.log_text("nan_logs", log)
            self.logs.clear()

    def on_fit_end(self, trainer: pl.Trainer, net: pl.LightningModule) -> None:
        """
        Removes the NaN/Inf hooks from the model once training has concluded

        Args:
            trainer: The PyTorch lightning trainer orchestrating the model training
            net: The neural network being checked for NaN/Inf
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
