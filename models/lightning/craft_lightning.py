"""
PyTorch Lightning wrapper used for training the CRAFT model
"""
from typing import Dict, Union, Tuple, List


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class CRAFTLightningWrapper(pl.LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        loss_args: Dict,
        visualisation_dataloader: DataLoader | None = None
    ):
        """
        Initialises a CRAFTLightningWrapper object, which supports the use of PyTorch lightning utilities (callbacks, logging) during training

        Args:
            net: The CRAFT model being trained
            loss_args: A dictionary used to specify the loss function and any associated parameters
            visualisation_data: Test set data used to visualise the bounding box changes after each epoch
        """
        super().__init__()
        self.net = net
        self.loss_args = loss_args
        self.visualisation_dataloader = visualisation_dataloader
        self.visualisation_output = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input batch, performs inference and computes the loss value using the provided loss arguments

        Args:
            batch: A dictionary containing images and their corresponding ground truth heatmaps (along with their resizing ratios which are not of use here)

        Returns:
            The loss values for the given batch
        """
        images = batch['image']
        gt_char_maps = batch['char_map']
        gt_affinity_maps = batch['affinity_map']

        preds, _ = self.net(images)
        preds = preds.permute(0, 3, 1, 2).contiguous()
        pred_char_maps = preds[:, 0, :, :]
        pred_affinity_maps = preds[:, 1, :, :]

        loss_fn = self.loss_args["loss_fn"]

        if loss_fn.__name__ == "CRAFT_loss":
            hard_loss = loss_fn(
                pred_char_map=pred_char_maps,
                pred_affinity_map=pred_affinity_maps,
                gt_char_map=gt_char_maps,
                gt_affinity_map=gt_affinity_maps,
                affinity_weight=self.loss_args["affinity_weight"]
            )
        else:
            hard_loss = loss_fn(
                pred_char_map=pred_char_maps,
                pred_affinity_map=pred_affinity_maps,
                gt_char_map=gt_char_maps,
                gt_affinity_map=gt_affinity_maps,
                neg_ratio=self.loss_args["neg_ratio"],
                num_min_neg=self.loss_args["num_min_neg"]
            )

        soft_loss = torch.tensor(0.0)
        alpha = 0
        if self.loss_args.get("distillation", False):

            teacher = self.loss_args["teacher"]
            temperature = self.loss_args["temperature"]
            alpha = self.loss_args["alpha"]

            with torch.no_grad():
                soft_targets, _ = teacher(images)
                soft_targets = soft_targets.permute(0, 3, 1, 2).contiguous()

            pred_char_maps_sigmoid = torch.sigmoid(
                pred_char_maps / temperature
            )
            pred_affinity_maps_sigmoid = torch.sigmoid(
                pred_affinity_maps / temperature
            )

            soft_char_maps_sigmoid = torch.sigmoid(
                soft_targets[:, 0, :, :] / temperature
            )
            soft_affinity_maps_sigmoid = torch.sigmoid(
                soft_targets[:, 1, :, :] / temperature
            )

            if loss_fn.__name__ == "CRAFT_loss":
                soft_loss = (temperature ** 2) * loss_fn(
                    pred_char_map=pred_char_maps_sigmoid,
                    pred_affinity_map=pred_affinity_maps_sigmoid,
                    gt_char_map=soft_char_maps_sigmoid,
                    gt_affinity_map=soft_affinity_maps_sigmoid,
                    affinity_weight=self.loss_args["affinity_weight"]
                )
            else:
                soft_loss = (temperature ** 2) * loss_fn(
                    pred_char_map=pred_char_maps_sigmoid,
                    pred_affinity_map=pred_affinity_maps_sigmoid,
                    gt_char_map=soft_char_maps_sigmoid,
                    gt_affinity_map=soft_affinity_maps_sigmoid,
                    neg_ratio=self.loss_args["neg_ratio"],
                    num_min_neg=self.loss_args["num_min_neg"],
                )

        total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
        return total_loss, hard_loss, soft_loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Given an input batch, performs inference and computes the training loss (total, hard, and soft). At the end of each epoch, the epoch losses are calculated and logged via the lightning logger.

        Args:
            batch: A dictionary containing images and their corresponding ground truth heatmaps (along with their resizing ratios which are not of use here)
            batch_idx: The index of the batch being processed

        Returns:
            The total loss for the given input batch
        """
        total_loss, hard_loss, soft_loss = self.compute_loss(batch)
        self.log(
            "train_loss_total",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            "train_loss_hard",
            hard_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss_soft",
            soft_loss,
            on_step=False,
            on_epoch=True,
        )
        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Given an input batch, performs inference and computes the validation loss. At the end of each epoch, the epoch loss is calculated and logged via the lightning logger.

        Args:
            batch: A dictionary containing images and their corresponding ground truth heatmaps (along with their resizing ratios which are not of use here)
            batch_idx: The index of the batch being processed

        Returns:
            The total loss for the given input batch
        """
        total_loss, _, _ = self.compute_loss_components(batch)
        self.log(
            "val_loss_total",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return total_loss

    def on_validation_epoch_end(self) -> None:
        """
        Performs inference on the test set at the end of each epoch. The bounding box predictions are later used by the BoundingBoxLoggerCallback to visualise bounding box evolutions
        """
        if not self.visualisation_dataloader:
            return

        self.net.eval()
        with torch.no_grad():
            for batch in self.visualisation_dataloader:
                batch_size = batch["image"].shape[0]
                images = batch["image"]
                preds, _ = self.net(images)
                for image_num in range(batch_size):
                    self.visualisation_output.append(
                        {
                            "image": images[image_num],
                            "pred": preds[image_num].detach().cpu(),
                            "ratio_w": batch["ratio_w"][image_num].item(),
                            "ratio_h": batch["ratio_h"][image_num].item()
                        }
                    )
