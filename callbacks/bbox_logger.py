"""
PyTorch Lightning callback used to log annotated images for visual inspection of training progress
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger

from utils.craft_utils import adjustResultCoordinates, getDetBoxes
from utils.file_utils import saveResult


class BoundingBoxLoggerCallback(Callback):

    def __init__(self, text_threshold: float, link_threshold: float, low_text: float) -> None:
        """
        Initialises a BoundingBoxLoggerCallback object, which annotates and logs test set images with bounding box predictions

        Args:
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold
            low_text: Text lower-bound score
        """
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, net: pl.LightningModule
    ) -> None:
        """
        Given a list of visualisation outputs (stored in the lightning module), annotates the images with bounding boxes and saves the files, logging to MLFlow if possible

        Args:
            trainer: The PyTorch lightning trainer orchestrating the model training
            net: The network being trained
        """

        current_epoch = trainer.current_epoch
        outputs = net.visualisation_output

        save_dir = f"visualisation/{current_epoch}"
        os.makedirs(save_dir, exist_ok=True)

        for output_num, output in enumerate(outputs):

            score_text = output["pred"][0, :, :, 0].numpy()
            score_link = output["pred"][0, :, :, 1].numpy()

            boxes, _ = getDetBoxes(
                textmap=score_text,
                linkmap=score_link,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                low_text=self.low_text,
                poly=False
            )

            boxes = adjustResultCoordinates(
                boxes, output["ratio_w"], output["ratio_h"]
            )

            image_path = f"{output_num}.png"
            saveResult(save_dir, output["image"], boxes, image_path)

            if isinstance(trainer.logger, MLFlowLogger):
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id,
                    f"{save_dir}/{image_path}",
                    artifact_path=f"epoch_{current_epoch}"
                )

        net.visualisation_output.clear()
