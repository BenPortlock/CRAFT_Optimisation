"""
Script used to create CRAFT-style heatmaps for a set of images
"""

import argparse
import os
from pathlib import Path
import random
from typing import List, Tuple

import albumentations as A
import cv2
from loguru import logger
import numpy as np

from data.heatmap_generator import get_heatmaps
from utils.file_utils import get_files
from utils.imgproc import loadImage, preprocess_image, cvt2HeatmapImg


def OBB_labels_to_CRAFT(
    labels: List[np.ndarray],
    scale_w: int,
    scale_h: int,
    ratio_w: float,
    ratio_h: float
) -> np.ndarray:
    """
    Given a list of YOLO-style OBB character quadrilaterals, denormalises the coordinates to reflect image size, resizes them to match the target canvas size, and downscales them by 2x to match the model output

    Args:
        labels: A list of character quadrilaterals, where each is a 4x2 array of normalised coordinates structured as top left -> top right -> bottom right -> bottom left, and each point is [col, row]
        scale_w: The width of the original image
        scale_h: The height of the original image
        ratio_w: The width resizing ratio required for bounding box annotation
        ratio_h: The height resizing ratio required for bounding box annotation

    Returns:
        An array of character quadrilaterals formatted for CRAFT compatibility
    """
    labels_arr = np.array(labels, dtype=np.float32)
    denormalised_char_labels = labels_arr * [scale_w, scale_h]
    resized_char_labels = denormalised_char_labels / [ratio_w, ratio_h]
    downscaled_char_labels = resized_char_labels / 2
    return downscaled_char_labels.astype(np.float32)


def augment_image_and_boxes(
    image: np.ndarray, boxes: np.ndarray, augmenter: A.Compose
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a set of augmentations to an image and its character quadrilaterals, returning the transformed results.

    Args:
        image: A numpy array representing the image to be augmented
        boxes: An array of character quadrilaterals, where each is a 4x2 array structured as top left -> top right -> bottom right -> bottom left, and each point is [col, row]
        augmenter: An albumentations.Compose object containing a list of possible augmentations

    Returns:
        The augmented image and its corresponding character quadrilaterals
    """
    h, w = image.shape[:2]
    boxes = np.clip(boxes, [0, 0], [w-1, h-1])
    keypoints = [tuple(point) for quad in boxes for point in quad]
    res = augmenter(image=image, keypoints=keypoints)
    new_image = res['image']
    new_keypoints = np.array(res['keypoints']).reshape(-1, 4, 2)
    return new_image, new_keypoints


def main(args: argparse.Namespace) -> None:

    augmenter = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(blur_limit=(3, 5), p=0.3),
            A.RandomRotate90(p=1),
            A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
        ],
        keypoint_params=A.KeypointParams(format="xy"))

    logger.info(f"{'-'*8} Loading Training Data {'-'*8}")
    training_data = []
    image_list, _, gt_files = get_files(args.data_dir)

    for image_path in (image_list):
        image = loadImage(image_path)
        image_name = Path(image_path).stem

        char_label_path = next(
            (f for f in gt_files if "char_labels" in f and image_name == Path(f).stem), None
        )
        word_group_path = next(
            (f for f in gt_files if "word_group_labels" in f and image_name ==
             Path(f).stem), None
        )
        if not (char_label_path and word_group_path):
            logger.warning(f"Skipping {image_name} – missing labels.")
            continue

        with open(char_label_path, "r") as char_labels_file:
            labels = char_labels_file.readlines()

        char_box_list = []
        for line in labels:
            coords = line.strip().split()
            if len(coords) != 9:
                continue
            coords = np.array(coords[1:], dtype=np.float32).reshape(4, 2)
            char_box_list.append(coords)
        if not char_box_list:
            logger.warning(
                f"{image_name} has the wrong/no char labels. Ignoring."
            )
            continue

        with open(word_group_path, "r") as word_groups_file:
            words = word_groups_file.readlines()

        word_group_list = []
        for line in words:
            word_indices = line.strip().split(",")[1:]
            word_group_list.append(np.array(word_indices, dtype=int))
        if not word_group_list:
            logger.warning(
                f"{image_name} has the no word groups. Ignoring."
            )
            continue

        training_data.append(
            {
                "image_name": image_name,
                "image": image,
                "char_labels": char_box_list,
                "word_groups": word_group_list
            }
        )

    logger.info(f"{'-'*8} All Files Processed - Creating Heatmaps {'-'*8}")

    random.shuffle(training_data)
    split_idx = int(len(training_data) * args.train_split)
    split_data = {
        "train": training_data[:split_idx],
        "val": training_data[split_idx:]
    }

    for split_name, dataset in split_data.items():

        logger.info(f"{'-'*8} Creating Heatmaps For {split_name} {'-'*8}")

        out_path = f"{args.output_dir}/{split_name}"
        output_dirs = {
            "images": f"{out_path}/images",
            "char_maps": f"{out_path}/char_maps",
            "affinity_maps": f"{out_path}/affinity_maps"
        }
        for dir in output_dirs.values():
            os.makedirs(dir, exist_ok=True)

        for data in dataset:

            image = data["image"]
            image_name = data["image_name"]
            original_h, original_w = image.shape[:2]

            _, ratio_w, ratio_h = preprocess_image(
                image=image,
                canvas_size=args.canvas_size,
                mag_ratio=args.mag_ratio,
            )

            converted_char_labels = OBB_labels_to_CRAFT(
                labels=data["char_labels"],
                scale_w=original_w,
                scale_h=original_h,
                ratio_w=ratio_w,
                ratio_h=ratio_h
            )

            char_map, affinity_map = get_heatmaps(
                output_size=args.canvas_size // 2,
                char_quads=converted_char_labels,
                word_groups=data["word_groups"],
                kernel_size=64,
                sigma=0.4
            )

            cv2.imwrite(
                f"{output_dirs['images']}/{image_name}.jpg", image[:, :, ::-1]
            )
            logger.info(f"{output_dirs['images']}/{image_name} saved")

            np.save(f"{output_dirs['char_maps']}/{image_name}.npy", char_map)
            logger.info(f"{output_dirs['char_maps']}/{image_name} saved")

            np.save(
                f"{output_dirs['affinity_maps']}/{image_name}.npy", affinity_map
            )
            logger.info(f"{output_dirs['affinity_maps']}/{image_name} saved")

            if split_name == "train":

                augmented_image_name = f"{image_name}_augmented"

                denormalised_char_labels = np.array(
                    data["char_labels"], dtype=np.float32
                ) * [original_w, original_h]

                augmented_image, augmented_char_labels = augment_image_and_boxes(
                    image, denormalised_char_labels, augmenter
                )

                augmented_ratios = [
                    ratio_w, ratio_h] if augmented_image.shape == image.shape else [ratio_h, ratio_w]

                converted_augmented_char_labels = (
                    augmented_char_labels / augmented_ratios / 2
                ).astype(np.float32)

                augmented_char_map, augmented_affinity_map = get_heatmaps(
                    output_size=args.canvas_size // 2,
                    char_quads=converted_augmented_char_labels,
                    word_groups=data["word_groups"],
                    kernel_size=64,
                    sigma=0.4
                )

                cv2.imwrite(
                    f"{output_dirs['images']}/{augmented_image_name}.jpg",
                    augmented_image[:, :, ::-1]
                )
                logger.info(
                    f"{output_dirs['images']}/{augmented_image_name} saved"
                )

                np.save(
                    f"{output_dirs['char_maps']}/{augmented_image_name}.npy",
                    augmented_char_map
                )
                logger.info(
                    f"{output_dirs['char_maps']}/{augmented_image_name} saved"
                )

                np.save(
                    f"{output_dirs['affinity_maps']}/{augmented_image_name}.npy",
                    augmented_affinity_map
                )
                logger.info(
                    f"{output_dirs['affinity_maps']}/{augmented_image_name} saved"
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate CRAFT heatmaps for dataset")
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Folder containing intended training/validation data"
    )
    parser.add_argument(
        "--output_dir",
        default="train_val_data",
        type=str,
        help="Output folder to save train/val images and labels"
    )
    parser.add_argument(
        "--train_split",
        default=0.8,
        type=float,
        help="Proportion of data allocated for training (rest for val)"
    )
    parser.add_argument(
        "--canvas_size",
        default=1280,
        type=int,
        help="Image size to be used for training"
    )
    parser.add_argument(
        "--mag_ratio",
        default=1.5,
        type=float,
        help="Image magnification ratio"
    )
    args = parser.parse_args()
    main(args)
