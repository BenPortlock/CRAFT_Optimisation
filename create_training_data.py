import os
import cv2
import numpy as np
import argparse
import imgproc
import file_utils
from optimisation import preprocess_image
from pathlib import Path


def generate_isotropic_gaussian(kernel_size=64, sigma=0.5):

    values = np.linspace(-1.0, 1.0, kernel_size)
    x_coords, y_coords = np.meshgrid(values, values)
    dist_sqd = x_coords * x_coords + y_coords * y_coords
    gaussian = np.exp(-dist_sqd / (2.0 * (sigma ** 2))).astype(np.float32)
    gaussian /= gaussian.max()
    return gaussian


def warp_and_paste_gaussian(canvas, quad_coords, kernel):

    kernel_h, kernel_w = kernel.shape
    target_h, target_w = canvas.shape
    source_coords = np.array(
        [
            [0, 0],
            [kernel_w-1, 0],
            [kernel_w-1, kernel_h-1],
            [0, kernel_h-1]
        ],
        dtype=np.float32
    )
    target_coords = np.array(quad_coords, dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_coords, target_coords)
    warped_gaussian = cv2.warpPerspective(
        kernel, transform, (target_w, target_h), flags=cv2.INTER_LINEAR
    )
    np.maximum(canvas, warped_gaussian, out=canvas)
    return canvas


def order_coords_clockwise(coords):

    y_sorted = coords[np.argsort(coords[:, 1])]
    top, bottom = y_sorted[:2], y_sorted[2:]
    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
    return np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )


def generate_affinity_quad(char_quad_1, char_quad_2):

    # Compute character centers
    c1 = char_quad_1.mean(axis=0)
    c2 = char_quad_2.mean(axis=0)

    # Direction vector from char 1 to char 2
    vec = c2 - c1
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        vec_norm = 1.0  # avoid division by zero
    vec = vec / vec_norm

    # Perpendicular vector for width
    perp = np.array([-vec[1], vec[0]])

    # Approximate half-widths of each character
    half_width1 = np.linalg.norm(char_quad_1[0] - char_quad_1[3]) / 2
    half_width2 = np.linalg.norm(char_quad_2[0] - char_quad_2[3]) / 2

    # Construct the four corners of the affinity box
    top_left = c1 + perp * half_width1
    bottom_left = c1 - perp * half_width1
    top_right = c2 + perp * half_width2
    bottom_right = c2 - perp * half_width2

    affinity_quad = np.stack(
        [top_left, top_right, bottom_right, bottom_left], axis=0)
    return affinity_quad


def convert_labels_to_CRAFT(labels, original_h, original_w, ratio_w, ratio_h):

    labels_arr = np.array(labels, dtype=np.float32)
    denormalised_char_labels = labels_arr * [original_h, original_w]
    resized_char_labels = denormalised_char_labels / [ratio_w, ratio_h]
    downscaled_char_labels = resized_char_labels / 2
    return downscaled_char_labels


def generate_gt_maps(model_output_size, char_quads, word_groups, kernel_size=64, sigma=0.5):

    h = w = model_output_size
    kernel = generate_isotropic_gaussian(kernel_size, sigma)

    char_map = np.zeros((h, w), dtype=np.float32)
    for char in char_quads:
        warp_and_paste_gaussian(char_map, char, kernel)

    affinity_map = np.zeros((h, w), dtype=np.float32)
    affinity_quads = []
    for word in word_groups:
        for i in range(len(word)-1):
            char_1, char_2 = char_quads[word[i:i+2]]
            char_1 = order_coords_clockwise(char_1)
            char_2 = order_coords_clockwise(char_2)
            affinity_quad = generate_affinity_quad(char_1, char_2)
            warp_and_paste_gaussian(affinity_map, affinity_quad, kernel)
            affinity_quads.append(affinity_quad)

    return char_map, affinity_map, affinity_quads


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_dir",
        required=True,
        type=str,
        help="Folder containing training data"
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

    print("Loading Training Data")
    training_data = []
    image_list, _, gt_files = file_utils.get_files(args.training_dir)
    for k, image_path in enumerate(image_list):
        image = imgproc.loadImage(image_path)
        image_name = Path(image_path).stem

        ground_truths = [f for f in gt_files if image_name in f]
        if len(ground_truths) != 2:
            print(f"{image_name} does not have labels. Ignoring.")
            continue

        char_labels = word_groups = None
        for file in ground_truths:
            if "char_labels" in file:
                char_labels = file
            elif "word_group_labels" in file:
                word_groups = file
        if not char_labels or not word_groups:
            print(f"{image_name} has the wrong label files. Ignoring.")
            continue

        with open(char_labels, "r") as char_labels_file:
            labels = char_labels_file.readlines()

        char_box_list = []
        for line in labels:
            coords = line.strip().split()
            if len(coords) != 9:
                continue
            coords = np.array(coords[1:], dtype=np.float32).reshape(4, 2)
            char_box_list.append(coords)
        if not char_box_list:
            print(f"{image_name} has the wrong/no char labels. Ignoring.")
            continue

        with open(word_groups, "r") as word_groups_file:
            words = word_groups_file.readlines()

        word_group_list = []
        for line in words:
            word_indices = line.strip().split(",")[1:]
            word_group_list.append(np.array(word_indices, dtype=int))
        if not word_group_list:
            print(f"{image_name} has the no word groups. Ignoring.")
            continue

        training_data.append(
            {
                "image_name": image_name,
                "image": image,
                "char_labels": char_box_list,
                "word_groups": word_group_list
            }
        )

    print("Creating Ground Truth Heatmaps")
    for data in training_data:
        image = data["image"]
        image_name = data["image_name"]
        original_h, original_w = image.shape[:2]
        preprocessed_image, ratio_w, ratio_h = preprocess_image(
            image,
            args.canvas_size,
            args.mag_ratio,
            "cpu"
        )
        converted_char_labels = convert_labels_to_CRAFT(
            data["char_labels"],
            original_w,
            original_h,
            ratio_w,
            ratio_h
        )
        char_map, affinity_map, affinity_quads = generate_gt_maps(
            args.canvas_size // 2,
            converted_char_labels,
            data["word_groups"]
        )

        char_map_dir = f"{args.training_dir}/char_maps"
        if not os.path.isdir(char_map_dir):
            os.mkdir(char_map_dir)
        char_map_filename = f"{char_map_dir}/{image_name}.npy"
        np.save(char_map_filename, char_map)
        print(f"{char_map_filename} saved")

        affinity_map_dir = f"{args.training_dir}/affinity_maps"
        if not os.path.isdir(affinity_map_dir):
            os.mkdir(affinity_map_dir)
        affinity_map_filename = f"{affinity_map_dir}/{image_name}.npy"
        np.save(affinity_map_filename, affinity_map)

    print("Ground Truth Heatmaps Created and Saved")
