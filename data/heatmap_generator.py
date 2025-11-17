"""
Functions required to create gaussian-based (character and affinity) heatmaps for a given image
"""

from typing import List, Tuple

import cv2
import numpy as np


def get_isotropic_gaussian(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Generates a 2D, circular gaussian heatmap

    Args:
        kernel_size: The diameter of the generated gaussian
        sigma: The standard deviation of the generated gaussian

    Returns:
        An array of shape (kernel_size x kernel_size) containing a circular gaussian heatmap
    """
    values = np.linspace(-1.0, 1.0, kernel_size)
    x_coords, y_coords = np.meshgrid(values, values)
    gaussian = np.exp(
        -(x_coords ** 2 + y_coords ** 2) / (2 * sigma ** 2)
    ).astype(np.float32)

    mask = np.zeros((kernel_size, kernel_size), np.float32)
    centre = (kernel_size // 2, kernel_size // 2)
    radius = kernel_size // 2
    cv2.circle(mask, center=centre, radius=radius, thickness=-1, color=1)

    gaussian_circle = gaussian * mask
    gaussian_circle /= gaussian_circle.max()
    return gaussian_circle


def warp_and_paste_gaussian(
    canvas: np.ndarray, quad_coords: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """
    Warps a gaussian kernel to fit the dimensions of a bounding box, before pasting the result on the provided canvas

    Args:
        canvas: The template/background of the heatmap
        quad_coords: The coordinates of the target to which the gaussian kernel will be warped. 4x2 shape structured as top left -> top right -> bottom right -> bottom left
        kernel: The gaussian kernel to be warped

    Returns:
        The updated canvas containing the warped gaussian heatmap
    """
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
    transform = cv2.getPerspectiveTransform(source_coords, quad_coords)
    warped_gaussian = cv2.warpPerspective(
        kernel, transform, (target_w, target_h), flags=cv2.INTER_LINEAR
    )
    np.maximum(canvas, warped_gaussian, out=canvas)
    return canvas


def order_coords_clockwise(quad_coords: np.ndarray) -> np.ndarray:
    """
    Orders a set of points clockwise around their centroid

    Args:
        quad_coords: Array of [col, row] points to be sorted

    Returns:
        The quad_coords array sorted clockwise
    """
    centre = np.mean(quad_coords, axis=0)
    angles = np.arctan2(
        quad_coords[:, 1] - centre[1], quad_coords[:, 0] - centre[0]
    )
    sorted_indices = np.argsort(-angles)
    return quad_coords[sorted_indices]


def get_affinity_edge_pairs(
    char_quad: np.ndarray, affinity_vector: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given a quadrilateral and a vector, selects the vertex pairs that form the edges which are most aligned with the vector (i.e max dot product projection). This prevents the affinity box from collapsing, as would occur if the vertex pairs form edges that are (nearly) perpendicular to the vector

    Args:
        char_quad: One of the character quadrilaterals used to construct an affinity box. 4x2 shape structured as top left -> top right -> bottom right -> bottom left, where each point is [col, row]
        affinity_vector: A 2D array representing the vector between the centroids of adjacent characters
    Returns:
        A tuple indicating which vertices of the character quadrilateral should be paired together to form the vertices of an affinity box
    """
    char_quad_edge_0_1 = char_quad[0] - char_quad[1]
    char_quad_edge_0_3 = char_quad[0] - char_quad[3]

    normalised_affinity_vector = affinity_vector / np.linalg.norm(
        affinity_vector
    )
    normalised_char_quad_edge_0_1 = char_quad_edge_0_1 / np.linalg.norm(
        char_quad_edge_0_1
    )
    normalised_char_quad_edge_0_3 = char_quad_edge_0_3 / np.linalg.norm(
        char_quad_edge_0_3
    )

    char_quad_edge_0_1_score = abs(
        np.dot(normalised_char_quad_edge_0_1, normalised_affinity_vector)
    )
    char_quad_edge_0_3_score = abs(
        np.dot(normalised_char_quad_edge_0_3, normalised_affinity_vector)
    )
    return (
        ((0, 1), (2, 3)) if char_quad_edge_0_1_score > char_quad_edge_0_3_score else (
            (0, 3), (1, 2))
    )


def get_affinity_vertex(
    char_quad: np.ndarray, vertex_pair: Tuple[int, int]
) -> np.ndarray:
    """
    Forms a triangle using two vertices and the centroid of a character quadrilateral, then computes the centroid of this triangle. The result becomes a vertex in the associated affinity box

    Args:
        char_quad: One of the character quadrilaterals used to construct an affinity box. 4x2 shape structured as top left -> top right -> bottom right -> bottom left, where each point is [col, row]
        vertex_pair: A tuple specifying which vertices from the character quadrilateral should be used to compute the affinity vertex

    Returns:
        A 2D array representing a vertex of the affinity box
    """
    affinity_vertex = np.array(
        [
            char_quad[vertex_pair[0]],
            char_quad[vertex_pair[1]],
            char_quad.mean(axis=0)
        ]
    ).mean(axis=0)
    return affinity_vertex


def get_affinity_quad(
    char_quad_1: np.ndarray, char_quad_2: np.ndarray
) -> np.ndarray:
    """
    Given two character quadrilaterals, generates their affinity box

    Args:
        char_quad_1: One of the character quadrilaterals used to construct an affinity box. 4x2 shape structured as top left -> top right -> bottom right -> bottom left, where each point is [col, row]
        char_quad_2: One of the character quadrilaterals used to construct an affinity box. 4x2 shape structured as top left -> top right -> bottom right -> bottom left, where each point is [col, row]

    Returns:
        A 4x2 array representing the affinity box between adjacent characters
    """
    affinity_vector = char_quad_1.mean(axis=0) - char_quad_2.mean(axis=0)

    quad_1_pair_1, quad_1_pair_2 = get_affinity_edge_pairs(
        char_quad_1, affinity_vector
    )
    quad_2_pair_1, quad_2_pair_2 = get_affinity_edge_pairs(
        char_quad_2, affinity_vector
    )

    quad_1_tri_1_centre = get_affinity_vertex(char_quad_1, quad_1_pair_1)
    quad_1_tri_2_centre = get_affinity_vertex(char_quad_1, quad_1_pair_2)
    quad_2_tri_1_centre = get_affinity_vertex(char_quad_2, quad_2_pair_1)
    quad_2_tri_2_centre = get_affinity_vertex(char_quad_2, quad_2_pair_2)

    affinity_quad = order_coords_clockwise(
        np.array(
            [
                quad_1_tri_1_centre,
                quad_1_tri_2_centre,
                quad_2_tri_1_centre,
                quad_2_tri_2_centre,
            ],
            dtype=np.float32
        )
    )
    return affinity_quad


def get_heatmaps(
    char_quads: np.ndarray,
    word_groups: List[np.ndarray],
    output_size: int,
    kernel_size: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an array of character quadrilaterals and an array that allocates them (by index) to a given word, create 2 heatmaps: one for the characters and another for their inter-character affinities

    Args:
        char_quads: An array of character quadrilaterals, each of which is 4x2 array structured as top left -> top right -> bottom right -> bottom left, where each point is [col, row]
        word_groups: A list of arrays, each of which contains the indices of character quadrilaterals belonging to that word
        output_size: The pixel size (height and width) output by the model
        kernel_size: The diameter of the generated gaussian
        sigma: The standard deviation of the generated gaussian

    Returns:
        The character and affinity heatmaps for a given image
    """
    kernel = get_isotropic_gaussian(kernel_size, sigma)

    char_map = np.zeros((output_size, output_size), dtype=np.float32)
    affinity_map = np.zeros((output_size, output_size), dtype=np.float32)

    for char in char_quads:
        warp_and_paste_gaussian(char_map, char, kernel)

    for word in word_groups:
        for i in range(len(word)-1):
            char_1, char_2 = char_quads[word[i:i+2]]
            char_1 = order_coords_clockwise(char_1)
            char_2 = order_coords_clockwise(char_2)
            affinity_quad = get_affinity_quad(char_1, char_2)
            warp_and_paste_gaussian(affinity_map, affinity_quad, kernel)
    return char_map, affinity_map
