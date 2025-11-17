"""
Implementation of the CRAFT pixel-wise loss function as well as an OHEM alternative
"""
import torch


def CRAFT_loss(
    pred_char_map: torch.Tensor,
    pred_affinity_map: torch.Tensor,
    gt_char_map: torch.Tensor,
    gt_affinity_map: torch.Tensor,
    affinity_weight: float
) -> torch.Tensor:
    """
    Given a character and affinity heatmap, along with their corresponding ground truths, computes the total pixel-wise loss. Also accepts a modifier (affinity_weight) that adjusts the importance of each heatmap 

    Args:
        pred_char_map: The predicted character heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        pred_affinity_map: The predicted affinity heatmap - A 2D tensor with dimensions (input width / 2, input height / 2) 
        gt_char_map: The ground truth character heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        gt_affinity_map: The ground truth affinity heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        affinity_weight: Modifier used to adjust the importance of the character and affinity losses 

    Returns:
        The aggregated pixel-wise loss for a given image
    """
    loss_fn = torch.nn.MSELoss(reduction="sum")
    char_loss = loss_fn(pred_char_map, gt_char_map)
    affinity_loss = loss_fn(pred_affinity_map, gt_affinity_map)
    return (char_loss + affinity_loss * affinity_weight)


def _OHEM_loss(
    loss_map: torch.Tensor,
    gt_map: torch.Tensor,
    neg_ratio: float,
    num_min_neg: int
) -> torch.Tensor:
    """
    The logic for the OHEM_loss function, which computes the OHEM loss when given a pixel-wise loss map and its corresponding ground truth

    Args:
        loss_map: The pixel-wise loss map - A 2D tensor with dimensions (input width / 2, input height / 2)
        gt_map :  The ground truth heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        neg_ratio: The maximum number of negative loss pixels (false positives) that should be considered, relative to the number of positive loss pixels
        num_min_neg (int): The minimum number of negative loss pixels considered (relative to the number of positive loss pixels) when there are too few negative pixels to reach the neg_ratio threshold

    Returns:
        The aggregated pixel-wise OHEM loss for a heatmap
    """
    gt_positive_pixels = (loss_map > 0.1).float()
    positive_loss_region = loss_map * gt_positive_pixels

    gt_negative_pixels = (gt_map <= 0.1).float()
    negative_loss_region = loss_map * gt_negative_pixels

    num_gt_positive_pixels = torch.sum(gt_positive_pixels)
    num_gt_negative_pixels = torch.sum(gt_negative_pixels)

    if num_gt_positive_pixels != 0:
        if num_gt_negative_pixels < neg_ratio * num_gt_positive_pixels:
            negative_loss = (
                torch.sum(
                    torch.topk(
                        negative_loss_region.view(-1), num_min_neg, sorted=False
                    )[0]
                )
                / num_min_neg
            )
        else:
            negative_loss = torch.sum(
                (
                    torch.topk(
                        negative_loss_region.view(-1),
                        int(neg_ratio * num_gt_positive_pixels),
                        sorted=False,
                    )[0]
                    / (num_gt_positive_pixels * neg_ratio)
                )
            )
        positive_loss = (
            torch.sum(positive_loss_region) / num_gt_positive_pixels
        )
    else:
        negative_loss = torch.sum(
            (
                torch.topk(
                    negative_loss_region.view(-1),
                    num_min_neg,
                    sorted=False
                )[0]
                / num_min_neg
            )
        )
        positive_loss = 0.0
    total_loss = positive_loss + negative_loss
    return total_loss


def OHEM_loss(
    pred_char_map: torch.Tensor,
    pred_affinity_map: torch.Tensor,
    gt_char_map: torch.Tensor,
    gt_affinity_map: torch.Tensor,
    neg_ratio: float,
    num_min_neg: int
) -> torch.Tensor:
    """
     Given a character and affinity heatmap, along with their corresponding ground truths, computes the pixel-wise OHEM loss.

    Args:
        pred_char_map: The predicted character heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        pred_affinity_map: The predicted affinity heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        gt_char_map: The ground truth character heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        gt_affinity_map: The ground truth affinity heatmap - A 2D tensor with dimensions (input width / 2, input height / 2)
        neg_ratio: The maximum number of negative loss pixels (false positives) that should be considered, relative to the number of positive loss pixels
        num_min_neg (int): The minimum number of negative loss pixels considered (relative to the number of positive loss pixels) when there are too few negative pixels to reach the neg_ratio threshold

    Returns:
        The aggregated pixel-wise OHEM loss for a given image
    """
    pixelwise_loss_fn = torch.nn.MSELoss(reduction="none")
    pixelwise_char_loss = pixelwise_loss_fn(pred_char_map, gt_char_map)
    pixelwise_affinity_loss = pixelwise_loss_fn(
        pred_affinity_map, gt_affinity_map
    )
    OHEM_char_loss = _OHEM_loss(
        loss_map=pixelwise_char_loss,
        gt_map=gt_char_map,
        neg_ratio=neg_ratio,
        num_min_neg=num_min_neg
    )
    OHEM_affinity_loss = _OHEM_loss(
        loss_map=pixelwise_affinity_loss,
        gt_map=gt_affinity_map,
        neg_ratio=neg_ratio,
        num_min_neg=num_min_neg
    )
    return OHEM_char_loss + OHEM_affinity_loss
