"""
CONTENT
"""

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader
import torch_pruning as tp

from data.craft_dataset import CraftDataset
from models.architecture.craft import CRAFT
from utils.craft_utils import copyStateDict
from old_code.training import CRAFT_loss, train, validate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Folder containing training and validation data"
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Designated folder for saving pruned models"
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
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Number of images per training and validation batch"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device used to make predictions"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the model weights used for training"
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate used for parameter updates"
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs used for model training"
    )
    args = parser.parse_args()

    train_dataset = CraftDataset("train_val_data/train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    val_dataset = CraftDataset("train_val_data/val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    print("Loading CRAFT Model")
    net = CRAFT()
    net.load_state_dict(
        copyStateDict(
            torch.load(args.model_path, map_location=args.device)
        ),
        strict=False
    )
    net.to(args.device)
    pruned_model = copy.deepcopy(net)
    pruned_model = pruned_model.to(args.device)

    ignored_layers = []
    for name, module in pruned_model.named_modules():
        if "conv_cls.8" in name:
            ignored_layers.append(module)

    example_input = torch.rand(128, 3, 32, 32).to(args.device)
    macs, parameters = tp.utils.count_ops_and_params(net, example_input)

    iterative_steps = 20
    pruner = tp.pruner.MagnitudePruner(
        model=pruned_model,
        example_inputs=example_input,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=1,
        iterative_steps=iterative_steps,
        ignored_layers=ignored_layers,
        round_to=2,
    )

    prev_val_loss = validate(
        pruned_model, val_dataloader, CRAFT_loss, args.device
    )
    macs, parameters = tp.utils.count_ops_and_params(
        pruned_model, example_input
    )
    print(
        f"Initial Model Stats:\nValidation Loss: {prev_val_loss:.5f} | " f"MACS: {macs/1e9:.2f}G | Parameters: {parameters/1e6:.2f}M"
    )

    for iteration in range(1, iterative_steps + 1):

        print(f"Pruning Iteration {iteration}")
        pruner.step()
        pruned_val_loss = validate(
            pruned_model, val_dataloader, CRAFT_loss, args.device
        )
        macs, parameters = tp.utils.count_ops_and_params(
            pruned_model, example_input
        )
        current_pruning_ratio = 1 / iterative_steps * (iteration)

        train(
            net=pruned_model,
            dataloader=train_dataloader,
            epochs=args.epochs,
            loss_fn=CRAFT_loss,
            optimiser=torch.optim.Adam(
                pruned_model.parameters(), lr=args.lr
            ),
            device=args.device,
            batch_size=args.batch_size,
            teacher=net,
            silent=False
        )
        tuned_val_loss = validate(
            pruned_model, val_dataloader, CRAFT_loss, args.device
        )
        print(
            f"Post-Pruning Validation Loss: {pruned_val_loss:.5f} | "
            f"Fine-Tuned Validation Loss {tuned_val_loss:.5f} | "
            f"MACS: {macs/1e9:.2f}G | Parameters: "
            f"{parameters/1e6:.2f}M | "
            f"Proportion Pruned: {int(current_pruning_ratio * 100)}%"
        )
        print("SAVING MODEL")
        torch.save(
            pruned_model,
            (
                f"{args.save_dir}/CRAFT_Pruned_{int(current_pruning_ratio * 100)}_Weights_And_Arch.pth"
            )
        )
