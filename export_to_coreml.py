"""
Script used to export CRAFT from PyTorch to CoreML
"""

import argparse
import os

from loguru import logger
import torch
import coremltools as ct

from models.architecture.craft import CRAFT
from utils.craft_utils import copyStateDict


def main(args: argparse.Namespace) -> None:

    logger.info(f"{'-'*8} Loading the CRAFT detection model {'-'*8}")
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(args.model_path)))
    net.eval()

    logger.info(f"{'-'*8} Tracing the model with test input {'-'*8}")
    example_input = torch.rand(1, 3, args.canvas_size, args.canvas_size)
    traced_model = torch.jit.trace(net, example_input)

    # Define the input shapes accepted by the model in format (B C H W)
    # Using ct.RangeDim allows the model to accept multiple batch sizes
    input_shape = ct.Shape(
        shape=(
            ct.RangeDim(
                lower_bound=args.batch_min,
                upper_bound=args.batch_max),
            3,
            args.canvas_size,
            args.canvas_size
        )
    )

    logger.info(f"{'-'*8} Exporting the model to CoreML {'-'*8}")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=input_shape)],
        # Set compute_units as "ct.ComputeUnit.ALL" to allow gpu and neural engine use with cpu fallback as required
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        # Require MacOS 15+ to ensure opset 7+ for full quantisation
        minimum_deployment_target=ct.target.macOS15
    )

    coreml_file_name = f"{args.model_path.split('.')[0]}_CoreML.mlpackage"
    coreml_model.save(coreml_file_name)
    logger.info(
        f"{'-'*8} Model exported and saved to {coreml_file_name} {'-'*8}"
    )

    model_spec = ct.utils.load_spec(coreml_file_name)
    model_inputs = [inputs.name for inputs in model_spec.description.input]
    model_outputs = [
        outputs.name for outputs in model_spec.description.output
    ]
    logger.info(
        f"Provide inputs as a dictionary with the keys: {model_inputs}"
    )
    logger.info(
        f"Outputs returned as a dictionary with the keys: {model_outputs}"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Export CRAFT to CoreML")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the PyTorch model being exported. Exported file is saved to the same directory"
    )
    parser.add_argument(
        "--batch_min",
        type=int,
        required=True,
        help="Minimum batch size accepted by the model"
    )
    parser.add_argument(
        "--batch_max",
        type=int,
        required=True,
        help="Maximum batch size accepted by the model"
    )
    parser.add_argument(
        "--canvas_size",
        type=int,
        required=True,
        help="Frame width and height expected by the model (must match CRAFT's canvas size argument)"
    )
    args = parser.parse_args()
    main(args)