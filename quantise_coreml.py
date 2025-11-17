"""
Script used to quantise a CoreML CRAFT model to W8A8 precision
"""

import argparse

import coremltools as ct
import coremltools.optimize as cto
import coremltools.optimize.coreml as cto_coreml
from loguru import logger
import numpy as np

from utils.imgproc import loadImage, preprocess_image
from utils.file_utils import get_files, get_folder_size


def main(args: argparse.Namespace) -> None:

    # The base CRAFT CoreML model produces an intermediate output named input_137 which may be used by the refiner net. In order to quantise the model with CoreML, this node needs to be removed from the model spec outputs and the model spec functions.
    base_model_path = args.model_path
    spec_path = f"{base_model_path}/Data/com.apple.CoreML/model.mlmodel"
    model_spec = ct.utils.load_spec(spec_path)

    logger.info(
        f"{'-'*8} Removing problematic node 'input_137' from the model {'-'*8}"
    )
    # Create a new output list excluding input_137 and overwrite the model spec outputs
    new_outputs = [
        out for out in model_spec.description.output if out.name != "input_137"
    ]
    del model_spec.description.output[:]
    model_spec.description.output.extend(new_outputs)

    # Locate the input_137 node in the model's main function and remove it
    program = model_spec.mlProgram
    functions = program.functions
    main_func = functions["main"]
    block = main_func.block_specializations["CoreML8"]
    block.outputs[:] = [out for out in block.outputs if out != "input_137"]

    # In order to create the trimmed model, we need to build a CoreML package by combining the trimmed spec with the untrimmed weights. The weights related to input_137 will simply be ignored.
    trimmed_model = ct.models.MLModel(
        model_spec,
        weights_dir=f"{base_model_path}/Data/com.apple.CoreML/weights"
    )
    trimmed_model_path = f"{base_model_path.split('.')[0]}_Trimmed.mlpackage"
    trimmed_model.save(trimmed_model_path)

    # Now that we have removed the problematic node, we can quantise the model to W8A8 (8-bit weights and activations).
    # To quantise the model activations, we need to pass calibration data through the model in the same format that it would normally accept.
    logger.info(f"{'-'*8} Loading calibration data {'-'*8}")
    image_list, _, _ = get_files(args.data_dir)
    loaded_images = [loadImage(image_path) for image_path in image_list]

    # Obtain the image size accepted by the model and prepare calibration images accordingly
    trimmed_spec = trimmed_model.get_spec()
    trimmed_input = trimmed_spec.description.input[0].type.multiArrayType.shape
    canvas_size = trimmed_input[2]

    calibration_data = []
    for image in loaded_images:
        img, _, __ = preprocess_image(image, canvas_size, mag_ratio=1.5)
        img = np.expand_dims(img, axis=0)
        calibration_data.append({"x": img})

    logger.info(
        f"{'-'*8} Quantising the activations to 8-bit integers {'-'*8}"
    )
    activation_config = cto_coreml.OptimizationConfig(
        global_config=cto_coreml.experimental.OpActivationLinearQuantizerConfig(
            mode="linear_symmetric"
        )
    )

    trimmed_model_a8 = cto_coreml.experimental.linear_quantize_activations(
        trimmed_model,
        activation_config,
        sample_data=calibration_data
    )

    logger.info(
        f"{'-'*8} Quantising the weights to 8-bit integers {'-'*8}"
    )
    op_config = cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        weight_threshold=512
    )

    weights_config = cto.coreml.OptimizationConfig(global_config=op_config)

    trimmed_model_w8a8 = cto.coreml.linear_quantize_weights(
        trimmed_model_a8,
        config=weights_config
    )

    logger.info(f"{'-'*8} Saving the W8A8 Model {'-'*8}")
    w8a8_model_path = f"{base_model_path.split('.')[0]}_W8A8.mlpackage"
    trimmed_model_w8a8.save(w8a8_model_path)
    logger.info(f"W8A8 model saved to {w8a8_model_path}")

    logger.info(
        f"Model size before W8A8 quantisation: "
        f"{get_folder_size(trimmed_model_path)/1e6:.2f} MB"
    )
    logger.info(
        f"Model size after W8A8 quantisation: "
        f"{get_folder_size(w8a8_model_path)/1e6:.2f} MB"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Quantise CoreML CRAFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the CoreML model being quantised. Quantised file is saved to the same directory"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory used for W8A8 calibration"
    )
    args = parser.parse_args()
    main(args)
