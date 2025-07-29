import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import coremltools.optimize._utils as opt_utils
import coremltools.optimize.coreml as cto_coreml
import coremltools.optimize as cto

from optimisation import preprocess_image
import file_utils
import imgproc
import copy
import os
from google.protobuf.json_format import MessageToDict
import json

_orig = opt_utils.get_min_and_max_values


# def patched_get_min_and_max_values(activation_stats, var_name):
#     if var_name not in activation_stats or "rmin" not in activation_stats[var_name]:
#         print(f"[❌] Missing rmin/rmax for tensor: {var_name}")
#     return _orig(activation_stats, var_name)


# opt_utils.get_min_and_max_values = patched_get_min_and_max_values

# loaded_images = []
# image_list, _, _ = file_utils.get_files("test_images")
# for k, image_path in enumerate(image_list):
#     image = imgproc.loadImage(image_path)
#     loaded_images.append(image)
# calibration_list = []
# for image in loaded_images:
#     img, _, __ = preprocess_image(image, 1280, 1.5, "coreml")
#     calibration_list.append(img)

# # Path to your mlpackage model folder
# model_path = "weights/CoreML_CRAFT_trimmed.mlpackage/Data/com.apple.CoreML"
# spec_path = os.path.join(model_path, "model.mlmodel")

# # Load model spec only (does NOT load weights)
# spec = ct.utils.load_spec(spec_path)
# spec_trimmed = copy.deepcopy(spec)
# print(spec.description.output)
# new_model = ct.models.MLModel(
#     spec, weights_dir="weights/CoreML_CRAFT_trimmed.mlpackage/Data/com.apple.CoreML/weights")
# print(new_model.output_description)

# activation_config = cto_coreml.OptimizationConfig(
#     global_config=cto_coreml.experimental.OpActivationLinearQuantizerConfig(
#         mode="linear_symmetric"
#     )
# )

# # Calibrate activations using sample data (replace 'input_name' with actual input name)
# calibration_data = [{"x": sample_np}
#                     for sample_np in calibration_list]

# # Quantize activations
# try:
#     mlmodel_a8 = cto_coreml.experimental.linear_quantize_activations(
#         new_model,
#         activation_config,
#         sample_data=calibration_data
#     )
# except KeyError as e:
#     print("Missing activation stat for:", e)

# mlmodel_a8.save("weights/CoreML_CRAFT_A8_Trimmed.mlpackage")

# Reload specifying weights_dir again to enable next step
spec_a8 = ct.utils.load_spec("weights/CoreML_CRAFT_A8_Trimmed.mlpackage")
weights_dir_a8 = "weights/CoreML_CRAFT_A8_Trimmed.mlpackage/Data/com.apple.CoreML/weights"
mlmodel_a8_loaded = ct.models.MLModel(
    "weights/CoreML_Craft_A8_Trimmed.mlpackage")

op_config = cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", weight_threshold=512
)
config = cto.coreml.OptimizationConfig(global_config=op_config)

compressed_8_bit_model = cto.coreml.linear_quantize_weights(
    mlmodel_a8_loaded, config=config)

# mlmodel_w8a8 = quantization_utils.quantize_weights(
#     mlmodel_a8_loaded,
#     nbits=8,
#     quantization_mode="linear_symmetric"
# )

# Save final W8A8 quantized model
compressed_8_bit_model.save("weights/CoreML_CRAFT_W8A8.mlpackage")
