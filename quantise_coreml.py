import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import coremltools.optimize.coreml as cto_coreml
from optimisation import preprocess_image
import file_utils
import imgproc

import coremltools as ct
print(ct.utils._macos_version())

loaded_images = []
image_list, _, _ = file_utils.get_files("test_images")
for k, image_path in enumerate(image_list):
    image = imgproc.loadImage(image_path)
    loaded_images.append(image)
calibration_list = []
for image in loaded_images:
    img, _, __ = preprocess_image(image, 1280, 1.5, "coreml")
    calibration_list.append(img)

# Path to your mlpackage model folder
model_path = "weights/CoreML_CRAFT.mlpackage"

# Load model spec only (does NOT load weights)
spec = ct.utils.load_spec(model_path)
print("Model spec type:", spec.WhichOneof("Type"))

# Specify the directory where weights are stored (inside mlpackage/data/weights)
weights_dir = f"{model_path}/data/com.apple.CoreML/Weights"

# Create MLModel with both spec and weights_dir
net = ct.models.MLModel(spec, weights_dir=weights_dir)

activation_config = cto_coreml.OptimizationConfig(
    global_config=cto_coreml.experimental.OpActivationLinearQuantizerConfig(
        mode="linear_symmetric"
    )
)

# Calibrate activations using sample data (replace 'input_name' with actual input name)
calibration_data = [{"x": sample_np}
                    for sample_np in calibration_list]

# Quantize activations
mlmodel_a8 = cto_coreml.experimental.linear_quantize_activations(
    net,
    activation_config,
    sample_data=calibration_data
)

mlmodel_a8.save("weights/CRAFT_A8.mlpackage")

# Reload specifying weights_dir again to enable next step
spec_a8 = ct.utils.load_spec("weights/CRAFT_A8.mlpackage")
weights_dir_a8 = "weights/CRAFT_A8.mlpackage/Data/com.apple.CoreML/Weights"
mlmodel_a8_loaded = ct.models.MLModel(spec_a8, weights_dir=weights_dir_a8)

mlmodel_w8a8 = quantization_utils.quantize_weights(
    mlmodel_a8_loaded,
    nbits=8,
    quantization_mode="linear_symmetric"
)

# Save final W8A8 quantized model
mlmodel_w8a8.save("weights/CoreML_CRAFT_INT8_ALL.mlpackage")
