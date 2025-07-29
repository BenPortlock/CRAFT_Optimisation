import os
import imgproc
import file_utils
from optimisation import preprocess_image
import coremltools as ct
import coremltools.optimize as cto
import coremltools.optimize.coreml as cto_coreml

# The base CRAFT CoreML model produces an intermediate output named input_137 which may be used by the refiner net. In order to quantise the model with CoreML, this node needs to be removed from the model spec outputs and the model spec functions.
base_model_path = "weights/CoreML_CRAFT_FP16.mlpackage"
spec_path = f"{base_model_path}/Data/com.apple.CoreML/model.mlmodel"
model_spec = ct.utils.load_spec(spec_path)

# Create a new output list excluding input_137 and overwrite the model spec outputs
new_outputs = [
    output for output in model_spec.description.output if output.name != "input_137"
]
del model_spec.description.output[:]
model_spec.description.output.extend(new_outputs)

# Locate the input_137 node in the model's main function and remove it
program = model_spec.mlProgram
functions = program.functions
main_func = functions["main"]
block = main_func.block_specializations["CoreML8"]
block.outputs[:] = [out for out in block.outputs if out != "input_137"]

# In order to create the trimmed model, we need to build a CoreML model by combining the trimmed spec with the base weights. The weights related to input_137 will simply be ignored.
trimmed_model = ct.models.MLModel(
    model_spec,
    weights_dir=f"{base_model_path}/Data/com.apple.CoreML/weights")
trimmed_model_path = "weights/CoreML_CRAFT_Trimmed.mlpackage"
trimmed_model.save(trimmed_model_path)

# Now that we have removed the problematic node, we can quantise the model to W8A8 (8-bit weights and activations).
# To quantise the model activations, we need to pass calibration data through the model in the same format that it would normally accept.
loaded_images = []
image_list, _, _ = file_utils.get_files("calibration_images")
for k, image_path in enumerate(image_list):
    image = imgproc.loadImage(image_path)
    loaded_images.append(image)

# Obtain the image size accepted by the model and prepare calibration images accordingly
trimmed_spec = trimmed_model.get_spec()
trimmed_input = trimmed_spec.description.input[0].type.multiArrayType.shape
canvas_size = trimmed_input[2]
calibration_data = []
for image in loaded_images:
    img, _, __ = preprocess_image(image, canvas_size, 1.5, "coreml")
    calibration_data.append({"x": img})

# Quantise the activations to 8-bit integers
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

# Quantise the weights to 8-bit integers
op_config = cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    weight_threshold=512
)

weights_config = cto.coreml.OptimizationConfig(global_config=op_config)

trimmed_model_w8a8 = cto.coreml.linear_quantize_weights(
    trimmed_model_a8,
    config=weights_config
)

# Save final W8A8 quantized model
w8a8_model_path = "weights/CoreML_CRAFT_W8A8.mlpackage"
trimmed_model_w8a8.save(w8a8_model_path)
print(f"W8A8 model saved to {w8a8_model_path}")

# Compare model size after quantisation
print(
    f"Model size before W8A8 quantisation: "
    f"{file_utils.get_folder_size(trimmed_model_path)/1e6:.2f} MB"
)
print(
    f"Model size after W8A8 quantisation: "
    f"{file_utils.get_folder_size(w8a8_model_path)/1e6:.2f} MB"
)
