import os
import torch
import argparse
from collections import OrderedDict
import coremltools as ct
from craft import CRAFT


def copyStateDict(state_dict):

    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


if not os.path.isdir("weights"):
    os.mkdir("weights")

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--device",
    default="cpu",
    help="Device used to run the model trace (no impact on future use)"
)
args = parser.parse_args()

print("Loading the CRAFT detection model")
net = CRAFT()
net.load_state_dict(
    copyStateDict(
        torch.load("weights/craft_mlt_25k.pth", map_location=args.device)
    )
)
net = net.to(args.device)
net.eval()

print("Tracing the model with test input")
example_input = torch.rand(1, 3, args.canvas_size, args.canvas_size)
example_input = example_input.to(args.device)
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

print("Exporting the model to CoreML")
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape)],
    # Set compute_units as "ct.ComputeUnit.ALL" to allow gpu and neural engine use with cpu fallback as required
    compute_units=ct.ComputeUnit.ALL,
    convert_to="mlprogram",
    # Require MacOS 15+ to ensure opset 7+ which enables full quantisation
    minimum_deployment_target=ct.target.macOS15
)

coreml_file_name = "weights/CoreML_CRAFT_FP16.mlpackage"
coreml_model.save(coreml_file_name)
model_spec = ct.utils.load_spec(coreml_file_name)
model_inputs = [inputs.name for inputs in model_spec.description.input]
model_outputs = [outputs.name for outputs in model_spec.description.output]

print(f"\nModel exported and saved to {coreml_file_name}")
print(f"Provide inputs as a dictionary with the key(s): {model_inputs}")
print(f"Outputs returned as a dictionary with the key(s): {model_outputs}")
