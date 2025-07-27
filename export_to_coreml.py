import torch
import argparse
import importlib
import coremltools as ct
from collections import OrderedDict


def load_model_class(model_module, class_name):

    module = importlib.import_module(model_module)
    model_class = getattr(module, class_name)
    return model_class


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_module",
    type=str,
    required=True,
    help="Name of the model module/file"
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model class"
)
parser.add_argument(
    "--model_weights",
    type=str,
    required=True,
    help="Path to the PyTorch model weights being exported"
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
    "--frame_width",
    type=int,
    required=True,
    help="Frame width expected by the model"
)
parser.add_argument(
    "--frame_height",
    type=int,
    required=True,
    help="Frame height accepted by the model"
)
parser.add_argument(
    "--device",
    default="cpu",
    help="Device used for test run (no impact on future use)"
)
args = parser.parse_args()

print("Loading the PyTorch model")
ModelClass = load_model_class(args.model_module, args.model_name)
model = ModelClass()
model.load_state_dict(
    copyStateDict(torch.load(args.model_weights, map_location=args.device))
)
model = model.to(args.device)
model.eval()

print("Tracing the model with test input")
example_input = torch.rand(1, 3, args.frame_height, args.frame_width)
example_input = example_input.to(args.device)
traced_model = torch.jit.trace(model, example_input)

# Define the input shapes accepted by the model in format (B C H W)
# Using ct.RangeDim allows the batch size to vary within the range
acceptable_shape = ct.Shape(
    shape=(ct.RangeDim(
        lower_bound=args.batch_min,
        upper_bound=args.batch_max),
        3,
        args.frame_height,
        args.frame_width))

print("Exporting the model to CoreML")
cmlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=acceptable_shape)],
    # Set compute_units as "ct.ComputeUnit.ALL" to allow gpu and neural engine use with cpu fallback as required
    compute_units=ct.ComputeUnit.ALL,
)
cmlmodel.save(f"CoreML_{args.model_name}.mlpackage")
model_spec = ct.utils.load_spec(f"CoreML_{args.model_name}.mlpackage")
model_inputs = [inputs.name for inputs in model_spec.description.input]
model_outputs = [outputs.name for outputs in model_spec.description.output]

print(f"\nModel exported and saved to CoreML_{args.model_name}.mlpackage")
print(
    f"Model inputs are provided as a dictionary with the following keys: {model_inputs}"
)
print(
    f"Model outputs are stored as a dictionary with the following keys: {model_outputs}\n"
)
