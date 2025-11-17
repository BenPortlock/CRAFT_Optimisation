## CRAFT_Optimisation

### Overview

This repo explores some of the techniques available for reducing ML inference latency. The model of choice is Clova AI's CRAFT text detector (see: [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ))

## Updates

- **27 Jul, 2025**: Initial commit with support for cpu, mps, and coreml inference on both single and batch inputs.
- **29 Jul, 2025**: Refactored export_craft_to_coreml.py and quantise_craft_coreml.py and added support for W8A8 quantisation in the latter
- **19 Aug, 2025**: Added script to create CRAFT-compatible training data

#### Requirements

##### Essential

- torch==2.5.0
- torchvision==0.20.0
- opencv-python==4.12.0.88

##### Required For MPS and CoreML Functionality

- MacOS>=15
- coremltools==8.3.0
