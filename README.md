## CRAFT_Optimisation

### Overview

This repo is intended to demonstrate some of the techniques available for optimising ML inference speed. The model of choice is Clova AI's CRAFT text detector (see: [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ))

## Updates

**27 Jul, 2025**: Initial commit with support for cpu, mps, and coreml inference on both single and batch inputs.

#### Requirements

##### Essential

- torch==2.5.0
- torchvision==0.20.0
- opencv-python==4.12.0.88

##### Required For Mac Functionality

- MacOS>=15
- coremltools==8.3.0
