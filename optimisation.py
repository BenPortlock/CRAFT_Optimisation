import os
import sys
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils
from craft import CRAFT
from collections import OrderedDict
import coremltools as ct


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


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def make_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def preprocess_image(image, canvas_size, mag_ratio, device):

    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image,
        canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    h, w, _ = x.shape
    pad_w = canvas_size - w
    pad_h = canvas_size - h
    pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    x = np.pad(x, pad_width, mode='constant')

    if device == "coreml":
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
    else:
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
    return x, ratio_w, ratio_h


def postprocess_result(y, feature, refine_net, text_threshold, link_threshold, low_text, poly, ratio_w, ratio_h, device):

    if device == "coreml":
        score_text = y[0, :, :, 0]
        score_link = y[0, :, :, 1]
    else:
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    boxes, polys = craft_utils.getDetBoxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly
    )

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return polys, ret_score_text


def batch_inference(net, batch, text_threshold, link_threshold, low_text, device, poly, canvas_size, mag_ratio, refine_net=None):

    batch_size = len(batch)

    preprocessed_batch = []
    ratio_w_batch = []
    ratio_h_batch = []
    for image in batch:
        preprocessed_image, ratio_w, ratio_h = preprocess_image(
            image,
            canvas_size,
            mag_ratio,
            device
        )
        preprocessed_batch.append(preprocessed_image)
        ratio_w_batch.append(ratio_w)
        ratio_h_batch.append(ratio_h)
    if device == "coreml":
        inference_batch = np.concatenate(preprocessed_batch, axis=0)
    else:
        inference_batch = torch.cat(preprocessed_batch, dim=0).to(device)

    with torch.no_grad():
        inf_start = time.time()
        if device == "coreml":
            output = net.predict({"x": inference_batch})
            batch_ys = output["var_506"]
            # batch_features = output["input_137"]
            batch_features = None
        else:
            batch_ys, batch_features = net(inference_batch)
        inf_end = time.time()
        print(
            f"Batch Size: {batch_size} | Inference Time: {inf_end-inf_start:.4f}s")

    polys_batch = []
    score_text_batch = []
    for result_no in range(batch_size):
        y = batch_ys[result_no:result_no+1]
        # feature = batch_features[result_no:result_no+1]
        feature = None
        ratio_w = ratio_w_batch[result_no]
        ratio_h = ratio_h_batch[result_no]
        polys, score_text = postprocess_result(
            y,
            feature,
            refine_net,
            text_threshold,
            link_threshold,
            low_text,
            poly,
            ratio_w,
            ratio_h,
            device
        )
        polys_batch.append(polys)
        score_text_batch.append(score_text)

    return polys_batch, score_text_batch


def save_output(image_path, score_text, image, polys):

    filename, _ = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)
    file_utils.saveResult(
        image_path, image[:, :, ::-1], polys, dirname=result_folder
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument(
        '--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7,
                        type=float, help='Text confidence threshold')
    parser.add_argument('--low_text', default=0.4,
                        type=float, help='Text low-bound score')
    parser.add_argument('--link_threshold', default=0.4,
                        type=float, help='Link confidence threshold')
    parser.add_argument('--cuda', default=False,
                        action="store_true", help='Use cuda for inference')
    parser.add_argument('--mps', default=False,
                        action="store_true", help='Use mps for inference')
    parser.add_argument('--coreml', default=False,
                        action="store_true", help='Use CoreML model for inference')
    parser.add_argument('--canvas_size', default=1280,
                        type=int, help='Image size for inference')
    parser.add_argument('--mag_ratio', default=1.5,
                        type=float, help='Image magnification ratio')
    parser.add_argument('--poly', default=False,
                        action='store_true', help='Enable polygon type')
    parser.add_argument('--show_time', default=False,
                        action='store_true', help='Show processing time')
    parser.add_argument('--test_folder', default='/data/',
                        type=str, help='Folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true',
                        help='Enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth',
                        type=str, help='Pretrained refiner model')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Number of images per inference batch')
    args = parser.parse_args()

    image_list, _, _ = file_utils.get_files(args.test_folder)
    num_images = len(image_list)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    device = "coreml" if args.coreml else "cuda" if args.cuda else "mps" if args.mps else "cpu"

    if device == "coreml":
        print("\nLoading CoreML Model")
        net = ct.models.MLModel("weights/CoreML_CRAFT_W8A8.mlpackage")
    else:
        print(f"Loading {device.capitalize()} Model")
        net = CRAFT()
        net.load_state_dict(copyStateDict(torch.load(
            args.trained_model, map_location=device)))
        net.to(device)
        if device == "cuda":
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        net.eval()

    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if device == "cuda":
            refine_net.load_state_dict(
                copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.to(device)
            refine_net = torch.nn.DataParallel(refine_net)
        elif device == "mps":
            refine_net.load_state_dict(copyStateDict(
                torch.load(args.refiner_model, map_location=device)))
            refine_net = refine_net.to(device)
        else:
            refine_net.load_state_dict(copyStateDict(
                torch.load(args.refiner_model, map_location=device)))

        refine_net.eval()
        args.poly = True

    print("\nLoading Test Images")
    loaded_images = []
    for k, image_path in enumerate(image_list):
        image = imgproc.loadImage(image_path)
        loaded_images.append(image)

    print("Preparing Image Batches")
    loaded_batches = make_batches(loaded_images, args.batch_size)
    image_path_batches = make_batches(image_list, args.batch_size)

    print("Warming Up The Device")
    warmup_array = np.random.rand(
        args.batch_size,
        3,
        args.canvas_size,
        args.canvas_size)
    if device == "coreml":
        output = net.predict({"x": warmup_array})
    else:
        warmup_array = torch.from_numpy(warmup_array).to(device)
        net(warmup_array)

    print("Beginning Model Testing")
    start_time = time.time()
    img_count = 0
    for input_batch, path_batch in zip(loaded_batches, image_path_batches):
        polys_batch, score_text_batch = batch_inference(
            net,
            input_batch,
            args.text_threshold,
            args.link_threshold,
            args.low_text,
            device,
            args.poly,
            args.canvas_size,
            args.mag_ratio,
            refine_net
        )
        for image, polys, score_text, image_path in zip(input_batch, polys_batch, score_text_batch, path_batch):
            save_output(image_path, score_text, image, polys)
            img_count += 1
            print(
                f"Finished Inference For {(img_count)}/{num_images} Images", end="\r"
            )

    elapsed_time = time.time() - start_time
    print(f"\nInference Finished: {(num_images/elapsed_time):.2f}FPS")
else:
    print("optimisation.py is being imported")
