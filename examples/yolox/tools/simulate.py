import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import nncase
import numpy as np
import torch

from compile import preproc
from decoder import decode_outputs, postprocess
from visualize import vis
from termcolor import colored

COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", )


class ResizeInfo(NamedTuple):
    ratio: float
    raw_img: np.ndarray


def decode(prediction: torch.Tensor, test_size=[224, 224],
           num_classes=80,
           conf_thre=0.1,
           nms_thre=0.1,
           strides=[8, 16, 32],
           ):

    prediction = decode_outputs(prediction, prediction.dtype, np.array(test_size), strides)
    print(f"{colored('prediction range :','blue')} <{prediction.min()}, {prediction.max()}>")
    outputs = postprocess(prediction, num_classes, conf_thre, nms_thre)
    return outputs


def visual(output: torch.Tensor, ratio: float, raw_img: np.ndarray, cls_conf=0.35) -> np.ndarray:
    if output is None:
        return raw_img
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(raw_img, bboxes, scores, cls, cls_conf, COCO_CLASSES)
    return vis_res


def main(kmodel: str, test_size: list, img_path: str):
    sim = nncase.Simulator()

    with open(kmodel, 'rb') as f:
        sim.load_model(f.read())

    raw_img = cv2.imread(img_path)
    ratio = min(test_size[0] / raw_img.shape[0], test_size[1] / raw_img.shape[1])

    img, _ = preproc(raw_img, test_size, transpose=True)

    img = np.expand_dims(img, 0)
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(img))
    sim.run()
    prediction = sim.get_output_tensor(0)
    outputs = decode(torch.from_numpy(prediction.to_numpy()), test_size)
    res_img = visual(outputs[0], ratio, raw_img)
    plt.imshow(res_img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX Simulate Demo!")
    parser.add_argument('kmodel', default='yolox_nano_224_new.kmodel', help='bin path')
    parser.add_argument("img_path", default=None)
    parser.add_argument('--test_size', default=[224, 224],
                        nargs='+', help='test size')
    args = parser.parse_args()
    main(args.kmodel, args.test_size, args.img_path)
