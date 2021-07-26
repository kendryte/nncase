import numpy as np
import cv2
import sys
from pathlib import Path
import argparse
import os
DEBUG = 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLOX k210 Demo!")
    parser.add_argument('image', default='images/dog.png', help='image path')
    parser.add_argument('bin', default='k210/yolox_detect_example/input.bin', help='bin path')
    parser.add_argument("--tsize", type=str, default=None, help="test image size")
    args = parser.parse_args()
    if args.tsize is not None:
        input_size = [args.tsize, args.tsize]
    else:
        input_size = [224, 224]
    assert(os.path.exists(args.image)), "the image path is invalid !"
    image = cv2.imread(args.image)
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    if r != 1:
        resized_img = cv2.resize(
            image, (int(image.shape[1] * r), int(image.shape[0] * r)), interpolation=cv2.INTER_LINEAR
        )
        padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img
        image = padded_img
    if DEBUG:
        cv2.imwrite('debug.jpg',image)
    image = image[:, :, ::-1]
    image = np.moveaxis(image, 2, 0)
    image.tofile(args.bin)
