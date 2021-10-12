import argparse
import os
import sys
from pathlib import Path

import cv2
import nncase
import numpy as np
print(os.getpid())


def preproc(img, input_size, transpose=True):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    if transpose:
        padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img)
    return padded_img, r


def read_images(imgs_dir: str, test_size: list):
    imgs_dir = Path(imgs_dir)
    imgs = []
    for p in imgs_dir.iterdir():
        img = cv2.imread(str(p))
        img, _ = preproc(img, test_size, True)  # img [h,w,c] rgb,
        imgs.append(img)

    imgs = np.stack(imgs)
    return len(imgs), imgs.tobytes()


def main(onnx: str, kmodel: str, target: str, method: str, imgs_dir: str, test_size: list, legacy: bool, no_preprocess: bool):
    cpl_opt = nncase.CompileOptions()
    cpl_opt.preprocess = not no_preprocess
    # (x - mean) / scale
    if legacy:
        cpl_opt.swapRB = False  # legacy use RGB 
        cpl_opt.input_range = [0, 1]
        cpl_opt.mean = [0.485, 0.456, 0.406]
        cpl_opt.std = [0.229, 0.224, 0.225]
    else:
        cpl_opt.swapRB = True  # new model use BGR 
        cpl_opt.input_range = [0, 255]
        cpl_opt.mean = [0, 0, 0]
        cpl_opt.std = [1, 1, 1]
    cpl_opt.target = target  # cpu , k210, k510!
    cpl_opt.input_type = 'uint8'
    cpl_opt.input_layout = 'NCHW'
    cpl_opt.input_shape = [1, 3, 224, 224]
    cpl_opt.quant_type = 'uint8'  # uint8 or int8

    compiler = nncase.Compiler(cpl_opt)
    with open(onnx, 'rb') as f:
        imp_opt = nncase.ImportOptions()
        compiler.import_onnx(f.read(), imp_opt)
        # ptq
        if imgs_dir is not None:
            ptq_opt = nncase.PTQTensorOptions()
            ptq_opt.calibrate_method = method
            ptq_opt.samples_count, tensor_data = read_images(
                imgs_dir, test_size)
            ptq_opt.set_tensor_data(tensor_data)
            compiler.use_ptq(ptq_opt)
        compiler.compile()
        kmodel_bytes = compiler.gencode_tobytes()
    with open(kmodel, 'wb') as of:
        of.write(kmodel_bytes)
        of.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX Compile Demo!")
    parser.add_argument('onnx', default='model/yolox_nano_224_new.onnx', help='model path')
    parser.add_argument('kmodel', default='yolox_nano_224_new.kmodel', help='bin path')
    parser.add_argument('--target', default='cpu',
                        choices=['cpu', 'k210', 'k510'], help='compile target')
    parser.add_argument('--method', default='no_clip',
                        choices=['no_clip', 'l2', 'kld_m0', 'kld_m1', 'kld_m2', 'cdf'],
                        help='calibrate method')
    parser.add_argument('--test_size', default=[224, 224],
                        nargs='+', help='test size')
    parser.add_argument("--imgs_dir", default=None, help="images dir")
    parser.add_argument("--legacy", default=False,
                        action="store_true", help="To be compatible with older versions")
    parser.add_argument("--no_preprocess", default=False,
                        action="store_true", help="disable nncase preprocess for debug")

    args = parser.parse_args()
    main(args.onnx, args.kmodel, args.target, args.method, args.imgs_dir,
         args.test_size, args.legacy, args.no_preprocess)
