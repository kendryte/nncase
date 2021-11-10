import enum
import math
import os
import re
import struct
import numpy as np
from typing import List, Tuple
from pathlib import Path
from operator import le, lt, ge, gt, eq


def cosine(gt: np.ndarray, pred: np.ndarray, *args):
    return (gt @ pred) / (np.linalg.norm(gt, 2) * np.linalg.norm(pred, 2))


def euclidean(gt: np.ndarray, pred: np.ndarray, *args):
    return np.linalg.norm(gt - pred, 2)**2


def allclose(gt: np.ndarray, pred: np.ndarray, thresh: float):
    return np.allclose(gt, pred, atol=thresh)


def segment_close(gt: np.ndarray, pred: np.ndarray):
    bucket = np.digitize(gt, [0, 64, 128, 10 ** 18])
    seg_1 = (bucket == 1)
    seg_2 = (bucket == 2)
    seg_3 = (bucket == 3)
    ret = True
    if seg_1.size:
        ret &= np.allclose(gt[seg_1], pred[seg_1], atol=0.6)
    if seg_2.size:
        ret &= np.allclose(gt[seg_2], pred[seg_2], atol=2)
    if seg_3.size:
        ret &= np.allclose(gt[seg_3], pred[seg_3], rtol=8 / 128)
    return ret


def top1(gt_res, result_res):
    case_name = gt_res.split('/')[-2]
    result_data_dict = {}
    gt_data_dict = {}
    with open(gt_res, 'r') as gt_f, open(result_res, 'r') as result_f:
        for line in result_f.readlines():
            result_data_dict[line.strip('\n').split(' ')[0]] = line.strip('\n').split(' ')[1]

        for line in gt_f.readlines():
            print(line)
            gt_data_dict[line.strip('\n').split(' ')[0]] = line.strip('\n').split(' ')[1]

    true_result = 0
    for key, value in result_data_dict.items():
        if (int(value) == int(gt_data_dict[key])):
            true_result = true_result + 1
    result_map = true_result / len(result_data_dict)
    return result_map


simarity_func = {
    'cosine': cosine,
    'euclidean': euclidean,
    'allclose': np.allclose,
    'segment': segment_close,
    'top1': top1
}


def compare(result_path: Tuple[str, str],
            ground_truth_path: Tuple[str, str],
            dtype,
            simarity_name: str = 'cosine',
            threshold: float = 0.99,
            hist: bool = True) -> bool:
    # NOTE the result_path is Tuple[ bin_path, txt_path ]
    if simarity_name == "top1":
        simarity = top1(ground_truth_path, result_path)
        simarity_info = f"\n{simarity_name} similarity = {simarity}, threshold = {threshold}\n"
    else:
        ground_truth_path_bin, ground_truth_path_txt = result_path
        result_path_bin, result_path_txt = ground_truth_path
        # if simarity_name == "top1":
        #     result = top1(ground_truth_path_txt, result_path_txt, result_path_txt.split('/')[3])
        #     simarity_info = f"\n{simarity_name} similarity = {simarity}, threshold = {threshold}\n"
        # else:
        if 'npy' in ground_truth_path_bin:  # bfloat16
            # gt, pred = bytes.fromhex(gt.strip()), bytes.fromhex(pred.strip())
            # gt, pred = struct.unpack('>H', gt)[0], struct.unpack('>H', pred)[0]
            raise NotImplemented("need support bfloat16 judge!")
        else:
            gt_arr = np.fromfile(ground_truth_path_bin, dtype).astype(np.float32)
            pred_arr = np.fromfile(result_path_bin, dtype).astype(np.float32)
            if gt_arr.size == pred_arr.size:
                simarity = simarity_func[simarity_name](gt_arr, pred_arr)
            else:
                raise ValueError("The number of elements in gt and result not match\n")
            if hist:
                y, x = np.histogram(gt_arr - pred_arr, 100)
                p = Path(result_path_bin)
                np.savetxt(str(p.parent / (p.stem + '_hist.csv')),
                           np.stack((x[:-1], y)).T, fmt='%f', delimiter=',')
            simarity_info = f"\n{simarity_name} similarity = {simarity}, threshold = {threshold}\n"
    if simarity_name in ['cosine', 'euclidean', 'segment', 'top1']:
        compare_op = lt
    else:
        compare_op = gt
    if compare_op(simarity, threshold):
        return False, simarity_info
    return True, simarity_info
