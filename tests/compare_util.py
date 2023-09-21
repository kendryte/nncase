import enum
import math
import os
import re
import struct
import numpy as np
from typing import List, Tuple
from pathlib import Path
from operator import le, lt, ge, gt, eq
import test_utils


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


similarity_func = {
    'cosine': cosine,
    'euclidean': euclidean,
    'allclose': np.allclose,
    'segment': segment_close
}


def compare_binfile(result_path: Tuple[str, str],
                    ground_truth_path: Tuple[str, str],
                    dtype,
                    similarity_name: str = 'cosine',
                    threshold: float = 0.99,
                    hist: bool = True) -> bool:
    # NOTE the result_path is Tuple[ bin_path, txt_path ]
    ground_truth_path_bin, ground_truth_path_txt = result_path
    result_path_bin, result_path_txt = ground_truth_path
    if 'npy' in ground_truth_path_bin:  # bfloat16
        # gt, pred = bytes.fromhex(gt.strip()), bytes.fromhex(pred.strip())
        # gt, pred = struct.unpack('>H', gt)[0], struct.unpack('>H', pred)[0]
        raise NotImplemented("need support bfloat16 judge!")
    else:
        gt_arr = np.fromfile(ground_truth_path_bin, dtype).astype(np.float32)
        pred_arr = np.fromfile(result_path_bin, dtype).astype(np.float32)
        if gt_arr.size == pred_arr.size:
            similarity = similarity_func[similarity_name](gt_arr, pred_arr)
        else:
            raise ValueError("The number of elements in gt and result not match\n")
        if hist and not test_utils.in_ci:
            y, x = np.histogram(gt_arr - pred_arr, 100)
            p = Path(result_path_bin)
            np.savetxt(str(p.parent / (p.stem + '_hist.csv')),
                       np.stack((x[:-1], y)).T, fmt='%f', delimiter=',')
        similarity_info = f"\n{similarity_name} similarity = {similarity}, threshold = {threshold}\n"
    if similarity_name in ['cosine', 'euclidean', 'segment']:
        compare_op = lt
    else:
        compare_op = gt
    if compare_op(similarity, threshold):
        return False, similarity_info
    return True, similarity_info


def compare_ndarray(expected: np.ndarray,
                    actual: np.ndarray,
                    similarity_name: str = 'cosine',
                    threshold: float = 0.99,
                    dump_hist: bool = True,
                    dump_file: str = 'hist.csv') -> bool:

    if expected.size == actual.size:
        similarity = similarity_func[similarity_name](expected.flatten(), actual.flatten())
    else:
        return False, f"The numbers of elements in gt({expected.size}) and result({actual.size}) are not match.\n"

    if dump_hist:
        y, x = np.histogram(expected - actual, 100)
        np.savetxt(dump_file, np.stack((x[:-1], y)).T, fmt='%f', delimiter=',')
    similarity_info = f"{similarity_name} similarity = {similarity}, threshold = {threshold}\n"

    if similarity_name in ['cosine', 'euclidean', 'segment']:
        compare_op = lt
    else:
        compare_op = gt

    if compare_op(similarity, threshold):
        return False, similarity_info
    return True, similarity_info
