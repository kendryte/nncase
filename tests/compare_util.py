import enum
import math
import os
import re
import struct
import numpy as np
from typing import List, Tuple
from pathlib import Path


def cosine(gt: np.ndarray, pred: np.ndarray):
    return (gt @ pred) / (np.linalg.norm(gt, 2) * np.linalg.norm(pred, 2))


def euclidean(gt: np.ndarray, pred: np.ndarray):
    return np.linalg.norm(gt - pred, 2)**2


simarity_func = {
    'cosine': cosine,
    'euclidean': euclidean
}


def compare(result_path: Tuple[str, str],
            ground_truth_path: Tuple[str, str],
            simarity_name: str = 'cosine',
            threshold: float = 0.99,
            hist: bool = True) -> bool:
    # NOTE the result_path is Tuple[ bin_path, txt_path ]
    ground_truth_path_bin, ground_truth_path_txt = result_path
    result_path_bin, result_path_txt = ground_truth_path
    if 'npy' in ground_truth_path_bin:  # bfloat16
        # gt, pred = bytes.fromhex(gt.strip()), bytes.fromhex(pred.strip())
        # gt, pred = struct.unpack('>H', gt)[0], struct.unpack('>H', pred)[0]
        raise NotImplemented("need support bfloat16 judge!")
    else:  # float
        gt_arr = np.fromfile(ground_truth_path_bin, np.float32)
        pred_arr = np.fromfile(result_path_bin, np.float32)
        if gt_arr.size == pred_arr.size:
            simarity = simarity_func[simarity_name](gt_arr, pred_arr)
        else:
            raise ValueError("The number of elements in gt and result not match\n")
        if hist:
            y, x = np.histogram(gt_arr - pred_arr, 100)
            p = Path(result_path_bin)
            np.savetxt(p.parent / (p.stem + '_hist.csv'),
                       np.stack((x[:-1], y)).T, fmt='%f', delimiter=',')
    if simarity < threshold:
        print(f"{simarity_name} similarity is: {simarity} < {threshold}")
        return False
    return True
