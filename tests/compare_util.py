import enum
import math
import os
import re
import struct
import numpy as np

use_cosine_to_double_check = True


def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


def length(v):
    return math.sqrt(dot(v, v))


def cosine_similarity(v1, v2):
    return dot(v1, v2) / (length(v1) * length(v2))


DiffType = enum.Enum('DIFFTYPE', 'ABS REL')


class DiffState(enum.IntEnum):
    BAD = 0
    GOOD = 1


class VerboseType(enum.IntEnum):
    SILENT = 0
    PRINT_RESULT = 1
    PRINT_BAD = 2
    PRINT_EVERY = 3


class SegmentTolerance:

    def __init__(
            self,
            seg_min,
            seg_max,
            diff_thresh,
            diff_type: DiffType,
            verbose_type: VerboseType):

        self.seg_min = seg_min
        self.seg_max = seg_max
        self.diff_thresh = diff_thresh
        self.diff_type = diff_type
        self.diff_state = DiffState.GOOD
        self.verbose_type = verbose_type

        self.accumulated_diff = 0.0
        self.n_diff = 0

    def add_diff(self, diff, print_prefix=""):
        self.accumulated_diff += diff
        self.n_diff += 1

        current_state = DiffState.GOOD if diff <= self.diff_thresh else DiffState.BAD
        if current_state == DiffState.BAD:
            self.diff_state = current_state

        if self.verbose_type == VerboseType.PRINT_EVERY or \
                self.verbose_type == VerboseType.PRINT_BAD and current_state == DiffState.BAD:
            print(print_prefix + "{}".format(diff))

    def __str__(self):
        return "[{}, {}) thr={} type={} -> {} / {} = {}, {}".format(
            self.seg_min,
            self.seg_max,
            self.diff_thresh,
            self.diff_type,
            self.accumulated_diff,
            self.n_diff,
            self.accumulated_diff / self.n_diff if self.n_diff != 0 else 0,
            self.diff_state)


class Judge:

    def __init__(self, tolerances: [SegmentTolerance]):
        tolerances.sort(key=lambda x: x.seg_min)
        self.tolerances = tolerances
        self.n_outlier = 0
        self.cosine_similarity = 0

    def judge(self, gt, pred, print_prefix=""):
        for tol in self.tolerances:
            if tol.seg_min <= abs(gt) < tol.seg_max:
                diff = abs(gt - pred)
                if tol.diff_type == DiffType.REL:
                    diff /= gt

                tol.add_diff(diff, print_prefix)
                break
        else:
            # raise ValueError
            self.n_outlier += 1

    def is_good(self):
        if self.cosine_similarity > 0.98:
            return True

        for tol in self.tolerances:
            if tol.diff_state == DiffState.BAD:
                return False
        return True

    def __str__(self):
        s = '  ' + '\n  '.join([str(tol) for tol in self.tolerances]) + ' '
        if self.n_outlier:
            s += '\n  n_outlier: %d' % self.n_outlier + ' '
        return s


class Index(object):

    def __init__(self, shape: [int]):
        self.product_shape = self._get_product_shape(shape)

    def _get_product_shape(self, shape: [int]):
        cumprod = np.cumprod(shape[-1:0:-1]).tolist()
        cumprod = cumprod[::-1]
        cumprod.append(1)
        return cumprod

    def flatten_index_shape_index(self, flatten_index: int):
        shape_index = []
        for s in self.product_shape:
            index = flatten_index // s
            shape_index.append(index)
            flatten_index %= s

        return shape_index


def compare(
        ground_truth_path,
        result_path,
        verbose=VerboseType.PRINT_EVERY,
        judge=None,
        cfg=None):

    first_threshold = 0.6 if hasattr(
        cfg, 'op') and cfg.op == 'tf.reduce_prod' else 0.5
    first_threshold = 1.0 if hasattr(
        cfg, 'op') and cfg.op == 'random_connections' else first_threshold

    if judge is None:
        judge = Judge([
            SegmentTolerance(0, 64, first_threshold, DiffType.ABS, verbose),
            SegmentTolerance(64, 128, 2, DiffType.ABS, verbose),
            SegmentTolerance(128, 10 ** 18, 8 / 128, DiffType.REL, verbose),
            SegmentTolerance(10 ** 18, float('inf'), 44 / 128, DiffType.REL, verbose)])

    gt_num_lines = sum(1 for line in open(ground_truth_path)) - 1
    res_num_lines = sum(1 for line in open(result_path)) - 1

    with open(ground_truth_path, 'r') as fgt, open(result_path, 'r') as fpred:
        shape = fgt.readline()
        fpred.readline()
        shape = re.findall(r'(\d+)', shape)
        shape = [int(dim) for dim in shape]

        index = Index(shape)

        if gt_num_lines == res_num_lines:
            for i, (gt, pred) in enumerate(zip(fgt, fpred)):
                if i % 1000 == 0:
                    #print("Compare %d..." % i, end='\r')
                    pass
                try:
                    gt, pred = float(gt), float(pred)
                except ValueError:
                    gt, pred = bytes.fromhex(
                        gt.strip()), bytes.fromhex(pred.strip())
                    gt, pred = struct.unpack(
                        '>H', gt)[0], struct.unpack('>H', pred)[0]

                shape_index = index.flatten_index_shape_index(i)

                print_prefix = '[' + ','.join(map(str, shape_index)) + \
                    '] {} -> *{} != {} by '.format(i, gt, pred)
                judge.judge(gt, pred, print_prefix)

        elif gt_num_lines == 9 * res_num_lines:
            gt_elements = fgt.read().split('\n')[:-1]
            res_elements = fpred.read().split('\n')[:-1]

            for i, pred in enumerate(res_elements):
                if i % 1000 == 0:
                    #print("Compare %d..." % i, end='\r')
                    pass
                nearest_answer = 0
                nearest_diff = 10000
                for gt_index in range(9):
                    gt, pred = float(
                        gt_elements[gt_index * res_num_lines + i]), float(pred)

                    shape_index = index.flatten_index_shape_index(i)

                    print_prefix = '[' + ','.join(map(str, shape_index)) + \
                        '] {} -> *{} != {} by '.format(i, gt, pred)

                    if abs(gt - pred) < nearest_diff:
                        nearest_answer = gt
                        nearest_diff = abs(gt - pred)

                gt = nearest_answer
                judge.judge(gt, pred, print_prefix)

        else:
            print('# of elements in gt and result not match\n')
            raise ValueError

    #print("Pass!!" if judge.is_good() else "Fail..")
    print(judge)
    if judge.is_good() == False:
        with open(ground_truth_path, 'r') as fgt, open(result_path, 'r') as fpred:
            gt = fgt.readlines()[1:]
            pred = fpred.readlines()[1:]
            judge.cosine_similarity = cosine_similarity(
                [float(i) for i in gt], [float(i) for i in pred])
            print("cosine_similarity is: {}".format(judge.cosine_similarity))
    return judge


def compare_with_ground_truth(result_path, ground_truth_path, state, verbose):
    if not isinstance(ground_truth_path, list):
        if os.path.exists(result_path) and state == 0:
            ok = compare(ground_truth_path, result_path, verbose)
        else:

            class FailObject:

                def __init__(self, result_path, state):
                    self.result_path = result_path
                    self.state = state

                def is_good(self):
                    return False

                def __str__(self):
                    return '%s, return state: %d' % (self.result_path, self.state)

            ok = FailObject(result_path, state)
    else:
        for i in range(len(result_path)):
            if os.path.exists(result_path[0]) and state == 0:

                ok = compare(ground_truth_path[i], result_path[i], verbose)

            else:

                class FailObject:

                    def __init__(self, result_path, state):
                        self.result_path = result_path
                        self.state = state

                    def is_good(self):
                        return False

                    def __str__(self):
                        return '%s, return state: %d' % (self.result_path, self.state)

                ok = FailObject(result_path[i], state)

    return ok


def compare_results(case_dir, out_len, targets, enable_ptq, is_evaluation):
    for i in range(out_len):
        gt_file = os.path.join(case_dir, 'cpu_result{0}.txt'.format(i))
        for target in targets:
            nncase_file = os.path.join(case_dir, target)
            nncase_file = os.path.join(
                nncase_file, 'eval' if is_evaluation else 'infer')
            nncase_file = os.path.join(
                nncase_file, 'ptq' if enable_ptq else 'no_ptq', 'nncase_result{0}.txt'.format(i))

            judge = compare_with_ground_truth(
                nncase_file, gt_file, state=0, verbose=VerboseType.PRINT_RESULT)
            if judge.is_good():
                print("Pass {0}.{1}!!\n".format(target, i))
            else:
                print("Fail {0}.{1}..\n".format(target, i))
                return False
    return True
