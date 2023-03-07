import copy
import multiprocessing
import os
import re
import shutil
import struct
import uuid
from abc import ABCMeta, abstractmethod
from array import array
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import nncase
import numpy as np
import yaml
from PIL import Image

from compare_util import compare
from dataset_utils import *
from models.preprocess.preprocess import preprocess
import socket
import json

class Edict:
    def __init__(self, d: Dict[str, int]) -> None:
        assert (isinstance(d, dict)), "the Edict only accepct Dict for init"
        for name, value in d.items():
            if isinstance(value, (list, tuple)):
                setattr(self, name,
                        [Edict(x) if isinstance(x, dict) else x for x in value])
            else:
                if 'kwargs' in name:
                    setattr(self, name, value if value else dict())
                else:
                    if isinstance(value, dict):
                        setattr(self, name, Edict(value))
                    else:
                        setattr(self, name, value)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __repr__(self, indent=0) -> str:
        s: str = ''
        for k, v in self.__dict__.items():
            s += indent * ' ' + k + ' : '
            if isinstance(v, Edict):
                s += '\n' + v.__repr__(len(s) - s.rfind('\n'))
            else:
                s += v.__repr__().replace('\n', ' ')
            s += '\n'
        return s.rstrip('\n')

    @property
    def dict(self):
        return self.__dict__

    def update(self, d: Dict):
        for name, new_value in d.items():
            if name in self.keys():
                if isinstance(new_value, dict) and name == 'case':
                    old_value = getattr(self, name)
                    if old_value is None:
                        setattr(self, name, Edict(new_value))
                    if isinstance(old_value, (Edict, dict)):
                        old_value.update(new_value)
                elif isinstance(new_value, dict):
                    old_value = getattr(self, name)
                    if old_value is None:
                        setattr(self, name, Edict(new_value))
                    elif isinstance(old_value, (Edict, dict)):
                        old_value.update(new_value)
                elif isinstance(new_value, (list, tuple)) and name == 'specifics':
                    setattr(self, name, [])
                    assert (hasattr(self, 'common')
                            ), "The specifics new_value need common dict to overload !"
                    common = getattr(self, 'common')
                    for specific in new_value:
                        import_common = copy.deepcopy(common)
                        import_common.update(specific)
                        getattr(self, name).append(import_common)
                # elif isinstance(new_value)
                else:
                    setattr(self, name, new_value)
            else:
                setattr(self, name, new_value)


def generate_random(shape: List[int], dtype: np.dtype,
                    number: int, batch_size: int,
                    case_dir: str,
                    abs: bool = False) -> np.ndarray:
    if dtype == np.uint8:
        data = np.random.randint(0, 256, shape)
    elif dtype == np.int8:
        data = np.random.randint(-128, 128, shape)
    elif dtype == np.int32:
        data = np.random.randint(1, 3, size=shape, dtype='int32')
    elif dtype == np.int64:
        data = np.random.randint(-128, 128, size=shape, dtype='int64')
    elif dtype == np.bool:
        data = np.random.rand(*shape) > 0.5
    else:
        data = np.random.uniform(0.01, 1, shape)
    data = data.astype(dtype=dtype)
    if abs:
        return np.abs(data)
    return data


def _cast_bfloat16_then_float32(values: np.array):
    shape = values.shape
    values = values.reshape([-1])
    for i, value in enumerate(values):
        value = float(value)
        value = 1
        packed = struct.pack('!f', value)
        integers = [c for c in packed][:2] + [0, 0]
        value = struct.unpack('!f', bytes(integers))[0]
        values[i] = value


def deq_output(kmodel_info, data):
    with open(kmodel_info, 'r') as f:
        a = f.readlines()[2:4]
        scale = float(a[0].split(' ')[-1][:-1])
        zero_point = int(a[1].split(' ')[-1][:-1])
        return np.float32((data.astype(np.int) - zero_point) * scale)


def generate_image_dataset(shape: List[int], dtype: np.dtype,
                           batch_index: int, batch_size: int,
                           case_dir: str,
                           dir_path: str) -> np.ndarray:
    """ read image from folder, return the rgb image with padding, dtype = float32, range = [0,255]. same as k210 carmera.
    """
    assert (os.path.isdir(dir_path) or os.path.exists(dir_path))

    def preproc(img, input_size, transpose=True):
        # todo maybe need move this to postprocess
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
        return padded_img

    img_paths = []
    if os.path.isdir(dir_path):
        img_paths.extend([os.path.join(dir_path, p) for p in os.listdir(dir_path)])
    else:
        img_paths.append(dir_path)
    imgs = []
    transpose_flag = False
    if shape[1] in [1, 3]:
        transpose_flag = True
        shape = [shape[0], shape[2], shape[3], shape[1]]
    for p in img_paths[batch_index * batch_size:
    (batch_index + 1) * batch_size]:
        img = cv2.imread(p)
        img = preproc(img, shape[1:3], transpose_flag)  # img [h,w,c] rgb,
        imgs.append(img.astype(np.float32) / 255.)
    return np.stack(imgs)


def generate_imagenet_dataset(shape: List[int], dtype: np.dtype,
                              batch_index: int, batch_size: int,
                              case_dir: str,
                              dir_path: str) -> np.ndarray:
    """
    shape: [N,H,W,C]
    """
    dir_path = os.path.join(os.getenv('DATASET_DIR') if os.getenv('DATASET_DIR') else '', dir_path)
    assert (os.path.isdir(dir_path) or os.path.exists(dir_path))

    img_paths = []
    if os.path.isdir(dir_path):
        img_paths.extend([os.path.join(dir_path, p) for p in os.listdir(dir_path)])
    else:
        img_paths.append(dir_path)
    imgs = []
    for p in img_paths[0:batch_size]:
        img_data = Image.open(p).convert('RGB')
        img_data = np.asarray(img_data, dtype=dtype)
        model = case_dir.split('/')[-2]
        data = preprocess(model, img_data, shape)
        data = np.expand_dims(data, axis=0)
        imgs.append((data, p))
    return imgs


DataFactory = {
    'generate_random': generate_random,
    'generate_image_dataset': generate_image_dataset,
    'generate_imagenet_dataset': generate_imagenet_dataset
}


class TestRunner(metaclass=ABCMeta):
    def __init__(self, case_name, targets=None, overwrite_configs: Union[Dict, str] = None) -> None:
        config_root = os.path.dirname(__file__)
        with open(os.path.join(config_root, 'config.yml'), encoding='utf8') as f:
            cfg: dict = yaml.safe_load(f)
            config = Edict(cfg)
        config = self.update_config(config, overwrite_configs)
        self.cfg = self.validte_config(config)
        self.in_ci = os.getenv('CI', False)

        case_name = case_name.replace('[', '_').replace(']', '_')
        if self.in_ci:
            self.case_dir = os.path.join(self.cfg.setup.root, case_name + '-' + str(uuid.uuid4()))
        else:
            self.case_dir = os.path.join(self.cfg.setup.root, case_name)
        self.clear(self.case_dir)

        self.kpu_target = os.getenv('KPU_TARGET')
        self.port = os.getenv('PORT')
        self.nncase_test_ci = os.getenv('NNCASE_TEST_CI')

        if self.in_ci and self.cfg.case.generate_inputs.name == 'generate_random' and self.kpu_target is not None and self.port is not None and self.nncase_test_ci is not None and (targets is None or self.kpu_target in targets):
            new_targets = []
            new_targets.append(self.kpu_target)
        else:
            new_targets = targets
        self.validate_targets(new_targets)

        self.inputs: List[Dict] = []
        self.calibs: List[Dict] = []
        self.dump_range_data: List[Dict] = []
        self.outputs: List[Dict] = []
        self.input_paths: List[Tuple[str, str]] = []
        self.calib_paths: List[Tuple[str, str]] = []
        self.dump_range_data_paths: List[Tuple[str, str]] = []
        self.output_paths: List[Tuple[str, str]] = []
        self.model_type: str = ""
        self.model_path: str = ""
        self.pre_process: List[Dict] = []

        self.num_pattern = re.compile("(\d+)")

    def transform_input(self, values: np.array, type: str, stage: str):
        values = copy.deepcopy(values)
        if (len(values.shape) == 4 and (
                self.pre_process[0]['preprocess'] or self.cfg.case.generate_inputs.name != "generate_random")):
            if stage == "CPU":
                # onnx \ caffe
                if ((self.model_type == "onnx" or self.model_type == "caffe") and self.pre_process[5][
                    'model_layout'] == "NCHW"):
                    values = np.transpose(values, [0, 3, 1, 2])

            if type == 'float32':
                return values.astype(np.float32)
            elif type == 'uint8':
                if values.dtype == np.float32:
                    values = ((values) * 255).astype(np.uint8)
                return values
            elif type == 'int8':
                if values.dtype == np.float32:
                    values = (values * 255 - 128).astype(np.int8)
                return values
            else:
                raise TypeError(" Not support type for quant input")
        else:
            return values

    def get_process_config(self, config):
        # preprocess flag
        preprocess_flag = {}
        preprocess_flag['preprocess'] = config['preprocess']

        # dequant
        process_deq = {}
        process_deq['range'] = config['input_range']
        process_deq['input_type'] = config['input_type']

        # swapRB
        process_format = {}
        process_format['swapRB'] = config['swapRB']

        # letter box
        process_letterbox = {}
        process_letterbox['input_range'] = config['input_range']
        process_letterbox['model_shape'] = self.inputs[0]['model_shape'] if self.inputs else config['input_shape']
        process_letterbox['input_type'] = config['input_type']
        process_letterbox['input_shape'] = config['input_shape']
        process_letterbox['letterbox_value'] = config['letterbox_value']
        # norm
        process_norm = {}
        data = {}
        data = {
            'mean': config['mean'],
            'std': config['std']
        }
        process_norm['norm'] = data

        # get layout
        process_layout = {}
        process_layout['input_layout'] = config['input_layout']
        process_layout['model_layout'] = config['model_layout']

        self.pre_process.append(preprocess_flag)
        self.pre_process.append(process_deq)
        self.pre_process.append(process_format)
        self.pre_process.append(process_letterbox)
        self.pre_process.append(process_norm)
        self.pre_process.append(process_layout)

    def data_pre_process(self, data):
        data = copy.deepcopy(data)
        if self.pre_process[0]['preprocess'] and self.pre_process[3]['input_type'] == "float32":
            data = np.asarray(data, dtype=np.float32)
        if self.pre_process[0]['preprocess'] and len(data.shape) == 4:
            if self.pre_process[-1]['input_layout'] == 'NCHW':
                data = np.transpose(data, [0, 2, 3, 1])
            if self.pre_process[3]['input_type'] == "uint8":
                data *= 255.
            # elif self.cfg.case.compile_opt.kwargs['input_type'] == "int8":
            #     data *= 255.
            #     data -= 128.
            for item in self.pre_process:
                # dequantize
                if 'range' in item.keys() and 'input_type' in item.keys():
                    Q_max, Q_min = item['range'][1], item['range'][0]
                    if item['input_type'] == 'uint8':
                        range_min, range_max = 0, 255
                    elif item['input_type'] == 'int8':
                        range_min, range_max = -128, 127
                    else:
                        range_min, range_max = 0, 1
                    scale = (range_max - range_min) / (Q_max - Q_min)
                    bias = round((range_max * Q_min - range_min * Q_max) / (range_max - range_min))
                    data = data / scale + bias

                # swapRB
                if 'swapRB' in item.keys():
                    if data.shape[-1] != 3:
                        assert ("Please confirm your input channel is 3.")
                    if item['swapRB'] == True:
                        data = data[:, :, :, ::-1]
                        data = np.array(data)

                # LetterBox
                if 'input_range' in item.keys() and 'input_shape' in item.keys() and 'model_shape' in item.keys():
                    if item['input_shape'] != []:
                        model_shape: List = []
                        if self.model_type == "onnx" or self.model_type == "caffe":
                            model_shape = [1, item['model_shape'][2],
                                           item['model_shape'][3], item['model_shape'][1]]
                        else:
                            model_shape = item['model_shape']
                        if model_shape[1] != data.shape[1] or model_shape[2] != data.shape[2]:
                            in_h, in_w = data.shape[1], data.shape[2]
                            model_h, model_w = model_shape[1], model_shape[2]
                            ratio = min(model_h / in_h, model_w / in_w)
                            resize_shape = data.shape[0], int(round(
                                in_h * ratio)), int(round(in_w * ratio)), 3
                            resize_data = cv2.resize(data[0], (resize_shape[2],
                                                               resize_shape[1]), interpolation=cv2.INTER_LINEAR)
                            dh = model_shape[1] - resize_shape[1]
                            dw = model_shape[2] - resize_shape[2]
                            dh /= 2
                            dw /= 2
                            resize_data = np.array(resize_data, dtype=np.float32)
                            data = cv2.copyMakeBorder(resize_data, int(round(dh - 0.1)),
                                                      int(round(model_h - resize_shape[1] - round(dh - 0.1))),
                                                      int(round(dw - 0.1)), int(round(
                                    model_w - resize_shape[2] - round(dw - 0.1))), cv2.BORDER_CONSTANT, value=(
                                    item['letterbox_value'], item['letterbox_value'], item['letterbox_value']))

                            data = np.array(data, dtype=np.float32)
                            data = np.expand_dims(data, 0)

                # Normalize
                if 'norm' in item.keys():
                    for i in range(data.shape[-1]):
                        k = i
                        if data.shape[-1] > 3:
                            k = 0
                        data[:, :, :, i] = (data[:, :, :, i] - float(item['norm']['mean'][k])) / \
                                           float(item['norm']['std'][k])
        else:
            assert ("Please confirm your input shape and model shape is 4D!")

        return data

    def validte_config(self, config):
        in_ci = os.getenv('CI', False)
        if in_ci:
            config.judge.common.log_hist = False
            config.setup.log_txt = False
        return config

    def update_config(self, config: Edict, overwirte_configs: Dict) -> Edict:
        if overwirte_configs:
            if isinstance(overwirte_configs, str):
                overwirte_configs: dict = yaml.safe_load(overwirte_configs)
            config.update(overwirte_configs)
        return config

    def validate_targets(self, targets: List[str]):
        def _validate_targets(old_targets: List[str]):
            new_targets = []
            for t in old_targets:
                if nncase.test_target(t):
                    new_targets.append(t)
                else:
                    print("WARN: target[{0}] not found".format(t))
            return new_targets

        self.cfg.case.eval[0].update({"values": _validate_targets(
            targets if targets else self.cfg.case.eval[0].values)})
        self.cfg.case.infer[0].update({"values": _validate_targets(
            targets if targets else self.cfg.case.infer[0].values)})

    def run(self, model_path: Union[List[str], str]):
        # TODO add mulit process pool
        # case_name = self.process_model_path_name(model_path)
        # case_dir = os.path.join(self.cfg.setup.root, case_name)
        # if not os.path.exists(case_dir):
        #     os.makedirs(case_dir)
        if isinstance(model_path, str):
            case_dir = os.path.dirname(model_path)
        elif isinstance(model_path, list):
            case_dir = os.path.dirname(model_path[0])
        self.model_path = case_dir
        self.run_single(self.cfg.case, case_dir, model_path)
        if self.in_ci:
            shutil.rmtree(case_dir)

    def process_model_path_name(self, model_path: str) -> str:
        if Path(model_path).is_file():
            case_name = Path(model_path)
            return '_'.join(str(case_name.parent).split('/') + [case_name.stem])
        return model_path

    def clear(self, case_dir):
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
        os.makedirs(case_dir)

    @abstractmethod
    def parse_model_input_output(self, model_path: Union[List[str], str]):
        pass

    @abstractmethod
    def cpu_infer(self, case_dir: str, model_content: Union[List[str], str]):
        pass

    @abstractmethod
    def import_model(self, compiler, model_content, import_options):
        pass

    def run_single(self, cfg, case_dir: str, model_file: Union[List[str], str]):
        if not self.inputs:
            self.parse_model_input_output(model_file)

        on_board = self.in_ci and self.kpu_target is not None and self.port is not None and self.nncase_test_ci is not None and len(self.inputs) > 0 and len(self.outputs) > 0
        if on_board and cfg.generate_inputs.name == 'generate_imagenet_dataset':
            cfg.generate_inputs.batch_size = 1

        if on_board and cfg.generate_calibs.name == 'generate_imagenet_dataset':
            cfg.generate_calibs.batch_size = 1

        names, args = TestRunner.split_value(cfg.preprocess_opt)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            self.get_process_config(dict_args)
            self.generate_data(cfg.generate_inputs, case_dir,
                               self.inputs, self.input_paths, 'input', dict_args)
            self.generate_data(cfg.generate_calibs, case_dir,
                               self.calibs, self.calib_paths, 'calib', dict_args)
            self.generate_data(cfg.generate_dump_range_data, case_dir,
                               self.dump_range_data, self.dump_range_data_paths, 'dump_range_data', dict_args)

            # write preprocess options in test_result
            if dict_args['preprocess'] == True:
                str_preprocess_opt = ""
                pre_list = []
                for key, value in dict_args.items():
                    pre_list.append(str_preprocess_opt.join("{0}:{1}".format(key, value)))
                with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
                    f.write("\n----preprocess option----\n")
                    f.write('\n'.join(pre_list[:]) + "\n")
                    f.write("-------------------------\n")

            self.cpu_infer(case_dir, model_file, dict_args['input_type'],
                           "dataset" if cfg.generate_inputs.name == 'generate_imagenet_dataset' else "random")
            import_options, compile_options = self.get_compiler_options(dict_args, model_file)
            model_content = self.read_model_file(model_file)
            if cfg.generate_inputs.name != 'generate_imagenet_dataset':
                self.run_evaluator(cfg, case_dir, import_options,
                                   compile_options, model_content, dict_args)
            self.run_inference(cfg, case_dir, import_options,
                               compile_options, model_content, dict_args)

    def get_compiler_options(self, cfg, model_file):
        import_options = nncase.ImportOptions()
        compile_options = nncase.CompileOptions()
        if isinstance(model_file, str):
            if os.path.splitext(model_file)[-1] == ".tflite":
                compile_options.input_layout = cfg['input_layout']
                compile_options.output_layout = cfg['output_layout']
            elif os.path.splitext(model_file)[-1] == ".onnx":
                compile_options.input_layout = cfg['input_layout']
                compile_options.output_layout = cfg['output_layout']
        elif isinstance(model_file, list):
            if os.path.splitext(model_file[1])[-1] == ".caffemodel":
                compile_options.input_layout = cfg['input_layout']
                compile_options.output_layout = cfg['output_layout']

        for k, v in cfg.items():
            # model_layout just use in test_runner
            if k == "model_layout":
                continue
            e = '"'
            exec(f"compile_options.{k} = {e + v + e if isinstance(v, str) else v}")
        return import_options, compile_options

    def run_evaluator(self, cfg, case_dir, import_options, compile_options, model_content, preprocess_opt):
        names, args = TestRunner.split_value(cfg.eval)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            if dict_args['ptq'] and len(self.inputs) == 0:
                continue
            if cfg.compile_opt.dump_import_op_range and len(self.inputs) == 0:
                continue
            eval_output_paths = self.generate_evaluates(
                cfg, case_dir, import_options,
                compile_options, model_content, dict_args, preprocess_opt)
            judge, result = self.compare_results(
                self.output_paths, eval_output_paths, dict_args)
            assert (judge), 'Fault result in eval' + result

    def run_inference(self, cfg, case_dir, import_options, compile_options, model_content, preprocess_opt):
        names, args = TestRunner.split_value(cfg.infer)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            if dict_args['ptq'] and len(self.inputs) == 0:
                continue
            if cfg.compile_opt.dump_import_op_range and len(self.inputs) == 0:
                continue
            infer_output_paths = self.nncase_infer(
                cfg, case_dir, import_options,
                compile_options, model_content, dict_args, preprocess_opt)
            judge, result = self.compare_results(
                self.output_paths, infer_output_paths, dict_args)
            assert (judge), 'Fault result in infer' + result

    @staticmethod
    def split_value(kwcfg: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        arg_names = []
        arg_values = []
        for d in kwcfg:
            if type(d) is dict:
                arg_names.append(d['name'])
                arg_values.append(d['values'])
            else:
                arg_names.append(d.name)
                arg_values.append(d.values)
        return (arg_names, arg_values)

    def read_model_file(self, model_file: Union[List[str], str]):
        if isinstance(model_file, str):
            with open(model_file, 'rb') as f:
                return f.read()
        elif isinstance(model_file, list):
            model_content = []
            with open(model_file[0], 'rb') as f:
                model_content.append(f.read())
            with open(model_file[1], 'rb') as f:
                model_content.append(f.read())
            return model_content

    @staticmethod
    def kwargs_to_path(path: str, kwargs: Dict[str, str]):
        for k, v in kwargs.items():
            if isinstance(v, str):
                path = os.path.join(path, v)
            elif isinstance(v, bool):
                path = os.path.join(path, ('' if v else 'no') + k)
        return path

    def generate_evaluates(self, cfg, case_dir: str,
                           import_options: nncase.ImportOptions,
                           compile_options: nncase.CompileOptions,
                           model_content: Union[List[bytes], bytes],
                           kwargs: Dict[str, str],
                           preprocess: Dict[str, str]
                           ) -> List[Tuple[str, str]]:
        eval_dir = TestRunner.kwargs_to_path(
            os.path.join(case_dir, 'eval'), kwargs)
        compile_options.target = kwargs['target']
        compile_options.dump_dir = eval_dir
        compile_options.dump_asm = cfg.compile_opt.dump_asm
        compile_options.dump_ir = cfg.compile_opt.dump_ir
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)

        if cfg.compile_opt.dump_import_op_range:
            dump_range_options = nncase.DumpRangeTensorOptions()
            raw_inputs = [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in
                          self.dump_range_data]
            byte_inputs = np.asarray(raw_inputs[0]).tobytes()
            for i in range(1, len(raw_inputs)):
                byte_inputs += np.asarray(raw_inputs[i]).tobytes()
            dump_range_options.set_tensor_data(byte_inputs)
            dump_range_options.samples_count = cfg.generate_dump_range_data.batch_size
            compiler.dump_range_options(dump_range_options)
        if kwargs['ptq']:
            ptq_options = nncase.PTQTensorOptions()
            raw_inputs = [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in
                          self.calibs]
            byte_inputs = np.asarray(raw_inputs[0]).tobytes()
            for i in range(1, len(raw_inputs)):
                byte_inputs += np.asarray(raw_inputs[i]).tobytes()
            ptq_options.set_tensor_data(byte_inputs)
            ptq_options.samples_count = cfg.generate_calibs.batch_size
            compiler.use_ptq(ptq_options)

        evaluator = compiler.create_evaluator(3)
        eval_output_paths = []
        if cfg.generate_inputs.name == "generate_imagenet_dataset":
            # for i in range(len(self.inputs)):
            topk = []
            for in_data in self.inputs[0]['data']:
                input_tensor = nncase.RuntimeTensor.from_numpy(in_data[0])
                input_tensor.copy_to(evaluator.get_input_tensor(0))
                evaluator.run()
                result = evaluator.get_output_tensor(0).to_numpy()
                topk.append((in_data[1], get_topK(kwargs['target'], 1, result)))
            gnne_txt = "gnne_no_ptq" if kwargs['ptq'] is False else "gnne_ptq"
            eval_output_paths.append((
                os.path.join(eval_dir, gnne_txt) + '_0.bin',
                os.path.join(eval_dir, gnne_txt) + '_0.txt'))
            result.tofile(eval_output_paths[-1][0])
            with open(eval_output_paths[-1][1], 'a') as f:
                for i in range(len(topk)):
                    f.write(topk[i][0].split("/")[-1] + " " + str(topk[i][1]) + '\n')

        else:
            for i in range(len(self.inputs)):
                data = self.transform_input(self.data_pre_process(
                    self.inputs[i]['data']), "float32", "CPU")
                input_tensor = nncase.RuntimeTensor.from_numpy(data)
                if preprocess['preprocess']:
                    self.totxtfile(os.path.join(case_dir, f'eval_input.txt'), data)
                input_tensor.copy_to(evaluator.get_input_tensor(i))
                evaluator.run()

            for i in range(evaluator.outputs_size):
                result = evaluator.get_output_tensor(i).to_numpy()
                eval_output_paths.append((
                    os.path.join(eval_dir, f'nncase_result_{i}.bin'),
                    os.path.join(eval_dir, f'nncase_result_{i}.txt')))
                result.tofile(eval_output_paths[-1][0])
                self.totxtfile(eval_output_paths[-1][1], result)
        return eval_output_paths

    def nncase_infer(self, cfg, case_dir: str,
                     import_options: nncase.ImportOptions,
                     compile_options: nncase.CompileOptions,
                     model_content: Union[List[bytes], bytes],
                     kwargs: Dict[str, str],
                     preprocess: Dict[str, str]
                     ) -> List[Tuple[str, str]]:
        infer_dir = TestRunner.kwargs_to_path(
            os.path.join(case_dir, 'infer'), kwargs)
        compile_options.target = kwargs['target']
        compile_options.dump_dir = infer_dir
        compile_options.dump_asm = cfg.compile_opt.dump_asm
        compile_options.dump_ir = cfg.compile_opt.dump_ir
        compile_options.dump_quant_error = cfg.compile_opt.dump_quant_error
        compile_options.dump_import_op_range = cfg.compile_opt.dump_import_op_range
        compile_options.is_fpga = cfg.compile_opt.is_fpga
        compile_options.use_mse_quant_w = cfg.compile_opt.use_mse_quant_w
        compile_options.split_w_to_act = cfg.compile_opt.split_w_to_act
        compile_options.input_type = preprocess['input_type']
        compile_options.output_type = cfg.compile_opt.output_type
        compile_options.output_range = cfg.compile_opt.output_range
        compile_options.quant_type = cfg.compile_opt.quant_type
        compile_options.w_quant_type = cfg.compile_opt.w_quant_type
        compile_options.swapRB = preprocess['swapRB']
        if self.pre_process[0]['preprocess']:
            if self.pre_process[3]['input_shape'] != []:
                compile_options.input_shape = self.pre_process[3]['input_shape']
            else:
                if preprocess['model_layout'] == "":
                    if self.model_type == "tflite" and preprocess['input_layout'] == "NCHW":
                        compile_options.input_shape = np.array(
                            [self.pre_process[3]['model_shape'][0], self.pre_process[3]
                            ['model_shape'][3], self.pre_process[3]['model_shape'][1],
                             self.pre_process[3]['model_shape'][2]])
                    elif self.model_type != "tflite" and preprocess['input_layout'] == "NHWC":
                        compile_options.input_shape = np.array(
                            [self.pre_process[3]['model_shape'][0], self.pre_process[3]
                            ['model_shape'][2], self.pre_process[3]['model_shape'][3],
                             self.pre_process[3]['model_shape'][1]])
                else:
                    if preprocess['model_layout'] == "NHWC" and preprocess['input_layout'] == "NCHW":
                        compile_options.input_shape = np.array(
                            [self.pre_process[3]['model_shape'][0], self.pre_process[3]
                            ['model_shape'][3], self.pre_process[3]['model_shape'][1],
                             self.pre_process[3]['model_shape'][2]])
                    elif preprocess['model_layout'] == "NCHW" and preprocess['input_layout'] == "NHWC":
                        compile_options.input_shape = np.array(
                            [self.pre_process[3]['model_shape'][0], self.pre_process[3]
                            ['model_shape'][2], self.pre_process[3]['model_shape'][3],
                             self.pre_process[3]['model_shape'][1]])
                    else:
                        compile_options.input_shape = self.pre_process[3]['model_shape']
        else:
            compile_options.input_shape = self.pre_process[3]['model_shape']
        compile_options.input_range = preprocess['input_range']
        compile_options.preprocess = preprocess['preprocess']
        compile_options.mean = preprocess['mean']
        compile_options.std = preprocess['std']
        compile_options.input_layout = preprocess['input_layout']
        compile_options.output_layout = preprocess['output_layout']
        compile_options.model_layout = preprocess['model_layout']
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)

        if cfg.compile_opt.dump_import_op_range:
            dump_range_options = nncase.DumpRangeTensorOptions()
            raw_inputs = [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in
                          self.dump_range_data]
            byte_inputs = np.asarray(raw_inputs[0]).tobytes()
            for i in range(1, len(raw_inputs)):
                byte_inputs += np.asarray(raw_inputs[i]).tobytes()
            dump_range_options.set_tensor_data(byte_inputs)
            dump_range_options.samples_count = cfg.generate_dump_range_data.batch_size
            compiler.dump_range_options(dump_range_options)
        if kwargs['ptq']:
            ptq_options = nncase.PTQTensorOptions()
            if cfg.generate_calibs.name == "generate_imagenet_dataset":
                # ptq_options.set_tensor_data(np.asarray(
                #     [sample['data'] for sample in self.calibs]).tobytes())
                calib_len = len(self.calibs[0]['data'])
                byte_inputs = np.asarray(self.calibs[0]['data'][0][0]).tobytes()
                for i in range(1, len(self.calibs[0]['data'])):
                    byte_inputs += np.asarray(self.calibs[0]['data'][i][0]).tobytes()
                ptq_options.set_tensor_data(byte_inputs)
                ptq_options.calibrate_method = self.cfg.case.compile_opt.quant_method
            else:
                raw_inputs = [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in
                              self.calibs]
                byte_inputs = np.asarray(raw_inputs[0]).tobytes()
                for i in range(1, len(raw_inputs)):
                    byte_inputs += np.asarray(raw_inputs[i]).tobytes()
                ptq_options.set_tensor_data(byte_inputs)
                ptq_options.calibrate_method = self.cfg.case.compile_opt.quant_method
            ptq_options.samples_count = cfg.generate_calibs.batch_size
            compiler.use_ptq(ptq_options)

        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(os.path.join(infer_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)

        infer_output_paths: List[np.ndarray] = []

        on_board = self.in_ci and kwargs['target'] == self.kpu_target and self.port is not None and self.nncase_test_ci is not None and len(self.inputs) > 0 and len(self.outputs) > 0
        case_name = f'{os.path.basename(case_dir)}_{os.path.basename(infer_dir)}'

        if cfg.generate_inputs.name == "generate_imagenet_dataset":
            gnne_txt = "gnne_no_ptq" if kwargs['ptq'] is False else "gnne_ptq"
            infer_output_paths.append((
                os.path.join(infer_dir, gnne_txt) + '.bin',
                os.path.join(infer_dir, gnne_txt) + '.txt'))
            pool_num = 100
            ci_flag = os.getenv('LOCAL_CI', False)
            if ci_flag:
                pool_num = 10
            p = multiprocessing.Pool(pool_num)
            result = []
            for in_data in self.inputs[0]['data']:
                input_data = copy.deepcopy(in_data)
                if on_board:
                    on_board_run(kmodel, input_data, infer_output_paths, kwargs['target'], self.port, case_name, self.nncase_test_ci, len(self.inputs), len(self.outputs), self.model_type,
                        self.inputs[0]['model_shape'])
                else:
                    p.apply_async(sim_run, args=(
                        kmodel, input_data, infer_output_paths, kwargs['target'], self.model_type,
                        self.inputs[0]['model_shape']))
            p.close()
            p.join()

        else:
            if on_board:
                # connect server
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(('localhost', int(self.port)))

                # send header
                header_dict = {}
                header_dict['case'] = case_name
                header_dict['app'] = 1
                header_dict['kmodel']= 1
                header_dict['inputs'] = len(self.inputs)
                header_dict['outputs'] = len(self.outputs)
                client_socket.sendall(json.dumps(header_dict).encode())
                dummy = client_socket.recv(1024)

                # send app
                file_dict = {}
                file_dict['file_name'] = os.path.basename(self.nncase_test_ci)
                file_dict['file_size'] = os.path.getsize(self.nncase_test_ci)
                client_socket.sendall(json.dumps(file_dict).encode())
                dummy = client_socket.recv(1024)
                with open(self.nncase_test_ci, 'rb') as f:
                    client_socket.sendall(f.read())
                dummy = client_socket.recv(1024)

                # send kmodel
                file_dict['file_name'] = 'test.kmodel'
                file_dict['file_size'] = len(kmodel)
                client_socket.sendall(json.dumps(file_dict).encode())
                dummy = client_socket.recv(1024)
                client_socket.sendall(kmodel)
                dummy = client_socket.recv(1024)

                # send inputs
                for i in range(len(self.inputs)):
                    input_bin = os.path.join(case_dir, f'input_0_{i}.bin')
                    data = self.transform_input(
                        self.inputs[i]['data'], preprocess['input_type'], "infer")
                    dtype = preprocess['input_type']
                    if preprocess['preprocess']:
                        input_bin = os.path.join(case_dir, f'input_{i}_{dtype}.bin')
                        data.tofile(input_bin)
                        self.totxtfile(os.path.join(case_dir, f'input_{i}_{dtype}.txt'), data)

                    file_dict['file_name'] = f'input_0_{i}.bin'
                    file_dict['file_size'] = os.path.getsize(input_bin)
                    client_socket.sendall(json.dumps(file_dict).encode())
                    dummy = client_socket.recv(1024)
                    client_socket.sendall(data.tobytes())
                    dummy = client_socket.recv(1024)

                # infer result
                cmd_result = client_socket.recv(1024).decode()
                if cmd_result.find('succeed') != -1:
                    client_socket.sendall(f"pls send outputs".encode())

                    # recv outputs
                    for i in range(len(self.outputs)):
                        header = client_socket.recv(1024)
                        file_size = int(header.decode())
                        client_socket.sendall(f"pls send nncase_result_{i}.bin".encode())

                        recv_size = 0
                        buffer = bytearray(file_size)
                        while recv_size < file_size:
                            slice = client_socket.recv(4096)
                            buffer[recv_size:] = slice
                            recv_size += len(slice)

                        # save nncase_result
                        nncase_result = np.frombuffer(buffer, dtype=self.outputs[i]['dtype'])
                        nncase_result.tofile(os.path.join(infer_dir, f'nncase_result_{i}.bin'))
                        self.totxtfile(os.path.join(infer_dir, f'nncase_result_{i}.txt'), nncase_result)

                        # save nncase_vs_cpu_result
                        model_shape = self.outputs[i]['model_shape']
                        nncase_vs_cpu_result = nncase_result.reshape(model_shape)
                        if preprocess['preprocess'] and len(model_shape) == 4:
                            if (preprocess['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                                nncase_vs_cpu_result = nncase_result.reshape(model_shape[0], model_shape[2], model_shape[3], model_shape[1])
                                nncase_vs_cpu_result = np.transpose(nncase_vs_cpu_result, [0, 3, 1, 2])
                            elif (preprocess['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                                nncase_vs_cpu_result = nncase_result.reshape(model_shape[0], model_shape[3], model_shape[1], model_shape[2])
                                nncase_vs_cpu_result = np.transpose(nncase_vs_cpu_result, [0, 2, 3, 1])
                        infer_output_paths.append((
                            os.path.join(infer_dir, f'nncase_vs_cpu_result_{i}.bin'),
                            os.path.join(infer_dir, f'nncase_vs_cpu_result_{i}.txt')))
                        if cfg.compile_opt.output_type != "float32" and infer_dir.split('/')[-1] == "ptq":
                            nncase_vs_cpu_result.tofile(os.path.join(infer_dir, f'nncase_vs_cpu_result_{cfg.compile_opt.output_type}_{i}.bin'))
                            self.totxtfile(os.path.join(infer_dir, f'nncase_vs_cpu_result_{cfg.compile_opt.output_type}_{i}.txt'), nncase_vs_cpu_result)
                            nncase_vs_cpu_result = deq_output(os.path.join(infer_dir, f'kmodel_info.txt'), nncase_vs_cpu_result)
                        nncase_vs_cpu_result.tofile(infer_output_paths[-1][0])
                        self.totxtfile(infer_output_paths[-1][1], nncase_vs_cpu_result)

                        client_socket.sendall(f"recv nncase_result_{i}.bin succeed".encode())

                    client_socket.close()
                else:
                    client_socket.close()
                    raise Exception(f'{cmd_result}')
            else:
                # run in simulator
                sim = nncase.Simulator()
                sim.load_model(kmodel)
                for i in range(len(self.inputs)):
                    data = self.transform_input(
                        self.inputs[i]['data'], preprocess['input_type'], "infer")
                    dtype = preprocess['input_type']
                    if preprocess['preprocess']:
                        data.tofile(os.path.join(case_dir, f'input_{i}_{dtype}.bin'))
                        self.totxtfile(os.path.join(case_dir, f'input_{i}_{dtype}.txt'), data)

                    sim.set_input_tensor(i, nncase.RuntimeTensor.from_numpy(data))
                sim.run()

                for i in range(sim.outputs_size):
                    nncase_result = sim.get_output_tensor(i).to_numpy()

                    # save nncase_result
                    nncase_result.tofile(os.path.join(infer_dir, f'nncase_result_{i}.bin'))
                    self.totxtfile(os.path.join(infer_dir, f'nncase_result_{i}.txt'), nncase_result)

                    # save nncase_vs_cpu_result
                    model_shape = self.outputs[i]['model_shape']
                    nncase_vs_cpu_result = nncase_result
                    if preprocess['preprocess'] and len(model_shape) == 4:
                        if (preprocess['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                            nncase_vs_cpu_result = np.transpose(nncase_vs_cpu_result, [0, 3, 1, 2])
                        elif (preprocess['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                            nncase_vs_cpu_result = np.transpose(nncase_vs_cpu_result, [0, 2, 3, 1])
                    infer_output_paths.append((
                        os.path.join(infer_dir, f'nncase_vs_cpu_result_{i}.bin'),
                        os.path.join(infer_dir, f'nncase_vs_cpu_result_{i}.txt')))
                    if cfg.compile_opt.output_type != "float32" and infer_dir.split('/')[-1] == "ptq":
                        nncase_vs_cpu_result.tofile(os.path.join(infer_dir, f'nncase_vs_cpu_result_{cfg.compile_opt.output_type}_{i}.bin'))
                        self.totxtfile(os.path.join(infer_dir, f'nncase_vs_cpu_result_{cfg.compile_opt.output_type}_{i}.txt'), nncase_vs_cpu_result)
                        nncase_vs_cpu_result = deq_output(os.path.join(infer_dir, f'kmodel_info.txt'), nncase_vs_cpu_result)
                    nncase_vs_cpu_result.tofile(infer_output_paths[-1][0])
                    self.totxtfile(infer_output_paths[-1][1], nncase_vs_cpu_result)

        return infer_output_paths

    def on_test_start(self) -> None:
        pass

    def generate_data(self, cfg, case_dir: str, inputs: List[Dict], path_list: List[str], name: str, preprocess_opt):
        for n in range(cfg.numbers):
            i = 0
            for input in inputs:
                shape = []
                if preprocess_opt['preprocess']:
                    if preprocess_opt['input_shape'] != []:
                        assert (len(preprocess_opt['input_shape']) == 4)
                        shape = copy.deepcopy(preprocess_opt['input_shape'])
                    else:
                        if preprocess_opt['model_layout'] is None:
                            if self.model_type == "tflite" and preprocess_opt['input_layout'] == "NCHW":
                                shape = copy.deepcopy(np.array(
                                    [input['model_shape'][0], input['model_shape'][3], input['model_shape'][1],
                                     input['model_shape'][2]]))
                            elif self.model_type != "tflite" and preprocess_opt['input_layout'] == "NHWC":
                                shape = copy.deepcopy(np.array(
                                    [input['model_shape'][0], input['model_shape'][2], input['model_shape'][3],
                                     input['model_shape'][1]]))
                            else:
                                shape = copy.deepcopy(input['model_shape'])
                        else:
                            if preprocess_opt['model_layout'] == "NHWC" and preprocess_opt['input_layout'] == "NCHW":
                                shape = copy.deepcopy(np.array(
                                    [input['model_shape'][0], input['model_shape'][3], input['model_shape'][1],
                                     input['model_shape'][2]]))
                            elif preprocess_opt['model_layout'] == "NCHW" and preprocess_opt['input_layout'] == "NHWC":
                                shape = copy.deepcopy(np.array(
                                    [input['model_shape'][0], input['model_shape'][2], input['model_shape'][3],
                                     input['model_shape'][1]]))
                            else:
                                shape = copy.deepcopy(input['model_shape'])
                else:
                    shape = copy.deepcopy(input['model_shape'])
                if shape[0] != cfg.batch_size:
                    shape[0] *= cfg.batch_size
                if self.model_type != "tflite" and cfg.name == "generate_imagenet_dataset" and self.pre_process[5][
                    'model_layout'] == "NCHW":
                    shape = shape[0], shape[2], shape[3], shape[1]
                data = DataFactory[cfg.name](shape, input['dtype'], n,
                                             cfg.batch_size, self.model_path, **cfg.kwargs)

                if cfg.name != "generate_imagenet_dataset":
                    path_list.append(
                        (os.path.join(case_dir, f'{name}_{n}_{i}.bin'),
                         os.path.join(case_dir, f'{name}_{n}_{i}.txt')))
                    data.tofile(path_list[-1][0])
                    self.totxtfile(path_list[-1][1], data)
                i += 1
                input['data'] = data

    def process_input(self, inputs: List[np.array], **kwargs) -> None:
        pass

    def process_output(self, outputs: List[np.array], **kwargs) -> None:
        pass

    def on_test_end(self) -> None:
        pass

    def compare_results(self,
                        ref_ouputs: List[Tuple[str]],
                        test_outputs: List[Tuple[str]],
                        kwargs: Dict[str, str]) -> Tuple[bool, str]:

        judeg_cfg = copy.deepcopy(self.cfg.judge.common)
        if self.cfg.judge.specifics:
            for specific in self.cfg.judge.specifics:
                if kwargs['target'] in specific.matchs.dict['target'] and kwargs['ptq'] == specific.matchs.dict['ptq']:
                    judeg_cfg.update(specific)
                    break

        i = 0
        for ref_file, test_file in zip(ref_ouputs, test_outputs):

            judge, simarity_info = compare(test_file, ref_file,
                                           self.outputs[i]['dtype'],
                                           judeg_cfg.simarity_name,
                                           judeg_cfg.threshold,
                                           judeg_cfg.log_hist)
            name_list = test_file[1].split(os.path.sep)
            kw_names = ' '.join(name_list[-len(kwargs) - 2:-1])
            j = self.num_pattern.findall(name_list[-1])
            result_info = "\n{0} [ {1} ] Output: {2}!!\n".format(
                'Pass' if judge else 'Fail', kw_names, j)
            result = simarity_info + result_info
            # print(result) temp disable
            with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
                f.write(result)
            i = i + 1
            if not judge:
                return False, result
        return True, result

    def totxtfile(self, save_path, ndarray: np.array, bit_16_represent=False):
        if self.cfg.setup.log_txt:
            if bit_16_represent:
                np.save(save_path, _cast_bfloat16_then_float32(ndarray))
            else:
                if ndarray.dtype == np.uint8:
                    fmt = '%u'
                elif ndarray.dtype == np.int8:
                    fmt = '%d'
                else:
                    fmt = '%f'
                np.savetxt(save_path, ndarray.flatten(), fmt=fmt, header=str(ndarray.shape))

            print("----> %s" % save_path)
