# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import copy
import os
import re
import shutil
import struct
from abc import ABCMeta, abstractmethod
from itertools import product
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import collections.abc

import nncase
import copy
import cv2
from PIL import Image
import numpy as np
import toml
from generator import *
from inference import *
from evaluator import *
from compare_util import *
from test_utils import *
from html import escape


class TestRunner(Evaluator, Inference, metaclass=ABCMeta):
    def __init__(self, case_name, override_cfg: str = None) -> None:
        config_root = os.path.dirname(__file__)
        default_cfg = toml.load(os.path.join(config_root, 'config.toml'))

        new_cfg = None
        if override_cfg is not None:
            if os.path.isfile(override_cfg):
                new_cfg = toml.load(override_cfg)
            else:
                new_cfg = toml.loads(override_cfg)
        config = self.update_config(default_cfg, new_cfg)
        self.cfg = self.validte_config(config)

        case_name = case_name.replace('[', '_').replace(']', '_')
        self.case_dir = os.path.join(self.cfg['root'], case_name)
        self.clear(self.case_dir)
        os.makedirs(self.case_dir)

        self.inputs: List[Dict] = []
        self.calibs: List[Dict] = []
        self.outputs: List[Dict] = []
        self.model_type: str = ""
        self.pre_process: List[Dict] = []

        self.num_pattern = re.compile("(\d+)")
        # [n, c, h, w].zip default_shape => [(n, 1), (c, 1), (h, 48), (w, 48)]
        self.default_shape = [1, 1, 48, 48, 24, 24]
        self.shape_vars = {}
        # used for tag dynamic model
        self.dynamic = False

        if self.cfg['infer_report_opt']['enabled']:
            self.infer_report_file = test_utils.infer_report_file(
                self.cfg['infer_report_opt']['report_name'])
            self.infer_report_dict = {
                'priority': 100,
                'kind': 'N/A',
                'model': 'N/A',
                'shape': 'N/A',
                'if_quant_type': 'uint8',
                'w_quant_type': 'uint8',
            }

    def transform_input(self, values: List[np.ndarray], type: str, stage: str) -> List[np.ndarray]:
        new_values = []
        compile_opt = self.cfg['compile_opt']
        for value in values:
            new_value = copy.deepcopy(value)
            if compile_opt['preprocess']:
                if stage == "CPU":
                    # onnx \ caffe
                    if (self.model_type == "onnx" or self.model_type == "caffe"):
                        if len(new_value.shape) == 4:
                            new_value = np.transpose(new_value, [0, 3, 1, 2])
                        else:
                            new_value = np.transpose(
                                new_value, [int(idx) for idx in compile_opt['input_layout'].split(",")])

                if type == 'float32':
                    new_value = new_value.astype(np.float32)
                elif type == 'uint8':
                    if new_value.dtype == np.float32:
                        new_value = (new_value * 255).astype(np.uint8)
                elif type == 'int8':
                    if new_value.dtype == np.float32:
                        new_value = (new_value * 255 - 128).astype(np.int8)
                else:
                    raise TypeError(" Not support type for quant input")

            new_values.append(new_value)
        return new_values

    def data_pre_process(self, values: List[np.ndarray]) -> List[np.ndarray]:
        new_values = []
        compile_opt = self.cfg['compile_opt']

        compile_opt['model_shape'] = self.inputs[0]['model_shape'] if self.inputs else config['input_shape']

        for value in values:
            new_value = copy.deepcopy(value)
            if compile_opt['preprocess'] and len(value.shape) == 4:
                if compile_opt['input_layout'] in ['NCHW', "0,2,3,1"]:
                    new_value = np.transpose(new_value, [0, 2, 3, 1])

                # dequantize
                if 'input_range' in compile_opt.keys() and 'input_type' in compile_opt.keys():
                    Q_max, Q_min = 0, 0
                    if compile_opt['input_type'] == 'uint8':
                        Q_max, Q_min = 255, 0
                    else:
                        continue
                    scale = (compile_opt['input_range'][1] -
                             compile_opt['input_range'][0]) / (Q_max - Q_min)
                    bias = round((compile_opt['input_range'][1] * Q_min - compile_opt['input_range'][0] *
                                  Q_max) / (compile_opt['input_range'][1] - compile_opt['input_range'][0]))
                    new_value = new_value * scale
                    new_value = new_value - bias

                # swapRB
                if 'swapRB' in compile_opt.keys():
                    if new_value.shape[-1] != 3:
                        assert("Please confirm your input channel is 3.")
                    if compile_opt['swapRB'] == True:
                        new_value = new_value[:, :, :, ::-1]
                        new_value = np.array(new_value)

                # LetterBox
                if 'input_range' in compile_opt.keys() and 'input_shape' in compile_opt.keys() and 'model_shape' in compile_opt.keys():
                    if compile_opt['input_shape'] != []:
                        model_shape: List = []
                        if self.model_type in ["onnx", "caffe"] and compile_opt['model_layout'] != 'NHWC':
                            model_shape = [compile_opt['model_shape'][0], compile_opt['model_shape'][2],
                                           compile_opt['model_shape'][3], compile_opt['model_shape'][1]]
                        else:
                            model_shape = compile_opt['model_shape']
                        if model_shape[1] != new_value.shape[1] or model_shape[2] != new_value.shape[2]:
                            in_h, in_w = new_value.shape[1], new_value.shape[2]
                            model_h, model_w = model_shape[1], model_shape[2]
                            ratio = min(model_h / in_h, model_w / in_w)
                            resize_shape = new_value.shape[0], round(
                                in_h * ratio), round(in_w * ratio), 3
                            resize_data = np.random.rand(*model_shape)
                            for batch_data in new_value:
                                tmp = cv2.resize(
                                    batch_data, (resize_shape[2], resize_shape[1]), interpolation=cv2.INTER_LINEAR)

                                dh = model_shape[1] - resize_shape[1]
                                dw = model_shape[2] - resize_shape[2]
                                dh /= 2
                                dw /= 2
                                tmp = np.array(tmp, dtype=np.float32)
                                tmp = cv2.copyMakeBorder(tmp, round(dh - 0.1), round(model_h - resize_shape[1] - round(dh - 0.1)), round(dw - 0.1), round(
                                    model_w - resize_shape[2] - round(dw - 0.1)), cv2.BORDER_CONSTANT, value=(compile_opt['letterbox_value'], compile_opt['letterbox_value'], compile_opt['letterbox_value']))
                                tmp = np.expand_dims(tmp, 0)
                                # print("resize_data.shape = ", resize_data.shape)
                                # print("tmp.shape = ", tmp.shape)
                                resize_data = np.concatenate([resize_data, tmp], axis=0)
                            new_value = np.array(resize_data[1:], dtype=np.float32)

                # Normalize
                if 'mean' in compile_opt.keys() and 'std' in compile_opt.keys():
                    for i in range(new_value.shape[-1]):
                        k = i
                        if new_value.shape[-1] > 3:
                            k = 0
                        new_value[:, :, :, i] = (new_value[:, :, :, i] - float(compile_opt['mean'][k])) / \
                            float(compile_opt['std'][k])
            else:
                assert("Please confirm your input shape and model shape is 4D!")
            new_values.append(new_value)

        return new_values

    def validte_config(self, config):
        # disable all dump in CI
        if test_utils.in_ci():
            config['dump_hist'] = False
            config['compile_opt']['dump_asm'] = False
            config['compile_opt']['dump_quant_error'] = False

        # check target
        for k, v in config['target'].items():
            if not nncase.check_target(k):
                v['eval'] = False
                v['infer'] = False
                print("WARN: target[{0}] not found".format(k))

        # disable cpu target in k230/k510 CI
        if test_utils.in_ci() and test_utils.kpu_targets() != ['']:
            config['target']['cpu']['eval'] = False
            config['target']['cpu']['infer'] = False

        return config

    def update_config(self, config: Dict, override_cfg: Dict) -> Dict:
        if override_cfg:
            for k, v in override_cfg.items():
                if isinstance(v, collections.abc.Mapping):
                    config[k] = self.update_config(config.get(k, {}), v)
                else:
                    config[k] = v
            return config

        return config

    def clear(self, case_dir):
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)

    @ abstractmethod
    def parse_model(self, model_path: Union[List[str], str]):
        pass

    @abstractmethod
    def cpu_infer(self, case_dir: str, model_content: Union[List[str], str]):
        pass

    @abstractmethod
    def import_model(self, compiler, model_content, import_options):
        pass

    def config_cmds(self):
        return []

    def stat_target(self, infer_dir, results):
        pass

    def run(self, model_file: Union[List[str], str]):
        if not self.inputs:
            self.parse_model(model_file)

        self.generate_all_data()
        self.write_compile_opt()

        expected = self.cpu_infer(model_file)
        targets = self.cfg['target']
        model_content = self.read_model_file(model_file)
        import_options = nncase.ImportOptions()

        compiler = None
        dump_hist = self.cfg['dump_hist']
        for k_target, v_target in targets.items():
            tmp_dir = os.path.join(self.case_dir, 'tmp')
            if v_target['eval'] or v_target['infer']:
                compile_options = self.get_compile_options(k_target, tmp_dir)
                compiler = nncase.Compiler(compile_options)
                self.import_model(compiler, model_content, import_options)

            for stage in ['eval', 'infer']:
                if v_target[stage]:
                    for k_mode, v_mode in v_target['mode'].items():
                        if v_mode['enabled']:
                            os.makedirs(tmp_dir, exist_ok=True)
                            if stage == 'eval':
                                actual = self.run_evaluator(compiler, tmp_dir)
                            else:
                                actual = self.run_inference(
                                    compiler, k_target, k_mode == "ptq" and v_mode['enabled'], tmp_dir)
                            target_dir = os.path.join(self.case_dir, stage, k_target)
                            os.makedirs(target_dir, exist_ok=True)
                            mode_dir = os.path.join(target_dir, k_mode)
                            shutil.move(tmp_dir, mode_dir)
                            judge, result = self.compare_results(
                                expected, actual, stage, k_target, v_target['similarity_name'], k_mode, v_mode['threshold'], dump_hist, mode_dir)

                            if stage == 'infer' and self.cfg['infer_report_opt']['enabled']:
                                self.infer_report_dict['result'] = 'Pass' if judge else 'Fail'
                                self.infer_report_dict['remark'] = escape(
                                    result).replace('\n', '<br/>')
                                prefix, suffix = os.path.splitext(self.infer_report_file)
                                json_file = f'{prefix}_{os.path.basename(self.case_dir)}{suffix}'
                                dump_dict_to_json(self.infer_report_dict, json_file)
                            if not judge:
                                if test_utils.in_ci():
                                    self.clear(self.case_dir)
                                assert (judge), f"Fault result in {stage} + {result}"

        if test_utils.in_ci():
            self.clear(self.case_dir)

    def translate_shape(self, shape):
        if reduce(lambda x, y: x * y, shape) == 0:
            return [(i if i != 0 else d) for i, d in zip(shape, [3, 4, 256, 256])]
        else:
            return shape

    def set_shape_var(self, dict: Dict[str, int]):
        self.shape_vars = dict

    def set_quant_opt(self, compiler: nncase.Compiler):
        compile_opt = self.cfg['compile_opt']
        ptq_opt = self.cfg['ptq_opt']

        ptq_options = nncase.PTQTensorOptions()
        e = '"'
        for k, v in ptq_opt.items():
            exec(f"ptq_options.{k} = {e + v + e if isinstance(v, str) else v}")

        ptq_options.samples_count = self.cfg['generator']['calibs']['number']
        data = [self.transform_input(
            input['data'], compile_opt['input_type'], "infer") for input in self.calibs]
        ptq_options.set_tensor_data(data)

        compiler.use_ptq(ptq_options)

    def write_compile_opt(self):
        dict = self.cfg['compile_opt']
        if dict['preprocess'] == True:
            str_compile_opt = ""
            pre_list = []
            for key, value in dict.items():
                pre_list.append(str_compile_opt.join("{0}:{1}".format(key, value)))
            with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
                f.write("\n----compile option----\n")
                f.write('\n'.join(pre_list[:]) + "\n")
                f.write("-------------------------\n")

    def generate_all_data(self):
        self.generate_data('input', self.inputs,
                           self.cfg['compile_opt'], self.cfg['generator']['inputs'])
        self.generate_data('calib', self.calibs,
                           self.cfg['compile_opt'], self.cfg['generator']['calibs'])

    def get_compile_options(self, target, dump_dir):
        compile_options = nncase.CompileOptions()

        # update preprocess option
        compile_opt = self.cfg['compile_opt']
        e = '"'
        for k, v in compile_opt.items():
            # TODO: support model with unusual layout e.g.: onnx->NHWC
            if k == "model_layout" or k == "model_shape":
                continue
            if k == "target_options":
                target_options = nncase.CpuTargetOptions() if target == 'cpu' else None
                if target_options is not None:
                    for subk, subv in v.items():
                        exec(
                            f"target_options.{subk} = {e + subv + e if isinstance(v, str) else subv}")
                    compile_options.target_options = target_options
                continue
            exec(f"compile_options.{k} = {e + v + e if isinstance(v, str) else v}")

        compile_options.target = target
        compile_options.dump_dir = dump_dir

        return compile_options

    @staticmethod
    def split_value(kwcfg: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        arg_names = []
        arg_values = []
        for d in kwcfg:
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

    def generate_data(self, name: str, inputs: List[Dict], compile_opt, generator_cfg):
        os.mkdir(os.path.join(self.case_dir, name))
        method = generator_cfg['method']
        batch_number = generator_cfg['number']
        args = generator_cfg[method]['args']
        file_list = []

        if method == 'random':
            assert args in (True, False)
        elif method == 'bin':
            assert(os.path.isdir(args))
            for file in os.listdir(args):
                if file.endswith('.bin'):
                    file_list.append(os.path.join(args, file))
            file_list.sort()
        elif method == 'image':
            file_list.extend([os.path.join(args, p) for p in os.listdir(args)])
        elif method == 'constant_of_shape':
            assert len(args) != 0
        elif method == 'numpy':
            assert(os.path.isdir(args))
            for file in os.listdir(args):
                if file.endswith('.npy'):
                    file_list.append(os.path.join(args, file))
            file_list.sort()
        else:
            assert '{0} : not supported generator method'.format(method)

        generator = Generator()
        for input_idx, input in enumerate(inputs):
            samples = []
            input_shape = []
            dtype = input['dtype']
            if compile_opt['preprocess'] and compile_opt['input_shape'] != []:
                input_shape = copy.deepcopy(compile_opt['input_shape'])
                if compile_opt['input_type'] == "uint8":
                    dtype = np.uint8
                elif compile_opt['input_type'] == "float32":
                    dtype = np.float32
            else:
                input_shape = copy.deepcopy(input['model_shape'])
            if input_shape != [] and input_shape[0] != generator_cfg['batch']:
                input_shape[0] *= generator_cfg['batch']

            for batch_idx in range(batch_number):
                idx = input_idx * batch_number + batch_idx
                if method == 'random':
                    data = generator.from_random(input_shape, dtype, args)
                elif method == 'bin':
                    assert(idx < len(file_list))
                    data = generator.from_bin(input_shape, dtype, file_list[idx])
                elif method == 'image':
                    assert(idx < len(file_list))
                    data = generator.from_image(input_shape, dtype, file_list[idx])
                elif method == 'constant_of_shape':
                    data = generator.from_constant_of_shape(args, dtype)
                elif method == 'numpy':
                    data = generator.from_numpy(file_list[idx])
                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(self.case_dir, name,
                                               f'{name}_{input_idx}_{batch_idx}.bin'), data)
                    dump_txt_file(os.path.join(self.case_dir, name,
                                               f'{name}_{input_idx}_{batch_idx}.txt'), data)
                samples.append(data)
            input['data'] = samples

    def compare_results(self,
                        ref_ouputs: List[np.ndarray],
                        test_outputs: List[np.ndarray],
                        stage, target, similarity_name, mode, threshold, dump_hist, dump_dir) -> Tuple[bool, str]:
        i = 0
        judges = []
        result = ''
        for expected, actual in zip(ref_ouputs, test_outputs):
            expected = expected.astype(np.float32)
            actual = actual.astype(np.float32)
            dump_file = os.path.join(dump_dir, 'nncase_result_{0}_hist.csv'.format(i))
            judge, similarity_info = compare_ndarray(
                expected, actual, similarity_name, threshold, dump_hist, dump_file)
            result_info = "{0} [ {1} {2} {3} ] Output {4}:".format(
                'Pass' if judge else 'Fail', stage, target, mode, i)
            result += result_info + similarity_info
            i = i + 1
            judges.append(judge)

        with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
            f.write(result)
        return sum(judges) == len(judges), result
