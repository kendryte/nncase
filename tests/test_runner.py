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

import nncase
import struct
from compare_util import compare
import copy
import cv2
import numpy as np
import yaml
from inference import *
from evaluator import *

class Edict:
    def __init__(self, d: Dict[str, int]) -> None:
        assert(isinstance(d, dict)), "the Edict only accepct Dict for init"
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
                    for i in range(len(new_value['preprocess_opt'])):
                        self.case.preprocess_opt[i].values = copy.deepcopy(
                            new_value['preprocess_opt'][i]['values'])
                elif isinstance(new_value, dict):
                    old_value = getattr(self, name)
                    if old_value is None:
                        setattr(self, name, Edict(new_value))
                    elif isinstance(old_value, (Edict, dict)):
                        old_value.update(new_value)
                elif isinstance(new_value, (list, tuple)) and name == 'specifics':
                    setattr(self, name, [])
                    assert(hasattr(self, 'common')
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
                    abs: bool = False) -> np.ndarray:
    if dtype == np.uint8:
        data = np.random.randint(0, 256, shape)
    elif dtype == np.int8:
        data = np.random.randint(-128, 128, shape)
    elif dtype == bool:
        data = np.random.rand(*shape) > 0.5
    elif dtype == np.int32:
        data = np.random.randint(1, 5, size=shape, dtype='int32')
    elif dtype == np.int64:
        data = np.random.randint(1, 5, size=shape, dtype='int64')
        # data = np.random.randint(1, 128, size=shape, dtype='int64')
    else:
        data = np.random.rand(*shape)
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


def generate_image_dataset(shape: List[int], dtype: np.dtype,
                           batch_index: int, batch_size: int,
                           dir_path: str) -> np.ndarray:
    """ read image from folder, return the rgb image with padding, dtype = float32, range = [0,255]. same as k210 carmera.
    """
    assert(os.path.isdir(dir_path) or os.path.exists(dir_path))

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
    for p in img_paths[batch_index * batch_size:
                       (batch_index + 1) * batch_size]:
        img = cv2.imread(p)
        img = preproc(img, shape[1:3], False)  # img [h,w,c] rgb,
        imgs.append(img)
    return np.stack(imgs)


DataFactory = {
    'generate_random': generate_random,
    'generate_image_dataset': generate_image_dataset
}

# singleton
# if create compiler in each test, then will thorw exception: The configured user limit
globalCompiler = nncase.Compiler()

class TestRunner(Evaluator, Inference, metaclass=ABCMeta):
    def __init__(self, case_name, targets=None, overwrite_configs: Union[Dict, str] = None) -> None:
        config_root = os.path.dirname(__file__)
        with open(os.path.join(config_root, 'config.yml'), encoding='utf8') as f:
            cfg: dict = yaml.safe_load(f)
            config = Edict(cfg)
        config = self.update_config(config, overwrite_configs)
        self.cfg = self.validte_config(config)

        case_name = case_name.replace('[', '_').replace(']', '_')
        self.case_dir = os.path.join(self.cfg.setup.root, case_name)
        self.clear(self.case_dir)

        self.validate_targets(targets)

        self.inputs: List[Dict] = []
        self.calibs: List[Dict] = []
        self.dump_range_data: List[Dict] = []
        self.outputs: List[Dict] = []
        self.input_paths: List[Tuple[str, str]] = []
        self.calib_paths: List[Tuple[str, str]] = []
        self.dump_range_data_paths: List[Tuple[str, str]] = []
        self.output_paths: List[Tuple[str, str]] = []
        self.model_type: str = ""
        self.pre_process: List[Dict] = []

        self.num_pattern = re.compile("(\d+)")
        # [n, c, h, w].zip default_shape => [(n, 1), (c, 1), (h, 48), (w, 48)]
        self.default_shape = [1, 1, 48, 48, 24, 24]
        self.shape_vars = {}

    def transform_input(self, values: np.array, type: str, stage: str) -> np.ndarray:
        values = copy.deepcopy(values)
        if(len(values.shape) == 4 and self.pre_process[0]['preprocess']):
            if stage == "CPU":
                # onnx \ caffe
                if (self.model_type == "onnx" or self.model_type == "caffe"):
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

        self.pre_process.append(preprocess_flag)
        self.pre_process.append(process_deq)
        self.pre_process.append(process_format)
        self.pre_process.append(process_letterbox)
        self.pre_process.append(process_norm)
        self.pre_process.append(process_layout)

    def data_pre_process(self, data) -> np.ndarray:
        data = copy.deepcopy(data)
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
                    Q_max, Q_min = 0, 0
                    if item['input_type'] == 'uint8':
                        Q_max, Q_min = 255, 0
                    # elif item['input_type'] == 'int8':
                    #     Q_max, Q_min = 127, -128
                    else:
                        continue
                    scale = (item['range'][1] - item['range'][0]) / (Q_max - Q_min)
                    bias = round((item['range'][1] * Q_min - item['range'][0] *
                                  Q_max) / (item['range'][1] - item['range'][0]))
                    data = data * scale
                    data = data - bias

                # swapRB
                if 'swapRB' in item.keys():
                    if data.shape[-1] != 3:
                        assert("Please confirm your input channel is 3.")
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
                            resize_shape = data.shape[0], round(
                                in_h * ratio), round(in_w * ratio), 3
                            resize_data = cv2.resize(data[0], (resize_shape[2],
                                                               resize_shape[1]), interpolation=cv2.INTER_LINEAR)
                            dh = model_shape[1] - resize_shape[1]
                            dw = model_shape[2] - resize_shape[2]
                            dh /= 2
                            dw /= 2
                            resize_data = np.array(resize_data, dtype=np.float32)
                            data = cv2.copyMakeBorder(resize_data, round(dh - 0.1), round(model_h - resize_shape[1] - round(dh - 0.1)), round(dw - 0.1), round(
                                model_w - resize_shape[2] - round(dw - 0.1)), cv2.BORDER_CONSTANT, value=(item['letterbox_value'], item['letterbox_value'], item['letterbox_value']))

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
            assert("Please confirm your input shape and model shape is 4D!")

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
                if nncase.check_target(t):
                    new_targets.append(t)
                else:
                    print("WARN: target[{0}] not found".format(t))
            return new_targets
        self.cfg.case.eval[0].values = _validate_targets(
            targets if targets else self.cfg.case.eval[0].values)
        self.cfg.case.infer[0].values = _validate_targets(
            targets if targets else self.cfg.case.infer[0].values)

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
        self.run_single(self.cfg.case, case_dir, model_path)

    def process_model_path_name(self, model_path: str) -> str:
        if Path(model_path).is_file():
            case_name = Path(model_path)
            return '_'.join(str(case_name.parent).split('/') + [case_name.stem])
        return model_path

    def clear(self, case_dir):
        in_ci = os.getenv('CI', False)
        if in_ci:
            if os.path.exists(self.cfg.setup.root):
                shutil.rmtree(self.cfg.setup.root)
        else:
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir)
        os.makedirs(case_dir)

    @ abstractmethod
    def parse_model_input_output(self, model_path: Union[List[str], str]):
        pass

    @abstractmethod
    def cpu_infer(self, case_dir: str, model_content: Union[List[str], str]):
        pass

    @abstractmethod
    def import_model(self, compiler, model_content, import_options):
        pass

    def run_single(self, cfg, case_dir: str, model_file: Union[List[str], str]):
        # todo: move to run
        self.compiler = globalCompiler
        if not self.inputs:
            self.parse_model_input_output(model_file)
        for dict_args in self.make_args(cfg.preprocess_opt):
            self.get_process_config(dict_args)
            self.generate_all_data(case_dir, cfg, dict_args)
            self.write_preprocess_opt(dict_args)

            self.cpu_infer(case_dir, model_file, dict_args['input_type'])
            import_options, compile_options = self.get_compiler_options(dict_args, model_file)
            model_content = self.read_model_file(model_file)
            for eval_args in self.dispatch(cfg, cfg.eval):
                eval_output_paths = self.run_evaluator(eval_args, cfg, case_dir, import_options,
                                                       compile_options, model_content, dict_args)
                self.check_result(eval_output_paths, eval_args, 'eval')

            for infer_args in self.dispatch(cfg, cfg.infer):
                infer_output_paths = self.run_inference(infer_args, cfg, case_dir, import_options,
                                   compile_options, model_content, dict_args)
                self.check_result(infer_output_paths, infer_args, 'infer')

    def translate_shape(self, shape):
        if reduce(lambda x, y: x * y, shape) == 0:
            return [(i if i != 0 else d) for i, d in zip(shape, [3, 4, 256, 256])]
        else:
            return shape

    def set_shape_var(self, dict: Dict[str, int]):
        self.shape_vars = dict

    def check_result(self, nncase_output_paths, dict_args, stage):
        judge, result = self.compare_results(
            self.output_paths, nncase_output_paths, dict_args)
        assert(judge), f"Fault result in {stage} + {result}"

    def make_args(self, cfg_detail):
        names, args = self.split_value(cfg_detail)
        for combine_args in product(*args):
            yield dict(zip(names, combine_args))

    def dispatch(self, full_cfg, sub_cfg):
        for dict_args in self.make_args(sub_cfg):
            # if dict_args['ptq'] and len(self.inputs) != 1:
            #     continue
            # if full_cfg.compile_opt.dump_import_op_range and len(self.inputs) != 1:
            #     continue
            yield dict_args

    def set_quant_opt(self, cfg, kwargs, preprocess, compiler):
        if cfg.compile_opt.dump_import_op_range:
            dump_range_options = nncase.DumpRangeTensorOptions()
            dump_range_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.dump_range_data]).tobytes())
            dump_range_options.samples_count = cfg.generate_dump_range_data.batch_size
            # compiler.dump_range_options(dump_range_options)
        if kwargs['ptq']:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.calibs]))
            ptq_options.samples_count = cfg.generate_calibs.batch_size
            compiler.use_ptq(ptq_options, compiler._module.params)


    def write_preprocess_opt(self, dict_args):
        if dict_args['preprocess'] == True:
            str_preprocess_opt = ""
            pre_list = []
            for key, value in dict_args.items():
                pre_list.append(str_preprocess_opt.join("{0}:{1}".format(key, value)))
            with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
                f.write("\n----preprocess option----\n")
                f.write('\n'.join(pre_list[:]) + "\n")
                f.write("-------------------------\n")

    def generate_all_data(self, case_dir, cfg, dict_args):
        self.generate_data(cfg.generate_inputs, case_dir,
                           self.inputs, self.input_paths, 'input', dict_args)
        self.generate_data(cfg.generate_calibs, case_dir,
                           self.calibs, self.calib_paths, 'calib', dict_args)
        self.generate_data(cfg.generate_dump_range_data, case_dir,
                           self.dump_range_data, self.dump_range_data_paths, 'dump_range_data', dict_args)

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
            e = '"'
            exec(f"compile_options.{k} = {e + v + e if isinstance(v, str) else v}")
        return import_options, compile_options

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

    @ staticmethod
    def kwargs_to_path(path: str, kwargs: Dict[str, str]):
        for k, v in kwargs.items():
            if isinstance(v, str):
                path = os.path.join(path, v)
            elif isinstance(v, bool):
                path = os.path.join(path, ('' if v else 'no') + k)
        return path

    def on_test_start(self) -> None:
        pass

    def generate_data(self, cfg, case_dir: str, inputs: List[Dict], path_list: List[str], name: str, preprocess_opt):
        for n in range(cfg.numbers):
            i = 0
            for input in inputs:
                shape = copy.deepcopy(input['model_shape'])
                # if preprocess_opt['preprocess'] and preprocess_opt['input_shape'] != [] and len(preprocess_opt['input_shape']) == 4:
                #     shape = copy.deepcopy(preprocess_opt['input_shape'])
                # else:
                #     shape = copy.deepcopy(input['model_shape'])
                if shape[0] != cfg.batch_size:
                    shape[0] *= cfg.batch_size
                data = DataFactory[cfg.name](shape, input['dtype'], n, cfg.batch_size, **cfg.kwargs)

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
