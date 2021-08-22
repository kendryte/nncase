from posixpath import join
from typing import Sequence
import ncnn
import shutil
import os
import numpy as np
from numpy.core.defchararray import array
from numpy.lib.function_base import select
from test_runner import *
import io


class Layer:
    def __init__(self, type: str, name: str, inputs: Sequence[str] = [], outputs: Sequence[str] = [], params: dict = {}) -> None:
        self.type = type
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    def __str__(self) -> str:
        s = f"{self.type} {self.name} {len(self.inputs)} {len(self.outputs)}"
        s += ''.join([' ' + v for v in self.inputs])
        s += ''.join([' ' + v for v in self.outputs])
        s += ''.join([f" {k}={v}" for k, v in self.params.items()])
        return s

    def __repr__(self) -> str:
        return str(self)


class Net:
    def __init__(self) -> None:
        self.layers = []
        self.bin = io.BytesIO()
        self.blobs = 0

    def Input(self, name: str, w: int, h: int, c: int) -> str:
        self._add_layer("Input", name, outputs=[name], params={'0': w, '1': h, '2': c})
        return name

    def Convolution(self, name: str, input: str, outch: int, kernel_w: int, kernel_h: int,
                    dilation_w: int, dilation_h: int, stride_w: int, stride_h: int,
                    pad_left: int, pad_right: int, pad_top: int, pad_bottom: int,
                    pad_value: float, weights: np.ndarray, bias: np.ndarray = None) -> str:
        self._add_layer("Convolution", name, inputs=[input], outputs=[name], params={
            '0': outch,
            '1': kernel_w,
            '11': kernel_h,
            '2': dilation_w,
            '12': dilation_h,
            '3': stride_w,
            '13': stride_h,
            '4': pad_left,
            '15': pad_right,
            '14': pad_top,
            '16': pad_bottom,
            '18': pad_value,
            '5': 0 if bias is None else 1,
            '6': weights.size
        })
        self._add_bin(0, weights)
        if bias is not None:
            self._add_bin(1, bias)
        return name

    def _add_layer(self, type: str, name: str, inputs: Sequence[str] = [], outputs: Sequence[str] = [], params: dict = []):
        self.layers.append(Layer(type, name, inputs, outputs, params))
        self.blobs += len(outputs)

    def _add_bin(self, tag: int, arr: np.ndarray):
        self.bin.write(tag.to_bytes(4, byteorder='little'))
        self.bin.write(arr.tobytes())


class NcnnTestRunner(TestRunner):
    def __init__(self, case_name, targets=None, overwirte_configs: dict = None):
        super().__init__(case_name, targets, overwirte_configs)

    def from_ncnn(self, net: Net):
        param_file = os.path.join(self.case_dir, 'test.param')
        with open(param_file, 'w') as fp:
            print("7767517", file=fp)
            print(f"{len(net.layers)} {net.blobs}", file=fp)
            for l in net.layers:
                print(str(l), file=fp)

        bin_file = os.path.join(self.case_dir, 'test.bin')
        with open(bin_file, 'wb') as fb:
            fb.write(net.bin.getbuffer())

        return param_file, bin_file

    def run(self, param_file, bin_file):
        if self.case_dir != os.path.dirname(param_file):
            new_file = os.path.join(self.case_dir, 'test.param')
            shutil.copy(param_file, new_file)
            param_file = new_file
        if self.case_dir != os.path.dirname(bin_file):
            new_file = os.path.join(self.case_dir, 'test.bin')
            shutil.copy(bin_file, new_file)
            bin_file = new_file

        if not self.inputs:
            self.parse_model_input_output(param_file)

        super().run([param_file, bin_file])

    def parse_model_input_output(self, param_file: str):
        all_inputs = []
        all_outputs = []
        with open(param_file, 'r') as fp:
            for line in fp:
                tokens = line.split(' ', 4)
                if len(tokens) < 4:
                    continue
                op_type = tokens[0]
                name = tokens[1]
                in_num = int(tokens[2])
                out_num = int(tokens[3])
                tokens = tokens[-1].split(' ', in_num)
                inputs = tokens[:in_num]
                tokens = tokens[-1].split(' ', out_num)
                outputs = tokens[:out_num]
                all_inputs += inputs
                all_outputs += outputs

                if op_type != 'Input':
                    continue
                param_kvs = []
                for kv in tokens[-1].split(' '):
                    kv_tokens = kv.split('=')
                    param_kvs.append((int(kv_tokens[0]), int(kv_tokens[1])))
                params = dict(param_kvs)
                input_dict = {}
                input_dict['name'] = outputs[0]
                input_dict['dtype'] = np.float32
                input_dict['shape'] = [params[2], params[1], params[0]]
                self.inputs.append(input_dict)

                input_dict = {}
                input_dict['name'] = tokens[1]
                input_dict['dtype'] = np.float32
                input_dict['shape'] = [params[2], params[1], params[0]]
                self.calibs.append(input_dict)

        used_inputs = set(inputs)
        seen_outputs = set()
        for n in all_outputs:
            if not n in used_inputs and not n in seen_outputs:
                seen_outputs.add(n)
                input_dict = {}
                input_dict['name'] = n
                self.outputs.append(input_dict)

    def cpu_infer(self, case_dir: str, model_file: list[str]):
        with ncnn.Net() as net:
            ret = net.load_param(model_file[0])
            assert ret == 0
            ret = net.load_model(model_file[1])
            assert ret == 0

            with net.create_extractor() as ex:
                for input in self.inputs:
                    in_mat = ncnn.Mat(input['data'])
                    ex.input(input['name'], in_mat)

                i = 0
                for output in self.outputs:
                    bin_file = os.path.join(case_dir, f'cpu_result_{i}.bin')
                    text_file = os.path.join(case_dir, f'cpu_result_{i}.txt')
                    self.output_paths.append((bin_file, text_file))
                    out_mat = ncnn.Mat()
                    ex.extract(output['name'], out_mat)
                    out_arr = np.array(out_mat)
                    out_mat.release()
                    out_arr.tofile(bin_file)
                    self.totxtfile(text_file, out_arr)
                    i += 1

    def import_model(self, compiler, model_content, import_options):
        compiler.import_ncnn(model_content[0], model_content[1], import_options)
