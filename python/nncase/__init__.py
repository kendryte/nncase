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
#
"""nncase."""

from __future__ import annotations

import io
import itertools
import re
import subprocess
import shutil
import os
import sys
import string
import numpy as np
from pathlib import Path
from shutil import which
from typing import List
import platform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import _nncase
from _nncase import RuntimeTensor, TensorDesc, Simulator


def _initialize():
    compiler_path = os.getenv("NNCASE_COMPILER")
    if not compiler_path:
        compiler_path = os.path.join(os.path.dirname(_nncase.__file__),
                                     "nncase", "Nncase.Compiler.dll")
    _nncase.initialize(compiler_path)


_initialize()
# _nncase.launch_debugger()


class ImportOptions:
    def __init__(self) -> None:
        pass


class PTQTensorOptions:
    use_mix_quant: bool
    use_mse_quant_w: bool
    export_quant_scheme: bool
    export_weight_range_by_channel: bool
    dump_quant_error: bool
    dump_quant_error_symmetric_for_signed: bool
    quant_type: str
    w_quant_type: str
    calibrate_method: str
    finetune_weights_method: str
    input_mean: float
    input_std: float
    quant_scheme: str
    quant_scheme_strict_mode: bool
    samples_count: int
    cali_data: List[RuntimeTensor]

    def __init__(self) -> None:
        self.use_mix_quant: bool = False
        self.use_mse_quant_w = False
        self.export_quant_scheme: bool = False
        self.export_weight_range_by_channel: bool = False
        self.dump_quant_error: bool = False
        self.dump_quant_error_symmetric_for_signed: bool = True
        self.quant_type: str = "uint8"
        self.w_quant_type: str = "uint8"
        self.calibrate_method: str = "Kld"
        self.finetune_weights_method: str = "NoFineTuneWeights"
        self.input_mean: float = 0.5
        self.input_std: float = 0.5
        self.quant_scheme: str = ""
        self.quant_scheme_strict_mode: bool = False
        self.samples_count: int = 5
        self.cali_data: List[RuntimeTensor] = []

    def set_tensor_data(self, data: List[List[np.ndarray]]) -> None:
        reshape_data = list(map(list, zip(*data)))
        self.cali_data = [RuntimeTensor.from_numpy(
            d) for d in itertools.chain.from_iterable(reshape_data)]


class GraphEvaluator:
    _inputs: List[RuntimeTensor]
    _func: _nncase.Function
    _params: _nncase.Var
    _outputs: List[RuntimeTensor]

    def __init__(self, func: _nncase.Function) -> None:
        self._func = func
        self._params = func.parameters
        self._inputs = list([None] * len(self._params))
        self._outputs = None

    def get_input_tensor(self, index: int):
        assert index < len(self._inputs)
        tensor = self._inputs[index]
        return tensor.to_runtime_tensor() if tensor else None

    def set_input_tensor(self, index: int, value: RuntimeTensor):
        assert index < len(self._inputs)
        self._inputs[index] = _nncase.RTValue.from_runtime_tensor(value)

    def get_output_tensor(self, index: int):
        return self._outputs[index]

    def run(self):
        self._outputs = self._func.body.evaluate(self._params, self._inputs).to_runtime_tensors()

    @ property
    def outputs_size(self) -> int:
        return len(self._outputs)


class IRModule():
    _module: _nncase.IRModule = None

    def __init__(self, module: _nncase.IRModule):
        assert module.entry != None
        self._module = module

    @ property
    def entry(self) -> _nncase.IR.Function:
        return self._module.entry


class Compiler:
    _target: _nncase.Target
    _session: _nncase.CompileSession
    _compiler: _nncase.Compiler
    _compile_options: _nncase.CompileOptions
    _quantize_options: _nncase.QuantizeOptions
    _shape_bucket_options: _nncase.ShapeBucketOptions
    _module: IRModule

    def __init__(self, compile_options: CompileOptions) -> None:
        self._compile_options = _nncase.CompileOptions()
        self.__process_compile_options(compile_options)
        self._session = _nncase.CompileSession(self._target, self._compile_options)
        self._compiler = self._session.compiler
        self._quantize_options = None
        self._shape_bucket_options = _nncase.ShapeBucketOptions()
        self.init_shape_bucket_options(compile_options)

    def init_shape_bucket_options(self, compile_options: CompileOptions) -> None:
        self._shape_bucket_options = _nncase.ShapeBucketOptions()
        self._shape_bucket_options.segments_count = compile_options.shape_bucket_segments_count
        self._shape_bucket_options.enable = compile_options.shape_bucket_enable
        self._shape_bucket_options.range_info = compile_options.shape_bucket_range_info
        self._shape_bucket_options.segments_count = compile_options.shape_bucket_segments_count
        self._shape_bucket_options.fix_var_map = compile_options.shape_bucket_fix_var_map
        self._compile_options.shape_bucket_options = self._shape_bucket_options

    def compile(self) -> None:
        self._compiler.compile()

    @ property
    def module(self) -> IRModule:
        return self._module

    def create_evaluator(self, stage: int) -> GraphEvaluator:
        return GraphEvaluator(self._module.entry)

    def gencode(self, stream: io.RawIOBase) -> None:
        self._compiler.gencode(stream)

    def gencode_tobytes(self) -> bytes:
        code = io.BytesIO()
        self.gencode(code)
        return code.getvalue()

    def import_caffe(self, model: bytes, prototxt: bytes) -> None:
        raise NotImplementedError("import_caffe")

    def import_onnx(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.input_format = "onnx"
        self._import_onnx_module(model_content)

    def import_tflite(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.input_format = "tflite"
        self._import_tflite_module(model_content)

    def import_ncnn(self, model_param: bytes, model_bin: bytes, options: ImportOptions) -> None:
        self._compile_options.input_format = "ncnn"
        self._import_ncnn_module(model_param, model_bin)

    def use_ptq(self, ptq_dataset_options: PTQTensorOptions) -> None:
        dataset = [_nncase.RTValue.from_runtime_tensor(
            data) for data in ptq_dataset_options.cali_data]
        provider = _nncase.CalibrationDatasetProvider(
            dataset, ptq_dataset_options.samples_count, self._module.entry.parameters) if len(dataset) != 0 else []
        if not self._quantize_options:
            self._quantize_options = _nncase.QuantizeOptions()
            self._compile_options.quantize_options = self._quantize_options
        if len(dataset) != 0:
            self._quantize_options.calibration_dataset = provider
        self._quantize_options.model_quant_mode = _nncase.ModelQuantMode.UsePTQ

        if (ptq_dataset_options.calibrate_method == "NoClip"):
            self._quantize_options.calibrate_method = _nncase.CalibMethod.NoClip
        elif (ptq_dataset_options.calibrate_method == "Kld"):
            self._quantize_options.calibrate_method = _nncase.CalibMethod.Kld
        else:
            raise Exception("Unsupported Calibrate Method")

        if (ptq_dataset_options.finetune_weights_method == "NoFineTuneWeights"):
            self._quantize_options.finetune_weights_method = _nncase.FineTuneWeightsMethod.NoFineTuneWeights
        elif (ptq_dataset_options.finetune_weights_method == "UseSquant"):
            self._quantize_options.finetune_weights_method = _nncase.FineTuneWeightsMethod.UseSquant
        elif (ptq_dataset_options.finetune_weights_method == "UseAdaRound"):
            self._quantize_options.finetune_weights_method = _nncase.FineTuneWeightsMethod.UseAdaRound
        else:
            raise Exception("Unsupported Finetune Weights Method")

        if (ptq_dataset_options.quant_type == "uint8"):
            self._quantize_options.quant_type = _nncase.QuantType.Uint8
        elif (ptq_dataset_options.quant_type == "int8"):
            self._quantize_options.quant_type = _nncase.QuantType.Int8
        elif (ptq_dataset_options.quant_type == "int16"):
            self._quantize_options.quant_type = _nncase.QuantType.Int16
        else:
            raise Exception("Unsupported Quant Type")

        if (ptq_dataset_options.w_quant_type == "uint8"):
            self._quantize_options.w_quant_type = _nncase.QuantType.Uint8
        elif (ptq_dataset_options.w_quant_type == "int8"):
            self._quantize_options.w_quant_type = _nncase.QuantType.Int8
        elif (ptq_dataset_options.w_quant_type == "int16"):
            self._quantize_options.w_quant_type = _nncase.QuantType.Int16
        else:
            raise Exception("Unsupported Weights Quant Type")

        self._quantize_options.use_mix_quant = ptq_dataset_options.use_mix_quant
        self._quantize_options.quant_scheme = ptq_dataset_options.quant_scheme
        self._quantize_options.quant_scheme_strict_mode = ptq_dataset_options.quant_scheme_strict_mode
        self._quantize_options.export_quant_scheme = ptq_dataset_options.export_quant_scheme
        self._quantize_options.export_weight_range_by_channel = ptq_dataset_options.export_weight_range_by_channel
        self._quantize_options.dump_quant_error = ptq_dataset_options.dump_quant_error
        self._quantize_options.dump_quant_error_symmetric_for_signed = ptq_dataset_options.dump_quant_error_symmetric_for_signed

    def dump_range_options(self) -> DumpRangeTensorOptions:
        raise NotImplementedError("dump_range_options")

    def __process_compile_options(self, compile_options: CompileOptions) -> ClCompileOptions:
        self._target = _nncase.Target(compile_options.target)
        if compile_options.preprocess:
            self._compile_options.preprocess = compile_options.preprocess
            self._compile_options.swapRB = compile_options.swapRB
            if compile_options.input_type == "uint8":
                self._compile_options.input_type = _nncase.InputType.Uint8
            elif compile_options.input_type == "int8":
                self._compile_options.input_type = _nncase.InputType.Int8
            if compile_options.input_type == "float32":
                self._compile_options.input_type = _nncase.InputType.Float32
            self._compile_options.input_shape = str(compile_options.input_shape)[1:-1]
            self._compile_options.input_range = str(compile_options.input_range)[1:-1]
            self._compile_options.mean = str(compile_options.mean)[1:-1]
            self._compile_options.std = str(compile_options.std)[1:-1]
            self._compile_options.input_layout = compile_options.input_layout
            self._compile_options.output_layout = compile_options.output_layout
            self._compile_options.letterbox_value = compile_options.letterbox_value

        self._compile_options.input_file = compile_options.input_file
        dump_flags = _nncase.DumpFlags.Nothing if not compile_options.dump_ir else _nncase.DumpFlags(
            _nncase.DumpFlags.PassIR)
        if (compile_options.dump_asm):
            dump_flags = _nncase.DumpFlags(dump_flags | _nncase.DumpFlags.CodeGen)
        self._compile_options.dump_flags = dump_flags
        self._compile_options.dump_dir = compile_options.dump_dir

    def _import_onnx_module(self, model_content: bytes | io.RawIOBase) -> None:
        stream = io.BytesIO(model_content) if isinstance(model_content, bytes) else model_content
        self._module = IRModule(self._compiler.import_onnx_module(stream))

    def _import_tflite_module(self, model_content: bytes | io.RawIOBase) -> None:
        stream = io.BytesIO(model_content) if isinstance(model_content, bytes) else model_content
        self._module = IRModule(self._compiler.import_tflite_module(stream))

    def _import_ncnn_module(self, model_param: bytes | io.RawIOBase, model_bin: bytes | io.RawIOBase) -> None:
        param_stream = io.BytesIO(model_param) if isinstance(model_param, bytes) else model_param
        bin_stream = io.BytesIO(model_bin) if isinstance(model_bin, bytes) else model_bin
        self._module = IRModule(self._compiler.import_ncnn_module(param_stream, bin_stream))


def check_target(target: str):
    def test_target(target: str):
        return target in ["cpu", "k510", "k230", "xpu"]

    def target_exists(target: str):
        return _nncase.Target.exists(target)

    return test_target(target) and target_exists(target)


class DumpRangeTensorOptions:
    calibrate_method: str
    samples_count: int

    def set_tensor_data(self, data: bytes):
        pass


class CalibMethod:
    NoClip: int = 0
    Kld: int = 1
    Random: int = 2


class ModelQuantMode:
    NoQuant: int = 0
    UsePTQ: int = 1
    UseQAT: int = 2


class ClQuantizeOptions():
    CalibrationDataset: object
    CalibrationMethod: CalibMethod
    BindQuantMethod: bool
    UseSquant: bool
    UseAdaRound: bool


class ClCompileOptions():
    InputFile: str
    InputFormat: str
    Target: str
    DumpLevel: int
    DumpDir: str
    QuantType: int
    WQuantType: int
    OutputFile: str
    ModelQuantMode: int
    QuantizeOptions: ClQuantizeOptions
    SwapRB: bool
    InputRange: List[float]
    InputShape: List[int]
    InputType: str
    Mean: List[float]
    Std: List[float]
    PreProcess: bool
    InputLayout: str
    OutputLayout: str
    LetterBoxValue: float


class CompileOptions:
    target: str
    preprocess: bool
    swapRB: bool
    input_type: str
    input_shape: List[int]
    input_range: List[float]
    input_file: str
    mean: List[float]
    std: List[float]
    input_layout: str
    output_layout: str
    letterbox_value: float
    dump_asm: bool
    dump_ir: bool
    dump_dir: str
    shape_bucket_enable: bool
    shape_bucket_range_info: dict
    shape_bucket_segments_count: int
    shape_bucket_fix_var_map: dict

    def __init__(self) -> None:

        self.target = "cpu"
        self.preprocess = False
        self.swapRB = False
        self.input_type = "float32"
        self.input_shape = []
        self.input_range = []
        self.input_file = ""
        self.mean = [0, 0, 0]
        self.std = [1, 1, 1]
        self.input_layout = ""
        self.output_layout = ""
        self.letterbox_value = 0
        self.dump_asm = True
        self.dump_ir = False
        self.dump_dir = "tmp"
        self.shape_bucket_enable = False
        self.shape_bucket_range_info = {}
        self.shape_bucket_segments_count = 2
        self.shape_bucket_fix_var_map = {}


class ShapeBucketOptions:
    enable: bool
    var_map: dict
    range_info: dict
    segments_count: int
    fix_var_map: dict

    def __init__(self) -> None:
        self.enable = False
        self.var_map = {}
        self.range_info = {}
        self.segments_count = 2
        self.fix_var_map = {}
