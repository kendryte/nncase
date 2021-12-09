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
import clr
import sys
import os
import shutil


def _add_dllpath():
    nncase_cli_path = os.getenv("NNCASE_CLI")
    clr.AddReference("System.IO")
    clr.AddReference("System.Collections")
    for dll in ["Nncase.Cli",
                "Nncase.Core",
                "Nncase.EGraph",
                "Nncase.Graph",
                "Nncase.Evaluator",
                "Nncase.Importer",
                "Nncase.Pattern",
                "TorchSharp"]:
        clr.AddReference(os.path.join(nncase_cli_path, dll))

    naive_path = os.path.join(nncase_cli_path, 'runtimes', sys.platform + '-x64', 'native')
    for root, dirs, files in os.walk(naive_path):
        for file in files:
            dll_file = os.path.join(naive_path, file)
            target_file = os.path.join(nncase_cli_path, file)
            if not os.path.exists(target_file):
                shutil.copy(dll_file, target_file)

_add_dllpath()
from io import BytesIO
import numpy as np
from typing import Any, List, Dict, Tuple, Union
from System.IO import MemoryStream, Stream
from System.Collections.Generic import Dictionary
from System import (Array, Byte,
                    Int16,
                    Int32,
                    Int64,
                    SByte,
                    UInt16,
                    UInt32,
                    UInt64,
                    Decimal,
                    Double,
                    Single)
import Nncase as _nncase
from TorchSharp import torch as _torch


class ImportOptions:
    def __init__(self) -> None:
        pass


class PTQTensorOptions:
    calibrate_method: str
    input_mean: float
    input_std: float
    samples_count: int

    def __init__(self) -> None:
        pass

    def set_tensor_data(self, data: bytes) -> None:
        pass


class RuntimeTensor:
    _arr: _torch.Tensor
    netTypemap = {
        np.int8: SByte,
        np.int16: Int16,
        np.int32: Int32,
        np.int64: Int64,
        np.uint8: Byte,
        # np.float16: Float16,
        np.float32: Single,
        np.float64: Double,
    }
    npTypemap = {
        np.bool8: _torch.ScalarType.Bool,
        np.int8: _torch.ScalarType.Int8,
        np.int16: _torch.ScalarType.Int16,
        np.int32: _torch.ScalarType.Int32,
        np.int64: _torch.ScalarType.Int64,
        np.uint8: _torch.ScalarType.Byte,
        np.float16: _torch.ScalarType.Float16,
        np.float32: _torch.ScalarType.Float32,
        np.float64: _torch.ScalarType.Float64,
    }

    torchTypemap = {v: k for k, v in npTypemap.items()}

    def __init__(self, arr: Union[np.ndarray, _torch.Tensor]) -> None:
        if isinstance(arr, np.ndarray):
            arr = _torch.tensor(
                Array[self.netTypemap[arr.dtype.type]](arr.ravel().tolist()),
                Array[Int64](arr.shape),
                self.npTypemap[arr.dtype.type])
        self._arr = arr

    def copy_to(self, to: RuntimeTensor) -> None:
        to._arr = self._arr

    @staticmethod
    def from_numpy(arr: np.ndarray) -> RuntimeTensor:
        return RuntimeTensor(arr)

    def to_numpy(self) -> np.ndarray:
        if (not self._arr.is_contiguous()):
            self._arr = self._arr.contiguous()
        arr_bytes = _nncase.Evaluator.TorchExtentsion.ToSpan(self._arr)
        return np.frombuffer(arr_bytes, dtype=self.torchTypemap[self._arr.dtype]).reshape(self._arr.shape)

    def to_nncase(self) -> _torch.Tensor:
        return self._arr

    @ property
    def dtype(self) -> dtype:
        return self._arr.dtype

    @ property
    def shape(self) -> List[int]:
        return self._arr.shape


class MemoryRange:
    dtype: dtype
    location: int
    size: int
    start: int

    def __init__(self) -> None:
        pass


class Simulator:
    def __init__(self) -> None:
        pass

    def get_input_desc(self, index: int) -> MemoryRange:
        pass

    def get_input_tensor(self, index: int) -> RuntimeTensor:
        pass

    def get_output_desc(self, index: int) -> MemoryRange:
        pass

    def get_output_tensor(self, index: int) -> RuntimeTensor:
        pass

    def load_model(self, model: bytes) -> None:
        pass

    def run(self) -> None:
        pass

    def set_input_tensor(self, index: int, tensor: RuntimeTensor) -> None:
        pass

    def set_output_tensor(self, index: int, tensor: RuntimeTensor) -> None:
        pass

    @ property
    def inputs_size(self) -> int:
        pass

    @ property
    def outputs_size(self) -> int:
        pass


class GraphEvaluator:
    _inputs: List[RuntimeTensor]
    _module: Module
    _outputs: List[RuntimeTensor]

    def __init__(self, module: Module) -> None:
        self._module = module
        self._inputs = list([RuntimeTensor(None) for p in self._module.params])
        self._outputs = None

    def get_input_tensor(self, index: int):
        assert index < len(self._inputs)
        return self._inputs[index]

    def get_output_tensor(self, index: int):
        return self._outputs[index]

    def run(self):
        inputs = Dictionary[_nncase.IR.Var, _torch.Tensor]()
        for k, v in zip(self._module.params, self._inputs):
            inputs[k] = v.to_nncase()
        results = _nncase.Evaluator.Evaluator.Eval(self._module.entry, inputs)
        self._outputs = list([RuntimeTensor(res) for res in results])

    @ property
    def outputs_size(self) -> int:
        return len(self._outputs)


class Module():
    _module: _nncase.IR.Module = None

    def __init__(self, module: _nncase.IR.Module):
        assert module.Entry != None
        self._module = module

    @ property
    def entry(self) -> _nncase.IR.Function:
        return self._module.Entry

    @ property
    def params(self) -> List[_nncase.IR.Var]:
        return list([v for v in self.entry.Parameters])


class Compiler:
    _module: Module = None
    _compile_options: CliCompileOptions = None
    _compiler: _nncase.Cli.Commands.Compile

    def __init__(self, compile_options: CompileOptions) -> None:
        self.__process_compile_options(compile_options)
        self._compiler = _nncase.Cli.Commands.Compile()

    def __process_compile_options(self, compile_options: CompileOptions) -> CliCompileOptions:
        self._compile_options: CliCompileOptions = _nncase.Cli.Commands.CompileOptions()
        self._compile_options.DumpLevel = 3 if compile_options.dump_ir == True else 0
        self._compile_options.DumpDir = compile_options.dump_dir

    def compile(self) -> None:
        pass

    def create_evaluator(self, stage: int) -> GraphEvaluator:
        return GraphEvaluator(self._module)

    def gencode(self, stream: BytesIO) -> None:
        pass

    def gencode_tobytes(self) -> bytes:
        pass

    def import_caffe(self, model: bytes, prototxt: bytes) -> None:
        raise NotImplementedError("import_caffe")

    def import_onnx(self, model: bytes, options: ImportOptions) -> None:
        raise NotImplementedError("import_onnx")

    def import_tflite(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.InputFormat = "tflite"
        self._module = Module(self._compiler.ImportModule(
            MemoryStream(model_content), self._compile_options))

    def use_ptq(self, ptq_dataset_options: PTQTensorOptions) -> None:
        raise NotImplementedError("use_ptq")

    def dump_range_options(self) -> DumpRangeTensorOptions:
        raise NotImplementedError("dump_range_options")


def test_target(target: str):
    return target == 'cpu'


class DumpRangeTensorOptions:
    calibrate_method: str
    samples_count: int

    def set_tensor_data(self, data: bytes):
        pass


class CliCompileOptions():
    InputFile: str
    InputFormat: str
    Target: str
    DumpLevel: int
    DumpDir: str


class CompileOptions:
    benchmark_only: bool
    dump_asm: bool
    dump_dir: str
    dump_ir: bool
    swapRB: bool
    input_range: List[float]
    input_shape: List[int]
    input_type: str
    is_fpga: bool
    mean: List[float]
    std: List[float]
    output_type: str
    preprocess: bool
    quant_type: str
    target: str
    w_quant_type: str
    use_mse_quant_w: bool
    input_layout: str
    output_layout: str
    letterbox_value: float
    tcu_num: int

    def __init__(self) -> None:
        pass
