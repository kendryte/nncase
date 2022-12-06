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

import re
import subprocess
import shutil
import os
import sys
from pathlib import Path
from shutil import which
import platform
import pythonnet
from enum import Enum

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    if retval != 0:
        raise Exception(f"exit code: {retval} \n in command:\n{cmd} \n error msg:{p.stdout.readlines()}")
    lines = p.stdout.read().splitlines()
    return lines

def find_output(lines, string):
    infos = [str(line)[:-1] for line in lines if str(line).find(string) != -1]
    return infos

def check_env():
    env = os.environ
    errors = []
    if not "NNCASE_CLI" in env:
        errors.append("NNCASE_CLI not found")
    if not "PYTHONPATH" in env:
        errors.append("PYTHONPATH not found")
    # if not "PYTHONNET_PYDLL" in env:
    #     errors.append("PYTHONNET_PYDLL not found")
    return errors

init_pynet = True
if init_pynet:
    errors = check_env()
    if len(errors) > 0:
        raise Exception("check failed:\n" + {"\n".join(errors)})
    pythonnet.load("coreclr", runtime_config=os.path.join(os.environ.get('NNCASE_CLI'), 'Nncase.Cli.runtimeconfig.json'))

import clr
import sys
import os
import ctypes

import numpy
from numpy import empty
import subprocess



def _add_dllpath():
    nncase_cli_path = os.getenv("NNCASE_CLI")

    clr.AddReference("System.IO")
    clr.AddReference("System.Collections")
    for dll in ["Nncase.Cli",
                "Nncase.Core",
                # "Nncase.EGraph",
                "Nncase.Graph",
                "Nncase.Evaluator",
                "Nncase.Importer",
                "Nncase.Simulator",
                "Nncase.Compiler"]:
        clr.AddReference(os.path.join(nncase_cli_path, dll))

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


class ImportOptions:
    def __init__(self) -> None:
        pass


class PTQTensorOptions:
    calibrate_method: str
    input_mean: float
    input_std: float
    samples_count: int
    cali_data: list

    def __init__(self) -> None:
        pass

    def set_tensor_data(self, data: np.array) -> None:
        self.cali_data = [RuntimeTensor(d) for d in data]

#_nncase.Compiler.PythonHelper.LaunchDebugger()

class RuntimeTensor:
    _arr: np.ndarray
    netTypeMap = {
        np.int8: SByte,
        np.int16: Int16,
        np.int32: Int32,
        np.int64: Int64,
        np.uint8: Byte,
        # np.float16: Float16,
        np.float32: Single,
        np.float64: Double,
    }
    
    npToDataTypeMap = {
        np.bool8: _nncase.DataTypes.Boolean,
        np.int8: _nncase.DataTypes.Int8,
        np.int16: _nncase.DataTypes.Int16,
        np.int32: _nncase.DataTypes.Int32,
        np.int64: _nncase.DataTypes.Int64,
        np.uint8: _nncase.DataTypes.UInt8,
        np.float16: _nncase.DataTypes.Float16,
        np.float32: _nncase.DataTypes.Float32,
        np.float64: _nncase.DataTypes.Float64,
    }

    toNpDataTypeMap = {v:k for k, v in npToDataTypeMap.items()}
    
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def copy_to(self, to: RuntimeTensor) -> None:
        to._arr = self._arr

    @staticmethod
    def from_numpy(arr: np.ndarray) -> RuntimeTensor:
        return RuntimeTensor(arr)

    def to_numpy(self) -> np.ndarray:
        if isinstance(self._arr, np.ndarray):
            return self._arr
        else:
            tensor_const = _nncase.IR.TensorConst.FromTensor(self._arr)
            arr_bytes = _nncase.Compiler.PythonHelper.BytesBufferFromTensor(tensor_const.Value)
            shape = tensor_const.Value.Shape.ToValueArray()
            return np.frombuffer(arr_bytes, dtype=self.toNpDataTypeMap[tensor_const.Value.ElementType]).reshape(shape)
        
    def to_nncase_value(self) -> _nncase.IValue:
        dtype = self.npToDataTypeMap[self._arr.dtype.type]
        return _nncase.Compiler.PythonHelper.TensorValueFromBytes(dtype, self._arr.tobytes(), list(self._arr.shape))

    def to_nncase_tensor(self) -> _nncase.Tensor:
        dtype = self.npToDataTypeMap[self._arr.dtype.type]
        return _nncase.Compiler.PythonHelper.TensorFromBytes(dtype, self._arr.tobytes(), list(self._arr.shape))

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

class RTTensor:
    rtTypeToNp = {
        _nncase.Runtime.TypeCode.Boolean: np.bool8,
        _nncase.Runtime.TypeCode.Int8: np.int8,
        _nncase.Runtime.TypeCode.Int16: np.int16,
        _nncase.Runtime.TypeCode.Int32: np.int32,
        _nncase.Runtime.TypeCode.Int64: np.int64,
        _nncase.Runtime.TypeCode.UInt8: np.uint8,
        _nncase.Runtime.TypeCode.Float16: np.float16,
        _nncase.Runtime.TypeCode.Float32: np.float32,
        _nncase.Runtime.TypeCode.Float64: np.float64,
    }
    
    def __init__(self, rt_tensor) -> None:
        self.rt_tensor = rt_tensor
        
    def from_tensor(tensor) -> RTTensor:
        t = tensor.to_nncase_tensor()
        rt_tensor = _nncase.Runtime.Interop.RTTensor.FromTensor(t)
        return RTTensor(rt_tensor)
    
    def to_numpy(self):
        arr_bytes = _nncase.Compiler.PythonHelper.GetRTTensorBytes(self.rt_tensor)
        shape = _nncase.Compiler.PythonHelper.GetRTTensorDims(self.rt_tensor)
        return np.frombuffer(arr_bytes, dtype=self.rtTypeToNp[self.rt_tensor.ElementType.TypeCode]).reshape(shape)

    def to_nncase(self):
        return self.rt_tensor

class Simulator:
    def __init__(self) -> None:
        self.interpreter = _nncase.Runtime.Interop.RTInterpreter.Create()
        self.inputs = []
        self.outputs = []

    def get_input_desc(self, index: int) -> MemoryRange:
        pass

    def get_input_tensor(self, index: int) -> RuntimeTensor:
        pass

    def get_output_desc(self, index: int) -> MemoryRange:
        pass

    def get_output_tensor(self, index: int):
        rt_tensor = self.outputs[index]
        return rt_tensor.to_numpy()

    def load_model(self, model: bytes) -> None:
        model_bytes = [i for i in model]
        mem = _nncase.Compiler.PythonHelper.ToMemory(model_bytes)
        self.interpreter.LoadModel(mem)

    def run(self) -> None:
        outputs = _nncase.Compiler.PythonHelper.RunSimulator(self.interpreter, self.all_nncase_input())
        self.outputs = list(map(RTTensor, outputs))

    def all_nncase_input(self):
        return list(map(lambda x: x.to_nncase(), self.inputs))

    def all_numpy_output(self):
        return map(lambda x: x.to_numpy(), self.outputs)

    def add_input_tensor(self, tensor: RuntimeTensor) -> None:
        rt_tensor = RTTensor.from_tensor(tensor)
        self.inputs.append(rt_tensor)

    def set_output_tensor(self, index: int, tensor: RuntimeTensor) -> None:
        pass

    @ property
    def inputs_size(self) -> int:
        pass

    @ property
    def outputs_size(self) -> int:
        return len(self.outputs)


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
        inputs = Dictionary[_nncase.IR.Var, _nncase.IValue]()
        for k, v in zip(self._module.params, self._inputs):
            inputs[k] = v.to_nncase_value()
        results: _nncase.IValue = _nncase.Compiler.PythonHelper.Evaluate(self._module.entry.Body, inputs).AsTensors()
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
    _compile_options: ClCompileOptions = None
    _quant_options: ClQuantizeOptions = None
    _compiler: _nncase.Compiler.Compiler

    def __init__(self) -> None:
        self._compiler = _nncase.Compiler.Compiler()
        self._compiler.Init()

    def set_compile_options(self, compile_options: CompileOptions):
        # format it.
        if (self._compile_options is None):
            self._compile_options: ClCompileOptions = _nncase.CompileOptions()
            self._compiler.UpdateCompileOptions(self._compile_options)
        self._compile_options.Target = compile_options.target
        self._compile_options.DumpLevel = 3 if compile_options.dump_ir == True else 0
        self._compile_options.DumpDir = compile_options.dump_dir
        self._compile_options.QuantizeOptions = _nncase.Quantization.QuantizeOptions()
        self._quant_options = self._compile_options.QuantizeOptions
        # update the CompilerService global compile options

    def compile(self) -> None:
        self._compiler.Compile()

    def create_evaluator(self, stage: int) -> GraphEvaluator:
        return GraphEvaluator(self._module)

    def gencode(self, stream: BytesIO) -> None:
        pass

    def gencode_tobytes(self) -> bytes:
        code = self._compiler.Gencode()
        arr = []
        for i in range(0, code.Length):
            arr.append(code[i])
        return bytes(arr)

    def import_caffe(self, model: bytes, prototxt: bytes) -> None:
        raise NotImplementedError("import_caffe")

    def import_onnx(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.InputFormat = "onnx"
        self._module = Module(self._compiler.ImportModule(
            MemoryStream(model_content)))
        
    def import_tflite(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.InputFormat = "tflite"
        self._module = Module(self._compiler.ImportModule(
            MemoryStream(model_content)))

    def use_ptq(self, ptq_dataset_options: PTQTensorOptions, params: list) -> None:
        dataset = [data.to_nncase_tensor() for data in ptq_dataset_options.cali_data]
        dataset = _nncase.Compiler.PythonHelper.MakeDatasetProvider(dataset, ptq_dataset_options.samples_count, params)
        self._quant_options.CalibrationDataset = dataset
        self._quant_options.CalibrationMethod = _nncase.Quantization.CalibMethod.NoClip
        self._compile_options.ModelQuantMode = _nncase.Quantization.ModelQuantMode.UsePTQ

    def dump_range_options(self) -> DumpRangeTensorOptions:
        raise NotImplementedError("dump_range_options")

def check_target(target: str):
    def test_target(target: str):
        return target in ["cpu", "k510", "k230"]

    def target_exist(target: str):
        return _nncase.Compiler.PythonHelper.TargetExist(target)

    return test_target(target) and target_exist(target)

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
  UseAdaRound  : bool 

class ClCompileOptions():
    InputFile: str
    InputFormat: str
    Target: str
    DumpLevel: int
    DumpDir: str
    QuantType: int
    QuantMode: int
    OutputFile: str
    ModelQuantMode: int
    QuantizeOptions: ClQuantizeOptions


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
