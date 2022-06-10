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
from shutil import which

def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    if retval != 0:
        raise Exception(f"exit code: {retval} \n in command:\n{cmd}")
    lines = p.stdout.read().splitlines()
    return lines

def check_dotnet():
    if which("dotnet") is None:
        return ["dotnet not found"]
    return []

def find_output(lines, string):
    infos = [str(line)[:-1] for line in lines if str(line).find(string) != -1]
    return infos

def get_dotnet_runtime_version():
    lines = run_cmd('dotnet --info')
    infos = find_output(lines, "Microsoft.AspNetCore.App")
    version = re.search(r"([1-9]\d|[1-9])(\.([1-9]\d|\d)){2}", infos[0])
    return version.group(0)

def get_pynet_init_file():
    pip_cmd = ""
    if which("pip") is not None:
        pip_cmd = "pip"
    elif which("pip3") is not None:
        pip_cmd = "pip3"
    else:
        raise "pip not found"
    lines = run_cmd(f"{pip_cmd} show pythonnet")
    location = find_output(lines, "Location")
    if location[0].find("not found"):
        run_cmd(f"{pip_cmd} install --pre pythonnet")
    # todo:check pythonnet version
    pn_root = location[0].split(': ')[1]
    if not os.path.exists(os.path.join(pn_root, '__init__.py')):
        pn_root = os.path.join(pn_root, 'pythonnet')
        if not os.path.exists(os.path.join(pn_root, '__init__.py')):
            raise Exception('pythonnet root path search failed')
    return pn_root

def generate_runtime_config(config_path, version):
    config = '''{
        "runtimeOptions": {
            "tfm": "net6.0",
            "framework": {
                "name": "Microsoft.NETCore.App",
                "version": "''' + version + '''"
            }
        }
    }'''
    with open(config_path, "w") as f:
        f.write(config)
    return config_path

def create_runtime_config(pn_root, version):
    config_file = 'runtime_config.json'
    config_path = os.path.join(pn_root, config_file)
    if os.path.exists(config_path):
        return config_path
    return generate_runtime_config(config_path, version)

def check_env():
    env = os.environ
    errors = []
    if not "NNCASE_CLI" in env:
        errors.append("NNCASE_CLI not found")
    if not "PYTHONPATH" in env:
        errors.append("PYTHONPATH not found")
    if not "PYTHONNET_PYDLL" in env:
        errors.append("PYTHONNET_PYDLL not found")
    return errors

def create_new_init_content(init_path, config_path):
    with open(init_path) as init:
        lines = init.readlines()
        i = next(i for i, line in enumerate(lines) if line.find("def set_default_runtime() -> None:") != -1)
        if lines[i+1].find(config_path) != -1:
            return
        set = f"""    set_runtime(clr_loader.get_coreclr(\"{config_path}\"))\n"""
        lines.insert(i + 1, set)
        for i in range(i + 2, i + 2 + 4):
            lines[i] = "#" + lines[i]
        init_content = "".join(lines)
        return init_content

def modify_pynet(pn_root):
    config_path = create_runtime_config(pn_root, version)
    init_path = os.path.join(pn_root, '__init__.py')
    new_content = create_new_init_content(init_path, config_path)
    if new_content is None:
        return
    with open(init_path, "w") as f:
        f.write(new_content)

init_pynet = True
if init_pynet:
    errors = check_dotnet()
    errors += check_env()
    if len(errors) > 0:
        raise Exception("check failed:\n" + {"\n".join(errors)})
    version = get_dotnet_runtime_version()
    pn_root = get_pynet_init_file()
    modify_pynet(pn_root)


import clr
import sys
import os

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
from System import Memory
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

    def __init__(self) -> None:
        pass

    def set_tensor_data(self, data: bytes) -> None:
        pass


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
        self.interpreter = _nncase.Runtime.Interop.RTInterpreter()
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
        results: _nncase.IValue = _nncase.CompilerServices.Evaluate(self._module.entry.Body, inputs).AsTensors()
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
    _compiler: _nncase.Compiler.Compiler

    def __init__(self, compile_options: CompileOptions) -> None:
        self.__process_compile_options(compile_options)
        self._compiler = _nncase.Compiler.Compiler()
        self._compiler.init()

    def __process_compile_options(self, compile_options: CompileOptions) -> CliCompileOptions:
        self._compile_options: CliCompileOptions = _nncase.CompileOptions(False)
        self._compile_options.Target = compile_options.target
        self._compile_options.DumpLevel = 3 if compile_options.dump_ir == True else 0
        self._compile_options.DumpDir = compile_options.dump_dir

    def compile(self) -> None:
        self._compiler.Compile(self._compile_options)

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
            MemoryStream(model_content), self._compile_options))
        
    def import_tflite(self, model_content: bytes, options: ImportOptions) -> None:
        self._compile_options.InputFormat = "tflite"
        self._module = Module(self._compiler.ImportModule(
            MemoryStream(model_content), self._compile_options))

    def use_ptq(self, ptq_dataset_options: PTQTensorOptions) -> None:
        raise NotImplementedError("use_ptq")

    def dump_range_options(self) -> DumpRangeTensorOptions:
        raise NotImplementedError("dump_range_options")


def test_target(target: str):
    return target in ["cpu", "k510"]


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
    UsePTQ: bool
    QuantType: int
    QuantMode: int


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
