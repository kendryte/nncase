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

from onnx.onnx_ml_pb2 import ModelProto
import pytest
import os
import torch
import torchvision as tv
import numpy as np
import sys
import nncase
import requests
import onnxsim
import onnx
from io import BytesIO

TEMP_DIR = "tmp"
MODEL_DIR = "models"

MODELS = {
    "mnist": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx",
        "in_shapes": {"Input3": [1, 1, 28, 28]}
    },
    "mobilenet_v2": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "in_shapes": {"data": [1, 3, 224, 224]}
    }
}


def _download(url, name, in_shapes):
    filename = os.path.join(MODEL_DIR, "source", name + ".onnx")
    if not os.path.exists(filename):
        req = requests.get(url)
        onnx_model, check = onnxsim.simplify(
            onnx.load_model(BytesIO(req.content)), check_n=3, input_shapes=in_shapes)
        # assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_model, filename)

    with open(filename, "rb") as file:
        return file.read()


def _make_module(name, target):
    model = MODELS[name]
    url = model["url"]
    onnx_model = _download(url, name, model["in_shapes"])

    # import
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_layout = "NCHW"
    compile_options.output_layout = "NCHW"
    compile_options.dump_dir = os.path.join(TEMP_DIR, name)
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_quant_error = False
    compile_options.dump_import_op_range = False
    compile_options.use_mse_quant_w = True
    compile_options.split_w_to_act = False
    compile_options.benchmark_only = True
    compiler = nncase.Compiler(compile_options)
    import_options = nncase.ImportOptions()
    compiler.import_onnx(onnx_model, import_options)

    # compile
    compiler.compile()
    kmodel = compiler.gencode_tobytes()
    with open(os.path.join(MODEL_DIR, target, name + ".kmodel"), 'wb') as f:
        f.write(kmodel)


def _make_cpu_models():
    target = "cpu"
    model_names = [
        "mnist",
        "mobilenet_v2"
    ]

    for model_name in model_names:
        _make_module(model_name, target)


if __name__ == "__main__":
    _make_cpu_models()
