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
"""System test: test demo"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
from tflite_test_runner import TfliteTestRunner
from onnx_test_runner import OnnxTestRunner
import os


def gen_modle_list():

    model_file_list = list()
    g = os.walk(
        r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/conv_act1_fuse_base_case/")
    # g = os.walk(r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/conv_base_case/")
    # g = os.walk(
    #     r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/conv_pdp_fuse_base_case/")
    # g = os.walk(r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/group_conv_base_case/")
    # g = os.walk(
    #     r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/transposed_conv_base_case/")
    # g = os.walk(
    #     r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/dilated_conv_base_case/")
    # g = os.walk(
    #     r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/conv_dw_fuse_base_case/")
    # g = os.walk(r"/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/mini-test/matmul_base_case/")

    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            model_file = os.path.join(path, dir_name + "/model_f32.tflite")
            # model_file = os.path.join(path, dir_name + "/test.onnx")
            model_file_list.append(model_file)

    return model_file_list


def test_debug(request):

    # runner = TfliteTestRunner(request.node.name)
    # model_file = '/home/guodongliang/Compiler/NewArch/k230-gnne-compiler-tests/benchmark-test/mobilenetv2/tflite/model_f32.tflite'

    # override_config = """
    # [compile_opt]
    # shape_bucket_fix_var_map = { "batch_size"=1 }
    # """

    runner = OnnxTestRunner(request.node.name)
    # runner.set_shape_var({"seq_len": 10, "history_len": 0})
    # model_file = '/compiler/something/onnx_smooth/llm.onnx'
    # model_file = '/compiler/something/onnx/llm.onnx'
    # model_file = '/compiler/something/onnx_layer_1/llm.onnx'
    # model_file = '/data/models/deepseek/DeepSeek-R1-Distill-Qwen-1.5B/onnx/llm.onnx'
    model_file = '/data/models/qwen/Qwen3-0.6B/onnx/llm.onnx'
    runner.set_shape_var({"seq_len": 14, "history_len": 0})
    runner.run(model_file)

    # runner.set_shape_var({"batch_size": 1})
    # runner.run(model_file)


# @pytest.mark.parametrize('model_name', gen_modle_list())
# def test_debug(model_name, request):
#     runner = TfliteTestRunner(request.node.name)
#     # runner = OnnxTestRunner(request.node.name, ['k230'])
#     runner.run(model_name)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_debug.py'])
