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
"""Test utility"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import subprocess
import tensorflow as tf
import torch
import numpy as np
import shutil
import struct
import nncase
import compare_util

process_num = 16
output_root = './tmp'


def save_numpy_array_as_txt(save_path, value_np, bit_16_represent=False):
    if bit_16_represent:
        np.save(save_path, _cast_bfloat16_then_float32(value_np))
    else:
        with open(save_path, 'w') as f:
            shape_info = "shape: (" + ",".join(str(dim)
                                               for dim in value_np.shape) + ")\n"
            f.write(shape_info)

            for val in value_np.reshape([-1]):
                f.write("%f\n" % val)
    print("----> %s" % save_path)


def _cast_bfloat16_then_float32(values: np.array):
    shape = values.shape
    values = values.reshape([-1])
    for i, value in enumerate(values):
        value = float(value)
        packed = struct.pack('!f', value)
        integers = [c for c in packed][:2] + [0, 0]
        value = struct.unpack('!f', bytes(integers))[0]
        values[i] = value

    values = values.reshape(shape)
    return values


def tf_module_to_tflite(case_name, module):
    pb_export_dir = os.path.join(output_root, case_name)
    tf.saved_model.save(module, pb_export_dir, module.__call__)
    converter = tf.lite.TFLiteConverter.from_saved_model(pb_export_dir)
    tflite_model = converter.convert()
    tflite_export_file = os.path.join(output_root, case_name, 'test.tflite')
    with open(tflite_export_file, 'wb') as f:
        f.write(tflite_model)
    return tflite_model


def torch_module_to_onnx(case_name, module, in_shape, opset_version=11):
    case_dir = os.path.join('./tmp', case_name)
    clear(case_dir)
    dummy_input = torch.randn(*in_shape)
    onnx_export_file = os.path.join(output_root, case_name, 'test.onnx')
    torch.onnx.export(module, dummy_input, onnx_export_file,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=opset_version)


def gen_input(case_name, input_tensor):
    input_dir = os.path.join(output_root, case_name)
    if input_tensor['dtype'] is np.uint8:
        input_data = np.random.randint(0, 256, input_tensor['shape'])
    elif input_tensor['dtype'] is np.int8:
        input_data = np.random.randint(-128, 128, input_tensor['shape'])
    else:
        input_data = np.random.rand(*input_tensor['shape'])
    input_data = input_data.astype(dtype=input_tensor['dtype'])
    input_data.tofile(os.path.join(input_dir, 'input.bin'))
    save_numpy_array_as_txt(os.path.join(input_dir, 'input.txt'), input_data)
    return input_data


def eval_tflite_gth(case_name, tflite):
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    input_tensor = interp.get_input_details()[0]
    input_id = input_tensor["index"]
    input = gen_input(case_name, input_tensor)
    interp.set_tensor(input_id, input)
    interp.invoke()

    out_len = len(interp.get_output_details())
    for i in range(out_len):
        output_id = interp.get_output_details()[i]["index"]

        result = interp.get_tensor(output_id)
        # if len(result.shape) == 4:
        #    result = np.transpose(result, [0, 3, 1, 2])
        result.tofile(os.path.join(output_root, case_name,
                                   'cpu_result{0}.bin'.format(i)))
        save_numpy_array_as_txt(os.path.join(
            output_root, case_name, 'cpu_result{0}.txt'.format(i)), result)
    return out_len, input


def compile_tflite_nncase(case_name, model, targets, input, enable_ptq):
    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.dump_asm = True
    compile_options.dump_ir = True
    for target in targets:
        kmodel_dir = os.path.join(
            output_root, case_name, target, 'ptq' if enable_ptq else 'no_ptq')
        if not os.path.exists(kmodel_dir):
            os.makedirs(kmodel_dir)
        compile_options.target = target
        compile_options.dump_dir = kmodel_dir
        compiler = nncase.Compiler(compile_options)
        compiler.import_tflite(model, import_options)
        if enable_ptq:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(input.tobytes())
            ptq_options.samples_count = 1
            compiler.use_ptq(ptq_options)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(os.path.join(kmodel_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)


def eval_nncase(case_name, input, targets):
    for target in targets:
        case_dir = os.path.join(output_root, case_name, target)
        with open(os.path.join(case_dir, 'test.kmodel'), 'rb') as f:
            kmodel = f.read()
            sim = nncase.Simulator()
            sim.load_model(kmodel)
            sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input))
            sim.run()
            for i in range(sim.outputs_size):
                result = sim.get_output_tensor(i).to_numpy()
                result.tofile(os.path.join(
                    case_dir, 'nncase_result{0}.bin'.format(i)))
                save_numpy_array_as_txt(os.path.join(
                    case_dir, 'nncase_result{0}.txt'.format(i)), result)


def test_tf_module(case_name, module, targets):
    case_dir = os.path.join('./tmp', case_name)
    clear(case_dir)
    tflite = tf_module_to_tflite(case_name, module)
    out_len, input = eval_tflite_gth(case_name, tflite)
    compile_tflite_nncase(case_name, tflite, targets, input, enable_ptq=False)
    compile_tflite_nncase(case_name, tflite, targets, input, enable_ptq=True)
    eval_nncase(case_name, input, targets)
    ret = compare_util.compare_results(case_dir, out_len, targets)
    assert ret


def clear(case_dir):
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
