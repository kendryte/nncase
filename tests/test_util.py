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
    tf.saved_model.save(module, pb_export_dir)
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


def gen_input(case_name, input_tensor, id):
    input_dir = os.path.join(output_root, case_name)
    if input_tensor['dtype'] is np.uint8:
        input_data = np.random.randint(0, 256, input_tensor['shape'])
    elif input_tensor['dtype'] is np.int8:
        input_data = np.random.randint(-128, 128, input_tensor['shape'])
    else:
        input_data = np.random.rand(*input_tensor['shape']) * 2 - 1
    input_data = input_data.astype(dtype=input_tensor['dtype'])
    input_data.tofile(os.path.join(input_dir, 'input{0}.bin'.format(id)))
    save_numpy_array_as_txt(os.path.join(
        input_dir, 'input{0}.txt'.format(id)), input_data)
    return input_data


def gen_calib_data(case_name, input_tensor, id, n):
    input_dir = os.path.join(output_root, case_name)
    input_tensor['shape'][0] *= n
    if input_tensor['dtype'] is np.uint8:
        input_data = np.random.randint(0, 256, input_tensor['shape'])
    elif input_tensor['dtype'] is np.int8:
        input_data = np.random.randint(-128, 128, input_tensor['shape'])
    else:
        input_data = np.random.rand(*input_tensor['shape']) * 2 - 1
    input_data = input_data.astype(dtype=input_tensor['dtype'])
    input_data.tofile(os.path.join(input_dir, 'calib_data{0}.bin'.format(id)))
    save_numpy_array_as_txt(os.path.join(
        input_dir, 'calib_data{0}.txt'.format(id)), input_data)
    return input_data


def eval_tflite_gth(case_name, tflite, n):
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    input_len = len(interp.get_input_details())
    inputs = []
    calib_datas = []
    for i in range(input_len):
        input_tensor = interp.get_input_details()[i]
        input_id = input_tensor["index"]
        input = gen_input(case_name, input_tensor, i)
        calib_data = gen_calib_data(case_name, input_tensor, i, n)
        interp.set_tensor(input_id, input)
        inputs.append(input)
        calib_datas.append(calib_data)
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
    return out_len, inputs, calib_datas


def graph_eval_tflite_nncase(case_name, model, targets, inputs, enable_ptq):
    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.dump_asm = True
    compile_options.dump_ir = True
    for target in targets:
        case_dir = os.path.join(output_root, case_name,
                                target, 'eval', 'ptq' if enable_ptq else 'no_ptq')
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        compile_options.target = target
        compile_options.dump_dir = case_dir
        compiler = nncase.Compiler(compile_options)
        compiler.import_tflite(model, import_options)
        evaluator = compiler.create_evaluator(3)
        for i in range(len(inputs)):
            input_tensor = nncase.RuntimeTensor.from_numpy(inputs[i])
            input_tensor.copy_to(evaluator.get_input_tensor(i))
        evaluator.run()
        for i in range(evaluator.outputs_size):
            result = evaluator.get_output_tensor(i).to_numpy()
            result.tofile(os.path.join(
                case_dir, 'nncase_result{0}.bin'.format(i)))
            save_numpy_array_as_txt(os.path.join(
                case_dir, 'nncase_result{0}.txt'.format(i)), result)


def compile_tflite_nncase(case_name, model, targets, inputs, n, enable_ptq):
    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.dump_asm = True
    compile_options.dump_ir = True
    for target in targets:
        kmodel_dir = os.path.join(
            output_root, case_name, target, 'infer', 'ptq' if enable_ptq else 'no_ptq')
        if not os.path.exists(kmodel_dir):
            os.makedirs(kmodel_dir)
        compile_options.target = target
        compile_options.dump_dir = kmodel_dir
        compiler = nncase.Compiler(compile_options)
        compiler.import_tflite(model, import_options)
        if enable_ptq:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(inputs[0].tobytes())
            ptq_options.samples_count = n
            compiler.use_ptq(ptq_options)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(os.path.join(kmodel_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)


def infer_nncase(case_name, inputs, targets, enable_ptq):
    for target in targets:
        case_dir = os.path.join(output_root, case_name,
                                target, 'infer', 'ptq' if enable_ptq else 'no_ptq')
        with open(os.path.join(case_dir, 'test.kmodel'), 'rb') as f:
            kmodel = f.read()
            sim = nncase.Simulator()
            sim.load_model(kmodel)
            for i in range(len(inputs)):
                sim.set_input_tensor(
                    i, nncase.RuntimeTensor.from_numpy(inputs[i]))
            sim.run()
            for i in range(sim.outputs_size):
                result = sim.get_output_tensor(i).to_numpy()
                result.tofile(os.path.join(
                    case_dir, 'nncase_result{0}.bin'.format(i)))
                save_numpy_array_as_txt(os.path.join(
                    case_dir, 'nncase_result{0}.txt'.format(i)), result)


def validate_targets(targets):
    new_targets = []
    for t in targets:
        if nncase.test_target(t):
            new_targets.append(t)
        else:
            print("WARN: target[{0}] not found".format(t))
    return new_targets


def test_tf_module(case_name, module, targets):
    targets = validate_targets(targets)
    case_name = case_name.replace('[', '_').replace(']', '_')
    case_dir = os.path.join('./tmp', case_name)
    clear(case_dir)
    n = 10
    tflite = tf_module_to_tflite(case_name, module)
    out_len, inputs, calib_datas = eval_tflite_gth(case_name, tflite, n)

    # evaluation
    graph_eval_tflite_nncase(case_name, tflite, targets, inputs, False)
    ret = compare_util.compare_results(
        case_dir, out_len, targets, enable_ptq=False, is_evaluation=True)
    assert ret

    # compile & infer
    for enable_ptq in [False, True]:
        if len(inputs) > 1 and enable_ptq:
            continue
        compile_tflite_nncase(case_name, tflite, targets,
                              calib_datas, n, enable_ptq=enable_ptq)
        infer_nncase(case_name, inputs, targets, enable_ptq=enable_ptq)
        ret = compare_util.compare_results(
            case_dir, out_len, targets, enable_ptq=enable_ptq, is_evaluation=False)
        assert ret


def clear(case_dir):
    in_ci = os.getenv('CI', False)
    if in_ci:
        if os.path.exists(output_root):
            shutil.rmtree(output_root)
    else:
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
    os.makedirs(case_dir)
