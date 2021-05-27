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
import onnx
from onnx import version_converter, helper
import onnxsim
import onnxoptimizer
import onnxruntime as ort

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
    return onnx_export_file

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

def map_onnx_to_numpy_type(onnx_type):
    ONNX_TO_NUMPY_DTYPE = {
        onnx.onnx_pb.TensorProto.FLOAT: np.float32,
        onnx.onnx_pb.TensorProto.FLOAT16: np.float16,
        onnx.onnx_pb.TensorProto.DOUBLE: np.float64,
        onnx.onnx_pb.TensorProto.INT32: np.int32,
        onnx.onnx_pb.TensorProto.INT16: np.int16,
        onnx.onnx_pb.TensorProto.INT8: np.int8,
        onnx.onnx_pb.TensorProto.UINT8: np.uint8,
        onnx.onnx_pb.TensorProto.UINT16: np.uint16,
        onnx.onnx_pb.TensorProto.INT64: np.int64,
        onnx.onnx_pb.TensorProto.UINT64: np.uint64,
        onnx.onnx_pb.TensorProto.BOOL: np.bool,
        onnx.onnx_pb.TensorProto.COMPLEX64: np.complex64,
        onnx.onnx_pb.TensorProto.COMPLEX128: np.complex128,
        onnx.onnx_pb.TensorProto.STRING: np.object,
    }

    return ONNX_TO_NUMPY_DTYPE[onnx_type]

def get_onnx_shape_dtype_inputs(input_tensors):
    inputs = []
    dtype_dict = {}
    shape_dict = {}

    for _, e in enumerate(input_tensors):
        inputs.append(e.name)
        onnx_type = e.type.tensor_type
        dtype_dict[e.name] = map_onnx_to_numpy_type(onnx_type.elem_type)
        shape_dict[e.name] = [(i.dim_value if i.dim_value!=0 else d) for i,d in zip(onnx_type.shape.dim,[1,3,224,224])]
    return inputs, dtype_dict, shape_dict

def preprocess_onnx_model(onnx_model, fix_bn=True, convert_version=True, simplify=True, import_test=True):
    args = {'fix_bn':fix_bn, 'convert_version':convert_version, 'simplify':simplify, 'import_test':import_test}
    try:
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        input_names = list(set(input_all) - set(input_initializer))
        input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]
        inputs, dtype_dict, shape_dict = get_onnx_shape_dtype_inputs(input_tensors)
        print('[info]: inputs:', inputs)
        print('[info]: dtype dict:', dtype_dict)
        print('[info]: shape dict:', shape_dict)

        if fix_bn:
            # fix https://github.com/onnx/models/issues/242
            for node in onnx_model.graph.node:
                if(node.op_type == "BatchNormalization"):
                    for attr in node.attribute:
                        if (attr.name == "spatial"):
                            attr.i = 1

        if convert_version:
            curret_version = onnx_model.opset_import[0].version
            for i in range(curret_version, 8):
                onnx_model = version_converter.convert_version(onnx_model, i+1)

        if simplify:
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            onnx_model, check = onnxsim.simplify(onnx_model, check_n=0, input_shapes=shape_dict, perform_optimization=False, skip_fuse_bn=True, skip_shape_inference=True, skipped_optimizers= [
                # 'eliminate_deadend',
                # 'eliminate_nop_dropout',
                # 'eliminate_nop_cast',
                # 'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                # 'extract_constant_to_initializer', 'eliminate_unused_initializer',
                # 'eliminate_nop_transpose',
                # 'eliminate_nop_flatten', 'eliminate_identity',
                'fuse_add_bias_into_conv',
                # 'fuse_consecutive_concats',
                # 'fuse_consecutive_log_softmax',
                # 'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
                # 'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
                # 'fuse_pad_into_conv', 'fuse_transpose_into_gemm', 'eliminate_duplicate_initializer'
            ])
            assert check, "Simplified ONNX model could not be validated"

        print('[info]: preprocess ONNX model success: ', args)
        return onnx_model
    except Exception as e:
        print('[info]: preprocess ONNX model failed: ', args)
        print(e)
        traceback.print_exc()
        return None

def gen_input(case_name, inputs, dtype_dict, shape_dict):
    input_dir = os.path.join(output_root, case_name)
    input_dict = {}
    for name in inputs:
        # rename the name to handle dir not found bug.
        rename = name.replace("/", "_")
        if dtype_dict[name] == 'int8':
            data = np.random.randint(-128, 128, shape_dict[name])
        elif dtype_dict[name] == 'uint8':
            data = np.random.randint(0, 256, shape_dict[name])
        else:
            data = np.random.rand(*shape_dict[name])
        data = data.astype(dtype=dtype_dict[name])
        data.tofile(os.path.join(input_dir, 'input_{0}.bin'.format(rename)))
        save_numpy_array_as_txt(os.path.join(input_dir, 'input_{0}.txt'.format(rename)), data)
        input_dict[name] = data
    return input_dict

def gen_calib_data(case_name, inputs, dtype_dict, shape_dict, n):
    input_dir = os.path.join(output_root, case_name)
    calib_datas = []
    for input in inputs:
        dtype = dtype_dict[input]
        shape = shape_dict[input]
        shape[0] *= n
        if dtype is np.uint8:
            calib_data = np.random.randint(0, 256, shape)
        elif dtype is np.int8:
            calib_data = np.random.randint(-128, 128, shape)
        else:
            calib_data = np.random.rand(*shape) * 2 - 1
        calib_data = calib_data.astype(dtype)
        calib_data.tofile(os.path.join(input_dir, 'calib_data.bin'))
        save_numpy_array_as_txt(os.path.join(input_dir, 'calib_data_{0}.txt'.format(input)), calib_data)
        calib_datas.append(calib_data)

    return calib_datas

def eval_onnx_gth(case_name, model_file, n):
    case_dir = os.path.join(output_root, case_name)

    # create session
    try:
        print('[info]: using simplified model')
        model_file = os.path.join(case_dir, 'simplified.onnx')
        sess = ort.InferenceSession(model_file)
    except Exception as e:
        print(e)
        try:
            print('[info]: using origin model')
            sess = ort.InferenceSession(model_file)
        except Exception as e:
            print(e)
            print('[info]: using converted model')
            onnx_model = onnx.load(model_file)
            onnx_model = version_converter.convert_version(onnx_model, 8)
            model_file = os.path.join(case_dir, 'converted.onnx')
            onnx.save_model(onnx_model, model_file)
            sess = ort.InferenceSession(model_file)

    # get shape/dtype of input
    onnx_model = onnx.load(model_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]
    out_len = len(onnx_model.graph.output)
    inputs, dtype_dict, shape_dict = get_onnx_shape_dtype_inputs(input_tensors)
    print('[info]: inputs:', inputs)
    print('[info]: dtype dict:', dtype_dict)
    print('[info]: shape dict:', shape_dict)

    # gen input
    inputs_dict = gen_input(case_name, inputs, dtype_dict, shape_dict)
    calib_list = gen_calib_data(case_name, inputs, dtype_dict, shape_dict, n)

    outputs = sess.run(None, inputs_dict)
    for i in range(out_len):
        result = outputs[i]
        result.tofile(os.path.join(case_dir, 'cpu_result{0}.bin'.format(i)))
        save_numpy_array_as_txt(os.path.join(case_dir, 'cpu_result{0}.txt'.format(i)), result)

    input_list = []
    for input_name in inputs_dict:
        input_list.append(inputs_dict[input_name])

    return out_len, input_list, calib_list

def graph_eval_onnx_nncase(case_name, model, targets, inputs, enable_ptq):
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
        compiler.import_onnx(model, import_options)
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

def compile_onnx_nncase(case_name, model_buf, targets, input, n, enable_ptq):
    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.dump_asm = True
    compile_options.dump_ir = True
    for target in targets:
        kmodel_dir = os.path.join(output_root, case_name, target, 'infer', 'ptq' if enable_ptq else 'no_ptq')
        compile_options.target = target
        compile_options.dump_dir = kmodel_dir
        compiler = nncase.Compiler(compile_options)
        compiler.import_onnx(model_buf, import_options)
        if enable_ptq:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(input.tobytes())
            ptq_options.samples_count = n
            compiler.use_ptq(ptq_options)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(os.path.join(kmodel_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_buf = f.read()
    return model_buf

def test_onnx_module(case_name, module, in_shape, targets):
    targets = validate_targets(targets)
    case_name = case_name.replace('[', '_').replace(']', '_')
    case_dir = os.path.join(output_root, case_name)
    clear(case_dir)
    n = 10

    model_file = torch_module_to_onnx(case_name, module, in_shape)

    # preprocess
    old_onnx_model = onnx.load(model_file)
    onnx_model = preprocess_onnx_model(old_onnx_model)
    onnx_model = onnx_model or preprocess_onnx_model(old_onnx_model, convert_version=False)
    onnx_model = onnx_model or preprocess_onnx_model(old_onnx_model, simplify=False)
    onnx_model = onnx_model or preprocess_onnx_model(old_onnx_model, convert_version=False, simplify=False)
    onnx_model = onnx_model or preprocess_onnx_model(old_onnx_model, fix_bn=False, convert_version=False, simplify=False)
    model_file = os.path.join(case_dir, 'simplified.onnx')
    onnx.save_model(onnx_model, model_file)

    onnx_buf = read_model_file(model_file)

    out_len, input_datas, calib_datas = eval_onnx_gth(case_name, model_file, n)

    # evaluation
    graph_eval_onnx_nncase(case_name, onnx_buf, targets, input_datas, False)
    ret = compare_util.compare_results(
        case_dir, out_len, targets, enable_ptq=False, is_evaluation=True)
    assert ret

    # compile & infer
    for enable_ptq in [False, True]:
        if len(input_datas) > 1 and enable_ptq:
            continue
        compile_onnx_nncase(case_name, onnx_buf, targets,
                              calib_datas, n, enable_ptq=enable_ptq)
        infer_nncase(case_name, input_datas, targets, enable_ptq=enable_ptq)
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
