# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import subprocess
import tensorflow as tf
import numpy as np
import shutil
import struct
import nncase

pb_export_dir = "./tmp/test_model"
tflite_export_file = "./tmp/test.tflite"
kmodel_export_dir = "./tmp"
kmodel_out_dir = "./tmp/kmodel_out"
expect_out_dir = "./tmp/expect_out"
input_dir = "./tmp/input"

use_cosine_to_double_check = True
process_num = 16


def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


def length(v):
    return sqrt(dot(v, v))


def cosine_similarity(v1, v2):
    return dot(v1, v2) / (length(v1) * length(v2))


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


def tf_module_to_tflite(module):
    if not os.path.exists(pb_export_dir):
        os.makedirs(pb_export_dir)
    tf.saved_model.save(module, pb_export_dir, module.__call__)
    converter = tf.lite.TFLiteConverter.from_saved_model(pb_export_dir)
    tflite_model = converter.convert()
    with open(tflite_export_file, 'wb') as f:
        f.write(tflite_model)
    return tflite_model


def gen_input(input_tensor):
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


def eval_tflite_gth(model):
    tflite = tf_module_to_tflite(model)
    interp = tf.lite.Interpreter(tflite)
    interp.allocate_tensors()
    input_tensor = interp.get_input_details()[0]
    input_id = input_tensor["index"]
    input = gen_input(input_tensor)
    interp.set_tensor(input_id, input)
    interp.invoke()

    out_len = len(interp.get_output_details())
    for i in range(out_len):
        output_id = interp.get_output_details()[i]["index"]

        result = interp.get_tensor(output_id)
        if len(result.shape) == 4:
            result = np.transpose(result, [0, 3, 1, 2])
        result.tofile(os.path.join(dir, 'cpu_result{0}.bin'.format(i)))
        save_numpy_array_as_txt(os.path.join(
            dir, 'cpu_result{0}.txt'.format(i)), result)
    return out_len, input


def compile_tflite_nncase(model, targets):
    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.dump_asm = True
    compile_options.dump_asm = True
    for target in targets:
        compile_options.target = target
        compiler = nncase.Compiler(compile_options)
        compiler.import_tflite(model, import_options)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        kmodel_dir = os.path.join(kmodel_export_dir, target)
        if not os.path.exists(kmodel_dir):
            os.makedirs(kmodel_dir)
        with open(os.path.join(kmodel_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)


def eval_kmodel(input, targets):
    for target in targets:
        with open(os.path.join(kmodel_export_dir, target, 'test.kmodel', 'rb')) as f:
            kmodel = f.read()
		sim = nncase.Simulator()
		sim.load_model(kmodel)


def clear():
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.makedirs('./tmp')


def copy_tflite(path):
    shutil.copyfile(path, tflite_export_file)


def copy_input(path):
    shutil.copyfile(path, input_dir)


def save_tflite(model):
    if not os.path.exists(pb_export_dir):
        os.makedirs(pb_export_dir)
    tf.saved_model.save(model, pb_export_dir, model.__call__)

    converter = tf.lite.TFLiteConverter.from_saved_model(pb_export_dir)
    tflite_model = converter.convert()
    f = open(tflite_export_file, 'wb')
    f.write(tflite_model)
    f.close()


def compile(args=[]):
    retcode = subprocess.call([ncc, 'compile', tflite_export_file, kmodel_export_file,
                               '-i', 'tflite', *args])
    print('retcode', retcode)
    assert retcode is 0


def infer(args=[]):
    if not os.path.exists(kmodel_out_dir):
        os.makedirs(kmodel_out_dir)
    retcode = subprocess.call(
        [ncc, 'infer', kmodel_export_file, kmodel_out_dir, '--dataset', input_dir, *args])
    print('retcode', retcode)
    assert retcode is 0


def save_expect_array(name, array):
    if not os.path.exists(expect_out_dir):
        os.makedirs(expect_out_dir)
    np.asarray(array, dtype=np.float32).tofile(
        expect_out_dir + '/' + name + '.bin')


def save_input_array(name, array):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    np.asarray(array, dtype=np.float32).tofile(input_dir + '/' + name + '.bin')


def load_input_array(name, shape):
    return np.fromfile(input_dir + '/' + name + '.bin', dtype=np.float32).reshape(shape)


def run_tflite(input):
    interp = tf.lite.Interpreter(tflite_export_file)
    interp.allocate_tensors()
    input_id = interp.get_input_details()[0]["index"]
    output_id = interp.get_output_details()[0]["index"]
    interp.set_tensor(input_id, input)
    interp.invoke()
    return interp.get_tensor(output_id)


def run_tflite_multi(input, out_num):
    interp = tf.lite.Interpreter(tflite_export_file)
    interp.allocate_tensors()
    input_id = interp.get_input_details()[0]["index"]
    interp.set_tensor(input_id, input)
    interp.invoke()
    out = []
    for i in range(0, out_num):
        output_id = interp.get_output_details()[i]["index"]
        out.append(interp.get_tensor(output_id))
    return out


def flatten_out(out):
    res = []
    for o1 in out:
        for o2 in o1.flatten():
            res.append(o2)
    return np.asarray(res)


def close_to(name, threshold):
    expect_arr = np.fromfile(expect_out_dir + '/' +
                             name + '.bin', dtype=np.float32)
    actual_arr = np.fromfile(kmodel_out_dir + '/' +
                             name + '.bin', dtype=np.float32)
    error = np.sum(np.square(expect_arr - actual_arr)) / len(expect_arr)
    print('e-a', expect_arr - actual_arr)
    print('exp', expect_arr)
    print('act', actual_arr)
    print('error:', error)
    assert error <= threshold
