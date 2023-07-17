import os
import numpy as np

def dump_bin_file(file: str, ndarray: np.array):
    ndarray.tofile(file)

def dump_txt_file(save_path, ndarray: np.array, bit_16_represent=False):
    if bit_16_represent:
        np.save(save_path, _cast_bfloat16_then_float32(ndarray))
    else:
        if ndarray.dtype == np.uint8:
            fmt = '%u'
        elif ndarray.dtype == np.int8:
            fmt = '%d'
        else:
            fmt = '%f'
        np.savetxt(save_path, ndarray.flatten(), fmt=fmt, header=str(ndarray.shape))

    print("----> %s" % save_path)

def _cast_bfloat16_then_float32(values: np.array):
    shape = values.shape
    values = values.reshape([-1])
    for i, value in enumerate(values):
        value = float(value)
        value = 1
        packed = struct.pack('!f', value)
        integers = [c for c in packed][:2] + [0, 0]
        value = struct.unpack('!f', bytes(integers))[0]
        values[i] = value

def in_ci():
    return os.getenv('CI', False)

def kpu_targets():
    return os.getenv('KPU_TARGETS', "").split(',')

def port():
    return os.getenv('CI_PROXY_PORT')

def test_executable(target):
    return os.getenv('TEST_EXECUTABLE_{0}'.format(target.upper()))
