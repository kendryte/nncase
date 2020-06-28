import pytest
import os
import subprocess
import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc
import onnx_importer.utils
import torch

output_size = [1, 3, 2, 2]

class GlobalAvgPoolModule(torch.nn.AdaptiveAvgPool2d):
    def __init__(self):
        super(GlobalAvgPoolModule, self).__init__(output_size=[1, 1])

module = GlobalAvgPoolModule()

@pytest.fixture
def input():
    return np.asarray([1, -2, 3, 4, -9, 5,
                       0, 7, -11, 4, 2, 19], dtype=np.float32).reshape(output_size)

def test_global_avg_pool(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input), opset_version=9)
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0)

def test_global_avg_pool_quant(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input), opset_version=9)
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'cpu',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.005)

if __name__ == "__main__":
    test_global_avg_pool()
    test_global_avg_pool_quant()
