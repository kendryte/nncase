import pytest
import os
import subprocess
import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc
import onnx_importer.utils
import torch

class BatchNorm2dModule(torch.nn.BatchNorm2d):
    def __init__(self):
        super(BatchNorm2dModule, self).__init__(num_features=3, momentum=None)

@pytest.fixture
def input():
    return np.asarray([1, -2, 3, 4, -9, 5,
                       0, 7, -11, 4, 2, 19,
                       8, -6, 0, -1, 5, 23,
                       9, -13, 7, 5, 4, -6],
                      dtype=np.float32).reshape([1, 3, 2, 4])

@pytest.fixture
def module(input):
	m = BatchNorm2dModule()
	m.train()
	m.forward(torch.from_numpy(input))
	m.eval()

	print("mean:", m.running_mean)
	print("var:", m.running_var)
	print("weight:", m.weight)
	print("bias:", m.bias)

	return m

def test_batchnorm(module, input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.02)

def test_batchnorm_k210(module, input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'k210',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.02)

def test_batchnorm_quant(module, input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'cpu',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.02)

if __name__ == "__main__":
    test_batchnorm()
    test_batchnorm_k210()
    test_batchnorm_quant()
