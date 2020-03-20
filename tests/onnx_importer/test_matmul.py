import pytest
import os
import subprocess
import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc
import onnx_importer.utils
import torch

class MatmulModule(torch.nn.Module):
    def __init__(self):
        super(MatmulModule, self).__init__()
        self.v = torch.tensor([[1., 2.], [3., 4.]])

    def forward(self, x):
        return torch.matmul(x, self.v)

module = MatmulModule()

@pytest.fixture
def input():
    return np.asarray([1, 2], dtype=np.float32).reshape([1,-1])

def test_matmul(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0)

def test_matmul_quant(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'cpu',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.005)

if __name__ == "__main__":
    test_matmul()
    test_matmul_quant()
