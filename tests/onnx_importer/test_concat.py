import pytest
import os
import subprocess
import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc
import onnx_importer.utils
import torch

class ConcatModule(torch.nn.Module):
    def __init__(self):
        super(ConcatModule, self).__init__()
        self.v = torch.tensor([[10., 10., 10.], [11., 11., 11.]]).reshape([1, 2, 3])

    def forward(self, x):
        return torch.cat([x, self.v], dim=-1)

module = ConcatModule()

@pytest.fixture
def input():
    return np.asarray([1.2,-2.6,3.1,4,-9.9,0.499], dtype=np.float32).reshape([1,2,-1])

def test_concat(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 1e-10)

def test_concat_quant(input):
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
    test_concat()
    test_concat_quant()
