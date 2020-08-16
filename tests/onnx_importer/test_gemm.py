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

class GemmModule(torch.nn.Module):
    def __init__(self):
        super(GemmModule, self).__init__()
        self.c = torch.tensor([[4, 5], [7, 8]], dtype=torch.float32)

    def forward(self, x):
        return torch.mm(x.squeeze(dim=0), self.c)

module = GemmModule()

@pytest.fixture
def input():
    return np.asarray([1, -2, 3,
                       0, 7, -11], dtype=np.float32).reshape([1, 3, 2])

def test_gemm(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0)

def test_gemm_k210(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'k210',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.05)

def test_gemm_quant(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'cpu',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.05)

if __name__ == "__main__":
    test_gemm()
    test_gemm_k210()
    test_gemm_quant()
