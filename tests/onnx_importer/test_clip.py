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

class ClipModule(torch.nn.Module):
    def __init__(self):
        super(ClipModule, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 3)

module = ClipModule()

@pytest.fixture
def input():
    return np.asarray([1, -2, 3,
                       0, 7, -11], dtype=np.float32).reshape([1, 3, 2])

def test_clip(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0)

def test_clip_k210(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'k210',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.05)

def test_clip_quant(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input))
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'cpu',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.05)

def test_clip_onnx_9(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input), opset_version=9)
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0)

def test_clip_k210_onnx_9(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input), opset_version=9)
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'k210',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.05)

def test_clip_quant_onnx_9(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input), opset_version=9)
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'uint8', '-t', 'cpu',
     '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
     '--input-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0.05)

if __name__ == "__main__":
    test_clip()
    test_clip_k210()
    test_clip_quant()

    test_clip_onnx_9()
    test_clip_k210_onnx_9()
    test_clip_quant_onnx_9()
