import pytest
import os
import subprocess
import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc
import onnx_importer.utils
import torch

class PadModule(torch.nn.ConstantPad2d):
    def __init__(self):
        super(PadModule, self).__init__(padding=(4, 3, 2, 1), value=0)

module = PadModule()

@pytest.fixture
def input():
    return np.asarray([1,-2,3,4,-9,0], dtype=np.float32).reshape([2,-1])

def test_pad(input):
    ncc.clear()

    ncc.save_input_array('test', input)
    onnx_importer.utils.save(module, torch.from_numpy(input), opset_version=9)
    ncc.save_expect_array('test', onnx_importer.utils.run(input))

    onnx_importer.utils.compile(['--inference-type', 'float'])

    ncc.infer(['--dataset-format', 'raw'])
    ncc.close_to('test', 0)

if __name__ == "__main__":
    test_pad()
    test_pad_quant()
