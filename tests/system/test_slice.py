import pytest
import os
import subprocess
import tensorflow as tf
import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc

class SliceModule(tf.Module):

  def __init__(self):
    super(SliceModule, self).__init__()
    self.v = tf.constant([9., -2., -7.])

  @tf.function(input_signature=[tf.TensorSpec([1,2,3], tf.float32)])
  def __call__(self, x):
    return x[:,:,1:-1]

module = SliceModule()

def init_values():
	input = np.asarray([1,-2,3,4,-9,0], dtype=np.float32).reshape([1,2,-1])
	ncc.save_input_array('test', input)
	ncc.save_expect_array('test', ncc.run_tflite(input))

def test_slice():
	ncc.clear()
	ncc.save_tflite(module)
	init_values()
	ncc.compile(['--inference-type', 'float'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 0)

if __name__ == "__main__":
	test_slice()