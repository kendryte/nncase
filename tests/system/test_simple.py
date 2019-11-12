import pytest
import os
import subprocess
import ncc
import tensorflow as tf
import numpy as np

class SimpeModule(tf.Module):

  def __init__(self):
    super(SimpeModule, self).__init__()
    self.v = tf.constant(9.)

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def __call__(self, x):
    return x * self.v

module = SimpeModule()

def init_values():
	input = np.asarray([1.], dtype=np.float32)
	ncc.save_input_array('test', input)
	ncc.save_expect_array('test', ncc.run_tflite(input[0]))

def test_simple():
	ncc.clear()
	ncc.save_tflite(module)
	init_values()
	ncc.compile(['--inference-type', 'float'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 0)
	
def test_simple_quant():
	ncc.clear()
	ncc.save_tflite(module)
	init_values()
	ncc.compile(['--inference-type', 'uint8',
	 '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
	 '--use-float-input'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 1e-3)

if __name__ == "__main__":
	test_simple()
	test_simple_quant()