import pytest
import os
import subprocess
import ncc
import tensorflow as tf

class SimpeMatMulModule(tf.Module):

  def __init__(self):
    super(SimpeMatMulModule, self).__init__()
    self.v = tf.constant([[1.,2.],[3.,4.]])

  @tf.function(input_signature=[tf.TensorSpec([1,2], tf.float32)])
  def __call__(self, x):
    return tf.matmul(x, self.v)

module = SimpeMatMulModule()

def init_values():
	ncc.save_input_array('test', [1.,2.])
	ncc.save_expect_array('test', [7.,10.])

def test_simple_matmul():
	ncc.clear()
	init_values()
	ncc.compile(module, ['--inference-type', 'float'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 0)
	
def test_simple_matmul_quant():
	ncc.clear()
	init_values()
	ncc.compile(module, ['--inference-type', 'uint8',
	 '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
	 '--use-float-input'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 1e-3)

if __name__ == "__main__":
	#test_simple_matmul()
	test_simple_matmul_quant()