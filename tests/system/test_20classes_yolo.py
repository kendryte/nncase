import pytest
import os
import subprocess
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
[sys.path.append(i) for i in ['.', '..']]
import ncc

dir_prefix = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../examples/20classes_yolo")

def normalize(x):
    return (x.astype(np.float32) - 127.5) / 128.0

def init_values():
	input = [normalize(plt.imread(dir_prefix + '/k210/kpu_20classes_example/dog.bmp'))]
	expect = ncc.run_tflite(input)
	expect = np.transpose(expect, [0,3,1,2])
	ncc.save_expect_array('test', expect)
	input = np.transpose(input, [0,3,1,2])
	ncc.save_input_array('test', input)

def test_simple():
	ncc.clear()
	ncc.copy_tflite(dir_prefix + '/model/20classes_yolo.tflite')
	init_values()
	ncc.compile(['--inference-type', 'float', '--max-allocator-solve-secs', '0'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 1e-6)
	
def test_simple_quant():
	ncc.clear()
	ncc.copy_tflite(dir_prefix + '/model/20classes_yolo.tflite')
	init_values()
	ncc.compile(['--inference-type', 'uint8', '-t', 'cpu',
	 '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
	 '--input-type', 'float', '--max-allocator-solve-secs', '0'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 1.3)
	
def test_simple_k210():
	ncc.clear()
	ncc.copy_tflite(dir_prefix + '/model/20classes_yolo.tflite')
	init_values()
	ncc.compile(['--inference-type', 'uint8', '-t', 'k210',
	 '--dataset', ncc.input_dir + '/test.bin', '--dataset-format', 'raw',
	 '--input-type', 'float', '--max-allocator-solve-secs', '0'])

	ncc.infer(['--dataset-format', 'raw'])
	ncc.close_to('test', 0.16)

if __name__ == "__main__":
	#test_simple()
	#test_simple_quant()
	test_simple_k210()