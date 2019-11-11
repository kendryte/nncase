import pytest
import os
import subprocess
import ncc
import tensorflow as tf

def test_simple_conv():
	root = tf.Module()
	root.v1 = tf.Variable(3.)
	root.v2 = tf.Variable(3.)
	root.f = tf.function(lambda x: root.v1 * root.v2 * x)

	input_data = tf.constant(1., shape=[1, 1])
	to_save = root.f.get_concrete_function(input_data)
	ncc.compile(root, to_save, ['--inference-type', 'float'])

if __name__ == "__main__":
	test_simple_conv()