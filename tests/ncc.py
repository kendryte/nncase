import pytest
import os
import subprocess
import tensorflow as tf
import numpy as np
import shutil

ncc = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../out/bin/ncc")
pb_export_dir = "./tmp/test_model"
tflite_export_file = "./tmp/test.tflite"
kmodel_export_file = "./tmp/test.kmodel"
kmodel_out_dir = "./tmp/kmodel_out"
expect_out_dir = "./tmp/expect_out"
input_dir = "./tmp/input"

def clear():
	if os.path.exists('./tmp'):
		shutil.rmtree('./tmp')
	os.makedirs('./tmp')

def copy_tflite(path):
	shutil.copyfile(path, tflite_export_file)

def copy_input(path):
	shutil.copyfile(path, input_dir)

def save_tflite(model):
	if not os.path.exists(pb_export_dir):
		os.makedirs(pb_export_dir)
	tf.saved_model.save(model, pb_export_dir, model.__call__)

	converter = tf.lite.TFLiteConverter.from_saved_model(pb_export_dir)
	tflite_model = converter.convert()
	f = open(tflite_export_file, 'wb')
	f.write(tflite_model)
	f.close()

def compile(args=[]):
	retcode = subprocess.call([ncc, 'compile', tflite_export_file, kmodel_export_file,
	 '-i', 'tflite', *args])
	print('retcode', retcode)
	assert retcode is 0

def infer(args=[]):
	if not os.path.exists(kmodel_out_dir):
		os.makedirs(kmodel_out_dir)
	retcode = subprocess.call([ncc, 'infer', kmodel_export_file, kmodel_out_dir, '--dataset', input_dir, *args])
	print('retcode', retcode)
	assert retcode is 0

def save_expect_array(name, array):
	if not os.path.exists(expect_out_dir):
		os.makedirs(expect_out_dir)
	np.asarray(array, dtype=np.float32).tofile(expect_out_dir + '/' + name + '.bin')

def save_input_array(name, array):
	if not os.path.exists(input_dir):
		os.makedirs(input_dir)
	np.asarray(array, dtype=np.float32).tofile(input_dir + '/' + name + '.bin')

def load_input_array(name, shape):
	return np.fromfile(input_dir + '/' + name + '.bin', dtype=np.float32).reshape(shape)

def run_tflite(input):
	interp = tf.lite.Interpreter(tflite_export_file)
	interp.allocate_tensors()
	input_id = interp.get_input_details()[0]["index"]
	output_id = interp.get_output_details()[0]["index"]
	interp.set_tensor(input_id, input)
	interp.invoke()
	return interp.get_tensor(output_id)

def close_to(name, threshold):
	expect_arr = np.fromfile(expect_out_dir + '/' + name + '.bin', dtype=np.float32)
	actual_arr = np.fromfile(kmodel_out_dir + '/' + name + '.bin', dtype=np.float32)
	error = np.sum(np.square(expect_arr - actual_arr)) / len(expect_arr)
	print('e-a', expect_arr - actual_arr)
	print('exp', expect_arr)
	print('act', actual_arr)
	print('error:', error)
	assert error <= threshold