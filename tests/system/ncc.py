import pytest
import os
import subprocess
import tensorflow as tf

ncc = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../out/bin/ncc")
pb_export_dir = "./tmp/test_model"
tflite_export_file = "./tmp/test.tflite"
kmodel_export_file = "./tmp/test.kmodel"

def compile(model, outputs, args=[]):
	if not os.path.exists(pb_export_dir):
		os.makedirs(pb_export_dir)
	tf.saved_model.save(model, pb_export_dir, outputs)

	converter = tf.lite.TFLiteConverter.from_saved_model(pb_export_dir)
	tflite_model = converter.convert()
	f = open(tflite_export_file, 'wb')
	f.write(tflite_model)
	f.close()

	retcode = subprocess.call([ncc, 'compile', tflite_export_file, kmodel_export_file,
	 '-i', 'tflite', *args])
	assert retcode is 0

	