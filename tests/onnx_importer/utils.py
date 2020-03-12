import subprocess

import torch
import onnxruntime as ort

from ncc import ncc, kmodel_export_file


onnx_export_file = "./tmp/test.onnx"


def save(model, input, opset_version=9):
	torch.onnx.export(model, input, onnx_export_file,
					  opset_version=opset_version)


def compile(args=[]):
	retcode = subprocess.call([ncc, 'compile', onnx_export_file, kmodel_export_file,
	 '-i', 'onnx', *args])
	print('retcode', retcode)
	assert retcode is 0


def run(input):
	print("onnx_export_file:", onnx_export_file)
	sess = ort.InferenceSession(onnx_export_file)
	input_name = sess.get_inputs()[0].name
	outputs = sess.run(None, { input_name: input })

	return outputs
