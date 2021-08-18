from onnxsim import simplify
import onnx
import argparse
import sys

if __name__=='__main__':
  onnx_model = onnx.load(sys.argv[1])  # load onnx model
  model_simp, check = simplify(onnx_model)
  assert check, "Simplified ONNX model could not be validated"
  onnx.save(model_simp, sys.argv[2])
