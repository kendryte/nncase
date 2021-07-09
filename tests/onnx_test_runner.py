from onnx import version_converter, helper
import onnxsim
import onnxruntime as ort
import onnx
import torch
import shutil
import os
import numpy as np
from test_runner import *

class OnnxTestRunner(TestRunner):
    def __init__(self, case_name, targets=None):
        super().__init__(case_name, targets)

    def from_torch(self, module, in_shape, opset_version=11):
        # export model
        dummy_input = torch.randn(*in_shape)
        model_file = os.path.join(self.case_dir, 'test.onnx')
        torch.onnx.export(module, dummy_input, model_file,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=opset_version)
        return model_file

    def from_onnx_helper(self, model_def):
        try:
            onnx.checker.check_model(model_def)
        except onnx.checker.ValidationError as e:
            print('The model is invalid: %s' % e)
        else:
            print('The model is valid!')

        model_file = os.path.join(self.case_dir, 'test.onnx')
        onnx.save(model_def, model_file)

        return model_file

    def run(self, model_file):
        if self.case_dir != os.path.dirname(model_file):
            shutil.copy(model_file, self.case_dir)
            model_file = os.path.join(self.case_dir, os.path.basename(model_file))

        # preprocess model
        old_onnx_model = onnx.load(model_file)
        onnx_model = self.preprocess_model(old_onnx_model)
        onnx_model = onnx_model or self.preprocess_model(
            old_onnx_model, convert_version=False)
        onnx_model = onnx_model or self.preprocess_model(
            old_onnx_model, simplify=False)
        onnx_model = onnx_model or self.preprocess_model(
            old_onnx_model, convert_version=False, simplify=False)
        onnx_model = onnx_model or self.preprocess_model(
            old_onnx_model, fix_bn=False, convert_version=False, simplify=False)

        model_file = os.path.join(
            os.path.dirname(model_file), 'simplified.onnx')
        onnx.save_model(onnx_model, model_file)

        super().run(model_file)

    def map_onnx_to_numpy_type(self, onnx_type):
        ONNX_TO_NUMPY_DTYPE = {
            onnx.onnx_pb.TensorProto.FLOAT: np.float32,
            onnx.onnx_pb.TensorProto.FLOAT16: np.float16,
            onnx.onnx_pb.TensorProto.DOUBLE: np.float64,
            onnx.onnx_pb.TensorProto.INT32: np.int32,
            onnx.onnx_pb.TensorProto.INT16: np.int16,
            onnx.onnx_pb.TensorProto.INT8: np.int8,
            onnx.onnx_pb.TensorProto.UINT8: np.uint8,
            onnx.onnx_pb.TensorProto.UINT16: np.uint16,
            onnx.onnx_pb.TensorProto.INT64: np.int64,
            onnx.onnx_pb.TensorProto.UINT64: np.uint64,
            onnx.onnx_pb.TensorProto.BOOL: bool,
            onnx.onnx_pb.TensorProto.COMPLEX64: np.complex64,
            onnx.onnx_pb.TensorProto.COMPLEX128: np.complex128,
            onnx.onnx_pb.TensorProto.STRING: object,
        }

        return ONNX_TO_NUMPY_DTYPE[onnx_type]

    def preprocess_model(self, onnx_model, fix_bn=True, convert_version=True, simplify=True, import_test=True):
        args = {'fix_bn': fix_bn, 'convert_version': convert_version,
                'simplify': simplify, 'import_test': import_test}
        try:
            shape_dict = {}
            for input in self.inputs:
                input_dict[input['name']] = input['shape']

            if fix_bn:
                # fix https://github.com/onnx/models/issues/242
                for node in onnx_model.graph.node:
                    if(node.op_type == "BatchNormalization"):
                        for attr in node.attribute:
                            if (attr.name == "spatial"):
                                attr.i = 1

            if convert_version:
                curret_version = onnx_model.opset_import[0].version
                for i in range(curret_version, 8):
                    onnx_model = version_converter.convert_version(
                        onnx_model, i+1)

            if simplify:
                onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"

            print('[info]: preprocess ONNX model success: ', args)
            return onnx_model
        except Exception as e:
            print('[info]: preprocess ONNX model failed: ', args)
            print(e)
            # traceback.print_exc()
            return None

    def parse_model_input_output(self, model_file: str):
        # TODO: onnx_model
        onnx_model = onnx.load(model_file)
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer]
        input_names = list(set(input_all) - set(input_initializer))
        input_tensors = [
            node for node in onnx_model.graph.input if node.name in input_names]

        # input
        for _, e in enumerate(input_tensors):
            onnx_type = e.type.tensor_type
            input_dict = {}
            input_dict['name'] = e.name
            input_dict['dtype'] = self.map_onnx_to_numpy_type(
                onnx_type.elem_type)
            input_dict['shape'] = [(i.dim_value if i.dim_value != 0 else d) for i, d in zip(
                onnx_type.shape.dim, [1, 3, 224, 224])]
            self.inputs.append(input_dict)
            self.calibs.append(input_dict.copy())

        # output

    def cpu_infer(self, case_dir: str, model_file: bytes):
        # create session
        try:
            print('[onnx]: using simplified model')
            sess = ort.InferenceSession(model_file)
        except Exception as e:
            print(e)
            try:
                print('[onnx]: using origin model')
                model_file = os.path.join(case_dir, 'test.onnx')
                sess = ort.InferenceSession(model_file)
            except Exception as e:
                print(e)
                print('[onnx]: using converted model')
                onnx_model = onnx.load(model_file)
                onnx_model = version_converter.convert_version(onnx_model, 8)
                model_file = os.path.join(case_dir, 'converted.onnx')
                onnx.save_model(onnx_model, model_file)
                sess = ort.InferenceSession(model_file)

        input_dict = {}
        for input in self.inputs:
            input_dict[input['name']] = input['data']

        outputs = sess.run(None, input_dict)
        i = 0
        for output in outputs:
            bin_file = os.path.join(case_dir, f'cpu_result_{i}.bin')
            text_file = os.path.join(case_dir, f'cpu_result_{i}.txt')
            self.output_paths.append((bin_file, text_file))
            output.tofile(bin_file)
            save_array_as_txt(text_file, output)
            i += 1

    def import_model(self, compiler, model_content, import_options):
        compiler.import_onnx(model_content, import_options)