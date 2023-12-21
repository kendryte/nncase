# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

from onnx import version_converter, helper, external_data_helper
import onnxsim
import onnxruntime as ort
import onnx
import torch
import shutil
import os
import numpy as np
from test_runner import *
from test_utils import *
from collections import ChainMap
import threading
import queue


class OnnxTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "onnx"

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
        if model_file.startswith('examples'):
            model_file = os.path.join(os.path.dirname(__file__), '..', model_file)
        elif model_file.startswith('onnx-models'):
            model_file = os.path.join(os.getenv('ONNX_MODELS_DIR'),
                                      model_file[len('onnx-models/'):])
        if self.case_dir != os.path.dirname(model_file):
            new_file = os.path.join(self.case_dir, 'test.onnx')
            shutil.copy(model_file, new_file)
            for tensor in external_data_helper._get_all_tensors(onnx.load(model_file, load_external_data=False)):
                if external_data_helper.uses_external_data(tensor):
                    info = external_data_helper.ExternalDataInfo(tensor)
                    file_location = external_data_helper._sanitize_path(info.location)
                    external_data_src_path = os.path.join(
                        os.path.dirname(model_file), file_location)
                    external_data_dst_path = os.path.join(
                        self.case_dir, file_location)
                    if not os.path.exists(external_data_dst_path):
                        os.symlink(external_data_src_path, external_data_dst_path)
            model_file = new_file

        if not self.inputs:
            self.parse_model(model_file)

        model_file = self.do_preprocess(model_file)

        super().run(model_file)

    def do_preprocess(self, model_file):
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
        onnx.save_model(onnx_model, model_file,
                        save_as_external_data=True if onnx_model.ByteSize() > 2147483648 else False)
        return model_file

    def preprocess_model(self, onnx_model, fix_bn=True, convert_version=True, simplify=True, import_test=True):
        args = {'fix_bn': fix_bn, 'convert_version': convert_version,
                'simplify': simplify, 'import_test': import_test}
        try:
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
                        onnx_model, i + 1)

            if simplify:
                onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                input_shapes = {}
                for input in self.inputs:
                    input_shapes[input['name']] = input['shape']

                onnx_model, check = onnxsim.simplify(
                    onnx_model, input_shapes=input_shapes, dynamic_input_shape=self.dynamic)
                assert check, "Simplified ONNX model could not be validated"

            print('[info]: preprocess ONNX model success: ', args)
            return onnx_model
        except Exception as e:
            print('[info]: preprocess ONNX model failed: ', args)
            print(e)
            # traceback.print_exc()
            return None

    def parse_model(self, model_file: str):
        onnx_model = onnx.load(model_file)
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        input_names = list(set(input_all) - set(input_initializer))
        input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]

        def to_dim_value(d, default_d):
            """
            if dim_value is not digit, it should be fixed.
            dim_value range: [0, inf)
            """
            if d.dim_param != "":
                if len(self.shape_vars):
                    # we should eval dim_param instead of get var value
                    # e.g. dim_param = dec_len - 1
                    return eval(f"{d.dim_param}", self.shape_vars)
                else:
                    # if not set shape vars, then return default d
                    # if it has multi input that all of them have var, must set shape var
                    # e.g.
                    # input0: [8,24,dec_len-1]
                    # input1: [8,dec_len-1,24]
                    return default_d
            else:
                return d.dim_value

        def translate_shape(shape, default_shape):
            return [to_dim_value(d, def_d) for d, def_d in zip(shape, default_shape)]

        # input
        for _, e in enumerate(input_tensors):
            onnx_type = e.type.tensor_type
            input_dict = {}
            input_dict['name'] = e.name
            input_dict['dtype'] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.elem_type]
            shape = translate_shape(onnx_type.shape.dim, self.default_shape)
            input_dict['shape'] = shape
            input_dict['model_shape'] = shape
            self.inputs.append(input_dict)
            self.calibs.append(copy.deepcopy(input_dict))
            # self.dump_range_data.append(copy.deepcopy(input_dict))

        def is_dynamic(output):
            dims = output.type.tensor_type.shape.dim
            return any(dim.dim_param != '' for dim in dims)

        outputs = onnx_model.graph.output
        self.dynamic = any(is_dynamic(output) for output in outputs)
        # make a static model for infer output
        if self.dynamic and onnx_model.ByteSize() < 2147483648:
            input_shapes = list(map(lambda input: {input['name']: input['shape']}, self.inputs))
            input_shapes = dict(ChainMap(*input_shapes))
            (onnx_model, _) = onnxsim.simplify(onnx_model, input_shapes=input_shapes)

        # output
        for e in onnx_model.graph.output:
            output_dict = {}
            onnx_type = e.type.tensor_type
            output_dict['name'] = e.name
            if onnx_type.elem_type == 0:
                output_dict['dtype'] = 'float32'
            else:
                output_dict['dtype'] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.elem_type]
            output_dict['model_shape'] = [i.dim_value for i in onnx_type.shape.dim]
            self.outputs.append(output_dict)

    def cpu_infer(self, model_file: bytes):
        generator_cfg = self.cfg['generator']['inputs']
        method = generator_cfg['method']
        number = generator_cfg['number']
        args = os.path.join(test_utils.test_root(), generator_cfg[method]['roofline_args'])

        file_list = []
        assert(os.path.isdir(args))
        for file in os.listdir(args):
            if file.endswith('.bin'):
                file_list.append(os.path.join(args, file))
        file_list.sort()

        # create session
        try:
            print('[onnx]: using simplified model')
            sess = ort.InferenceSession(model_file)
        except Exception as e:
            print(e)
            try:
                print('[onnx]: using origin model')
                model_file = os.path.join(self.case_dir, 'test.onnx')
                sess = ort.InferenceSession(model_file)
            except Exception as e:
                print(e)
                print('[onnx]: using converted model')
                onnx_model = onnx.load(model_file)
                onnx_model = version_converter.convert_version(onnx_model, 8)
                model_file = os.path.join(self.case_dir, 'converted.onnx')
                onnx.save_model(onnx_model, model_file)
                sess = ort.InferenceSession(model_file)

        q = queue.Queue(maxsize=self.postprocess_qsize)
        t = threading.Thread(target=self.postprocess, args=(q, ))
        t.start()

        c = 0
        for i in range(number):
            # set input
            input_dict = {}
            for idx, value in enumerate(self.inputs):
                input_dict[value['name']] = np.fromfile(
                    file_list[c], dtype=value['dtype']).reshape(value['model_shape'])
                c = c + 1

            # run
            outputs = sess.run(None, input_dict)

            # postprocess
            q.put(outputs)

            # debug
            if not test_utils.in_ci():
                for j, output in enumerate(outputs):
                    dump_bin_file(os.path.join(self.case_dir, f'cpu_result_{i}_{j}.bin'), output)
                    # dump_txt_file(os.path.join(self.case_dir, f'cpu_result_{i}_{j}.txt'), output)

        t.join()

    def import_model(self, compiler, model_content, import_options):
        compiler.import_onnx(model_content, import_options)
