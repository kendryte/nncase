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

import tensorflow as tf
from test_runner import *
import os
import shutil
from test_utils import *
import threading
import queue


class TfliteTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "tflite"
        self.interp = None

    def from_tensorflow(self, module):
        # export model
        tf.saved_model.save(module, self.case_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(self.case_dir)

        # convert model
        tflite_model = converter.convert()
        model_file = os.path.join(self.case_dir, 'test.tflite')
        with open(model_file, 'wb') as f:
            f.write(tflite_model)

        return model_file

    def from_keras(self, keras_model):
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        model_file = os.path.join(self.case_dir, 'test.tflite')
        with open(model_file, 'wb') as f:
            f.write(tflite_model)

        return model_file

    def run(self, model_file):
        if model_file.startswith('examples'):
            model_file = os.path.join(os.path.dirname(__file__), '..', model_file)
        if self.case_dir != os.path.dirname(model_file):
            shutil.copy(model_file, self.case_dir)
            model_file = os.path.join(
                self.case_dir, os.path.basename(model_file))
        super().run(model_file)

    def parse_model(self, model_path: str):
        self.interp = tf.lite.Interpreter(model_path=model_path)

        def translate_shape(shape):
            return [i if i > 0 else self.shape_vars["-1"] for i in shape]

        for item in self.interp.get_input_details():
            input_dict = {}
            input_dict['index'] = item['index']
            input_dict['name'] = item['name']
            input_dict['dtype'] = item['dtype']

            if len(self.shape_vars) == 0:
                # fixed shape
                shape = item['shape']
            else:
                # dynamic shape
                shape = item['shape_signature']

            if len(shape) <= 4:
                input_dict['model_shape'] = translate_shape(shape)
            else:
                if -1 in shape:
                    raise "tflite test_runner not supported dynamic shape which rank > 4"
                input_dict['model_shape'] = shape

            if -1 in shape:
                self.dynamic = True
                self.interp.resize_tensor_input(item['index'], input_dict['model_shape'])
            self.inputs.append(input_dict)
            self.calibs.append(copy.deepcopy(input_dict))

        for item in self.interp.get_output_details():
            output_dict = {}
            output_dict['index'] = item['index']
            output_dict['name'] = item['name']
            output_dict['dtype'] = item['dtype']
            output_dict['model_shape'] = item['shape']
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

        q = queue.Queue(maxsize=self.postprocess_qsize)
        t = threading.Thread(target=self.postprocess, args=(q, ))
        t.start()

        self.interp.allocate_tensors()
        c = 0
        for i in range(number):
            # set input
            for idx, value in enumerate(self.inputs):
                data = np.fromfile(file_list[c], dtype=value['dtype']).reshape(value['model_shape'])
                c = c + 1
                self.interp.set_tensor(value["index"], data)

            # run
            self.interp.invoke()

            # get output
            outputs = []
            for j, value in enumerate(self.outputs):
                output = self.interp.get_tensor(value['index'])
                outputs.append(output)
                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(self.case_dir, f'cpu_result_{i}_{j}.bin'), output)
                    # dump_txt_file(os.path.join(self.case_dir, f'cpu_result_{i}_{j}.txt'), output)

            # postprocess
            q.put(outputs)
        t.join()

    def import_model(self, compiler, model_content, import_options):
        compiler.import_tflite(model_content, import_options)
