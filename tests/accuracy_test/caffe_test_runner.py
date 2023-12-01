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

import caffe
from test_runner import *
import os
import shutil
import numpy as np
# from typing import Dict, List, Tuple, Union
from test_utils import *


class CaffeTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: dict = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "caffe"

    def run(self, model_file_list):
        super().run(model_file_list)

    def parse_model(self, model_path: Union[List[str], str]):
        caffe_model = caffe.Net(model_path[0], model_path[1], caffe.TEST)
        for i, name in enumerate(caffe_model._layer_names):
            if (caffe_model.layers[i].type == "Input"):
                input_dict = {}
                input_dict['name'] = name
                input_dict['dtype'] = np.float32
                input_dict['model_shape'] = list(caffe_model.blobs[name].data.shape)
                self.inputs.append(input_dict)
                self.calibs.append(copy.deepcopy(input_dict))
                # self.dump_range_data.append(copy.deepcopy(input_dict))

        used_inputs = set([name for _, l in caffe_model.bottom_names.items() for name in l])
        seen_outputs = set()
        for n in [name for _, l in caffe_model.top_names.items() for name in l]:
            if not n in used_inputs and not n in seen_outputs:
                seen_outputs.add(n)
                output_dict = {}
                output_dict['name'] = n
                output_dict['dtype'] = np.float32
                output_dict['model_shape'] = list(caffe_model.blobs[n].data.shape)
                self.outputs.append(output_dict)

    def cpu_infer(self, model_file_list):
        caffe_model = caffe.Net(model_file_list[0], model_file_list[1], caffe.TEST)

        number = self.cfg['generator']['inputs']['number']
        new_inputs = []
        for idx, value in enumerate(self.inputs):
            new_value = self.transform_input(self.data_pre_process(value['data']), "float32", "CPU")
            new_inputs.append(new_value)

            if self.cfg['compile_opt']['preprocess'] and not test_utils.in_ci():
                for n in range(number):
                    dump_bin_file(os.path.join(
                        self.case_dir, f'frame_input_{idx}_{n}.bin'), new_value[n])
                    dump_txt_file(os.path.join(
                        self.case_dir, f'frame_input_{idx}_{n}.txt'), new_value[n])

        outputs = []
        for n in range(number):
            output = []

            # set input
            for idx, value in enumerate(self.inputs):
                caffe_model.blobs[value['name']].data[...] = new_inputs[idx][n]

            # run
            results = caffe_model.forward()

            # get output
            for out in self.outputs:
                output.append(results[out['name']])
            outputs.append(output)

        return outputs

    def import_model(self, compiler, model_content, import_options):
        compiler.import_caffe(model_content[1], model_content[0])
