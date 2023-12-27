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

from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils
import preprocess_utils
from test_utils import *


class Evaluator:
    def run_evaluator(self, compiler, dump_dir):
        evaluator = compiler.create_evaluator(3)

        # transform input
        new_inputs = []
        for idx, value in enumerate(self.inputs):
            new_value = self.transform_input(
                value['data'], self.cfg['compile_opt']['input_type'], "infer")
            new_inputs.append(new_value)

        outputs = []
        number = self.cfg['generator']['inputs']['number']
        for n in range(number):
            # set input
            for idx, value in enumerate(self.inputs):
                evaluator.set_input_tensor(idx, nncase.RuntimeTensor.from_numpy(new_inputs[idx][n]))

            # run
            evaluator.run()

            # get output
            output = self.dump_outputs(evaluator, dump_dir)

            outputs.append(output)
        return outputs

    def dump_outputs(self, evaluator, eval_dir):
        results = []
        compile_opt = self.cfg['compile_opt']
        for i in range(evaluator.outputs_size):
            result = evaluator.get_output_tensor(i).to_numpy()
            if compile_opt['preprocess']:
                if(compile_opt['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                    result = np.transpose(result, [0, 3, 1, 2])
                elif (compile_opt['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                    result = np.transpose(result, [0, 2, 3, 1])
                elif compile_opt['output_layout'] not in ["NCHW", "NHWC"] and compile_opt['output_layout'] != "":
                    tmp_perm = [int(idx) for idx in compile_opt['output_layout'].split(",")]
                    result = np.transpose(
                        result, preprocess_utils.get_source_transpose_index(tmp_perm))
            os.makedirs(eval_dir, exist_ok=True)
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(eval_dir, f'nncase_result_{i}.bin'), result)
                dump_txt_file(os.path.join(eval_dir, f'nncase_result_{i}.txt'), result)

            results.append(result)
        return results
