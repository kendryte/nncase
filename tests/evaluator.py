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
        self.set_inputs(evaluator)
        evaluator.run()
        return self.dump_outputs(evaluator, dump_dir)

    def set_inputs(self, evaluator):
        for idx, i in enumerate(self.inputs):
            input_tensor: nncase.RuntimeTensor = nncase.RuntimeTensor.from_numpy(
                self.transform_input((i['data']), self.cfg['compile_opt']['input_type'], "infer")[0])
            evaluator.set_input_tensor(idx, input_tensor)

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
