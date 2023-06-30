from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils
import preprocess_utils


class Evaluator:
    def run_evaluator(self, eval_args, cfg, case_dir, import_options, compile_options, model_content, preprocess_opt):
        eval_output_paths = self.generate_evaluates(
            cfg, case_dir, import_options,
            compile_options, model_content, eval_args, preprocess_opt)
        return eval_output_paths

    def generate_evaluates(self, cfg, case_dir: str,
                           import_options: nncase.ImportOptions,
                           compile_options: nncase.CompileOptions,
                           model_content: Union[List[bytes], bytes],
                           kwargs: Dict[str, str],
                           preprocess: Dict[str, str]
                           ) -> List[Tuple[str, str]]:
        eval_dir = self.kwargs_to_path(
            os.path.join(case_dir, 'eval'), kwargs)
        compile_options.target = kwargs['target']
        compile_options.dump_dir = eval_dir
        compile_options.dump_asm = cfg.compile_opt.dump_asm
        compile_options.dump_ir = cfg.compile_opt.dump_ir
        compile_options = preprocess_utils.update_compile_options(compile_options, preprocess)
        compile_options.shape_bucket_options = nncase.ShapeBucketOptions()
        compile_options.shape_bucket_options.enable = False
        compile_options.shape_bucket_options.range_info = {}
        compile_options.shape_bucket_options.segments_count = 2
        compile_options.shape_bucket_options.fix_var_map = {}
        self.compiler = nncase.Compiler(compile_options)
        self.import_model(self.compiler, model_content, import_options)
        self.set_quant_opt(cfg, kwargs, preprocess, self.compiler)
        evaluator = self.compiler.create_evaluator(3)
        self.set_inputs(evaluator, preprocess)
        evaluator.run()
        eval_output_paths = self.dump_outputs(eval_dir, preprocess, evaluator)
        return eval_output_paths

    def set_inputs(self, evaluator, preprocess):
        for idx, i in enumerate(self.inputs):
            input_tensor: nncase.RuntimeTensor = nncase.RuntimeTensor.from_numpy(
                self.transform_input((i['data']), preprocess['input_type'], "infer")[0])
            evaluator.set_input_tensor(idx, input_tensor)

    def dump_outputs(self, eval_dir, preprocess, evaluator):
        eval_output_paths = []
        for i in range(evaluator.outputs_size):
            result = evaluator.get_output_tensor(i).to_numpy()
            if preprocess['preprocess']:
                if(preprocess['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                    result = np.transpose(result, [0, 3, 1, 2])
                elif (preprocess['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                    result = np.transpose(result, [0, 2, 3, 1])
                elif preprocess['output_layout'] not in ["NCHW", "NHWC"] and preprocess['output_layout'] != "":
                    tmp_perm = [int(idx) for idx in preprocess['output_layout'].split(",")]
                    result = np.transpose(
                        result, preprocess_utils.get_source_transpose_index(tmp_perm))
            os.makedirs(eval_dir, exist_ok=True)
            eval_output_paths.append((
                os.path.join(eval_dir, f'nncase_result_{i}.bin'),
                os.path.join(eval_dir, f'nncase_result_{i}.txt')))
            result.tofile(eval_output_paths[-1][0])
            if not test_utils.in_ci():
                self.totxtfile(eval_output_paths[-1][1], result)
        return eval_output_paths
