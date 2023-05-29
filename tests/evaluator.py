from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils


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
        compile_options.shape_bucket_options = nncase.ShapeBucketOptions()
        compile_options.shape_bucket_options.enable = False
        compile_options.shape_bucket_options.range_info = {}
        compile_options.shape_bucket_options.segments_count = 2
        compile_options.shape_bucket_options.fix_var_map = {}
        self.compiler = nncase.Compiler(compile_options)
        self.import_model(self.compiler, model_content, import_options)
        self.set_quant_opt(cfg, kwargs, preprocess, self.compiler)
        evaluator = self.compiler.create_evaluator(3)
        self.set_inputs(evaluator)
        evaluator.run()
        eval_output_paths = self.dump_outputs(eval_dir, evaluator)
        return eval_output_paths

    def set_inputs(self, evaluator):
        for i in range(len(self.inputs)):
            input_tensor: nncase.RuntimeTensor = nncase.RuntimeTensor.from_numpy(
                self.transform_input(self.data_pre_process(self.inputs[i]['data'])[0], "float32", "CPU"))
            evaluator.set_input_tensor(i, input_tensor)

    def dump_outputs(self, eval_dir, evaluator):
        eval_output_paths = []
        for i in range(evaluator.outputs_size):
            result = evaluator.get_output_tensor(i).to_numpy()
            os.makedirs(eval_dir, exist_ok=True)
            eval_output_paths.append((
                os.path.join(eval_dir, f'nncase_result_{i}.bin'),
                os.path.join(eval_dir, f'nncase_result_{i}.txt')))
            result.tofile(eval_output_paths[-1][0])
            if not test_utils.in_ci():
                self.totxtfile(eval_output_paths[-1][1], result)
        return eval_output_paths
