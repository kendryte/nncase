from test_runner import *
import os
import nncase
import numpy as np

class Evaluator:
    def run_evaluator(self, cfg, case_dir, import_options, compile_options, model_content, preprocess_opt):
        names, args = self.split_value(cfg.eval)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            if dict_args['ptq'] and len(self.inputs) != 1:
                continue
            if cfg.compile_opt.dump_import_op_range and len(self.inputs) != 1:
                continue
            eval_output_paths = self.generate_evaluates(
                cfg, case_dir, import_options,
                compile_options, model_content, dict_args, preprocess_opt)
            judge, result = self.compare_results(
                self.output_paths, eval_output_paths, dict_args)
            assert(judge), 'Fault result in eval' + result

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
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)
        self.set_quant_opt(cfg, kwargs, compiler, preprocess)
        evaluator = compiler.create_evaluator(3)
        self.set_inputs(evaluator)
        evaluator.run()
        eval_output_paths = self.dump_outputs(eval_dir, evaluator)
        return eval_output_paths

    def set_quant_opt(self, cfg, kwargs, compiler, preprocess):
        if cfg.compile_opt.dump_import_op_range:
            dump_range_options = nncase.DumpRangeTensorOptions()
            dump_range_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.dump_range_data]).tobytes())
            dump_range_options.samples_count = cfg.generate_dump_range_data.batch_size
            compiler.dump_range_options(dump_range_options)
        if kwargs['ptq']:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.calibs]).tobytes())
            ptq_options.samples_count = cfg.generate_calibs.batch_size
            compiler.use_ptq(ptq_options)


    def set_inputs(self, evaluator):
        for i in range(len(self.inputs)):
            input_tensor: nncase.RuntimeTensor = nncase.RuntimeTensor.from_numpy(
                self.transform_input(self.data_pre_process(self.inputs[i]['data']), "float32", "CPU"))
            input_tensor.copy_to(evaluator.get_input_tensor(i))

    def dump_outputs(self, eval_dir, evaluator):
        eval_output_paths = []
        for i in range(evaluator.outputs_size):
            result = evaluator.get_output_tensor(i).to_numpy()
            eval_output_paths.append((
                os.path.join(eval_dir, f'nncase_result_{i}.bin'),
                os.path.join(eval_dir, f'nncase_result_{i}.txt')))
            result.tofile(eval_output_paths[-1][0])
            self.totxtfile(eval_output_paths[-1][1], result)
        return eval_output_paths