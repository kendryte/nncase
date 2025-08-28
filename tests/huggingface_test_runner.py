from posixpath import join
from typing import Sequence
import shutil
import os
import numpy as np
from numpy.core.defchararray import array
from numpy.lib.function_base import select
from test_runner import *
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import nncase
from npy2json import convert_npy_to_json
from ml_dtypes import bfloat16
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def download_from_huggingface(model_api, tokenizer_api, model_name, need_save=False):
    print(f" Downloading \033[32m\033[1m {model_name} \033[0m from huggingface ... ")
    model_dir = os.path.join(os.path.dirname(__file__), "llm", model_name)
    print(f" model_dir: {model_dir}")
    if os.path.exists(model_dir):
        print(f"\033[32m\033[1m {model_name} \033[0m exits in \033[34m\033[5m {model_dir} \033[0m")
        return model_dir
    else:
        hf_home_env = os.getenv("HF_HOME")
        if hf_home_env is None:
            print(
                f"Please set your huggingface cache dir in environment variable\033[31m 10.10.1.11 'export HF_HOME=/compiler/share/huggingface_cache' \033[0m")
            # download the model from huggingface hub
            model_path = snapshot_download(repo_id=model_name)
        else:
            # if the model can't access in huggingface hub, you can download it from other source and put it in the cache dir ($HF_HOME/hub)
            # e.g.: modelscope download --model LLM-Research/Llama-3.2-1B-Instruct --local_dir $HF_HOME/hub/LLM-Research/Llama-3.2-1B-Instruct
            cache_model_dir = os.path.join(hf_home_env, "hub", model_name)
            if (os.path.exists(cache_model_dir)):
                model_path = cache_model_dir
            else:
                model_path = snapshot_download(repo_id=model_name)

    if need_save:
        try:
            model = model_api.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = tokenizer_api.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            raise os.error(
                f"\033[31m Download {model_name} has error. Make sure it's a valid repository. Or check your network!\033[0m")

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        model_dir = model_path
    print(
        f"\033[32m\033[1m {model_name} \033[0m has been downloaded into \033[34m\033[5m {model_dir} \033[0m")
    return model_dir


def recursive_stack(obj):
    if isinstance(obj, (list, tuple)):
        stacked = [recursive_stack(item) for item in obj]
        if all(isinstance(item, torch.Tensor) for item in stacked):
            return torch.stack(stacked)
        else:
            return stacked
    else:
        # numpy not support bf16 tensor
        if (obj.dtype == torch.bfloat16 or obj.dtype == torch.float16):
            obj = obj.to(torch.float32)
        if (obj.shape[0] != 1):
            return torch.unsqueeze(obj, 0)
        else:
            return obj


def dequantize_weights(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith(".safetensors") and not filename.endswith(".org.safetensors"):
            filepath = os.path.join(model_dir, filename)
            org_filepath = filepath.replace(".safetensors", ".org.safetensors")
            if os.path.exists(os.path.join(model_dir, org_filepath)):
                continue

            if not os.path.exists(org_filepath):
                os.rename(filepath, org_filepath)

            state_dict = load_file(org_filepath)

            for key in list(state_dict.keys()):
                if key.endswith('weight_scale'):
                    scale_tensor = state_dict[key].to(torch.float32)
                    weight_key = key.replace('.weight_scale', '.weight')
                    if weight_key in state_dict:
                        weight_tensor = state_dict[weight_key]
                        if scale_tensor.numel() == 1 or scale_tensor.shape[0] == weight_tensor.shape[0]:
                            weight_fp32 = weight_tensor.to(torch.float32)
                            scaled_weight = weight_fp32 * scale_tensor
                            state_dict[weight_key] = scaled_weight
                        else:
                            raise RuntimeError(
                                f"\033[31m weight_tensor {weight_key} and scale_tensor {key} shape not match! \033[0m")
                    else:
                        print(f"Warning: Corresponding weight {weight_key} not found, skipping.")

            save_file(state_dict, filepath)


def restore_weights(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith(".org.safetensors"):
            org_path = os.path.join(model_dir, filename)
            restored_path = org_path.replace(".org.safetensors", ".safetensors")
            os.rename(org_path, restored_path)
            print(f"Restored: {restored_path}")


def to_np_type(t: str):
    '''
    string to np.type
    '''
    if t == "float32":
        return np.float32
    elif t == "float16":
        return np.float16
    elif t == "bfloat16":
        return bfloat16
    else:
        return None


def dump_data_to_file(dir_path, file_path, data):
    dump_bin_file(os.path.join(dir_path, f'{file_path}.bin'), data)
    dump_txt_file(os.path.join(dir_path, f'{file_path}.txt'), data)
    dump_npy_file(os.path.join(dir_path, f'{file_path}.npy'), data)
    convert_npy_to_json(os.path.join(dir_path, f'{file_path}.npy'), dir_path)


def debug_actual_structure(actual):
    print(f'Type of actual: {type(actual)}')
    print(f'Length of actual: {len(actual)}')
    if actual:
        print(f'Type of actual[0]: {type(actual[0])}')
        if hasattr(actual[0], 'shape'):
            print(f'Shape of actual[0]: {actual[0].shape}')
        elif isinstance(actual[0], list):
            print(f'Length of actual[0]: {len(actual[0])}')
            if actual[0]:
                print(f'Type of actual[0][0]: {type(actual[0][0])}')

class HuggingfaceTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "huggingface"
        self.num_layers = -1
        self.local_inputs: List[Any] = []

    def get_result(self, model):
        results = []
        for idx in range(model.outputs_size):
            results.append(model.get_output_tensor(idx).to_numpy())
        return results

    def hf_eval(self, model, input_data):
        print(f"kv_object.context_lens.to_numpy(): {input_data[1].context_lens.to_runtime_tensor().to_numpy()}")
        print(f"kv_object.seq_lens.to_numpy(): {input_data[1].seq_lens.to_runtime_tensor().to_numpy()}")
        print(f"kv_object.block_tables.to_numpy(): {input_data[1].block_tables.to_runtime_tensor().to_numpy()}")
        print(f"kv_object.slot_mapping.to_numpy(): {input_data[1].slot_mapping.to_runtime_tensor().to_numpy()}")
        
        for idx, i in enumerate(input_data):
            value = None
            if isinstance(i, nncase._nncase.RefPagedAttentionKVCache):
                value = i.as_ivalue()
            else:
                value = nncase.RuntimeTensor.from_numpy(i['data'][0])
            model.set_input_tensor(idx, value)

        model.run()
        return self.get_result(model)

    def hf_infer(self, model, input_data):
        print(f"kv_object.context_lens.to_numpy(): {input_data[1].context_lens.to_numpy()}")
        print(f"kv_object.seq_lens.to_numpy(): {input_data[1].seq_lens.to_numpy()}")
        print(f"kv_object.block_tables.to_numpy(): {input_data[1].block_tables.to_numpy()}")
        print(f"kv_object.slot_mapping.to_numpy(): {input_data[1].slot_mapping.to_numpy()}")
        for idx, value in enumerate(input_data):
            new_data = None
            if isinstance(value, nncase.PagedAttentionKVCache):
                new_data = nncase.RuntimeTensor.from_object(value)
            else:
                new_data = nncase.RuntimeTensor.from_numpy(np.array(value['data'][0], dtype=np.int64))
            model.set_input_tensor(idx, new_data)
        model.run()
        return self.get_result(model)

    def pipeline_run(self, model, infer_or_eval):
        import copy
        input_data = self.local_inputs
        data = copy.deepcopy(input_data[0])
        loop_data = [data, input_data[1]]
        res = []
        for i in range(self.cfg['huggingface_options']['max_tokens']):
            result = np.array([0])
            print(loop_data[0])
            current_length = loop_data[0]['data'][0].shape[-1]
            print(f"current_length: {current_length}")
            kv_object = loop_data[1]['scheduler'].schedule([0], [current_length])
            if infer_or_eval == "infer":
                result = self.hf_infer(model, [loop_data[0], kv_object])
            else:
                result = self.hf_eval(model, [loop_data[0], kv_object])
            new_token = np.argmax(result[0][-1, :], axis=-1)
            res.append(new_token)
            loop_data[0]['data'] = [np.array([new_token], dtype=np.int64)]

        return np.array([res], dtype=np.int64)
    
    def prefill_run(self, model, infer_or_eval):
        input_data = self.local_inputs
        data = copy.deepcopy(input_data[0])
        loop_data = [data, input_data[1]]
        result = np.array([0])
        current_length = loop_data[0]['data'][0].shape[-1]
        kv_object = loop_data[1]['scheduler'].schedule([0], [current_length])
        if infer_or_eval == "infer":
            result = self.hf_infer(model, [loop_data[0], kv_object])
        else:
            result = self.hf_eval(model, [loop_data[0], kv_object])

        return result

    def from_huggingface(self, model_path):
        pass
    
    def huggingface_run(self, func, model_file, judge_type):
        if not self.inputs:
            self.parse_model(model_file)

        self.generate_all_data()
        self.local_inputs = [self.inputs[0], self.inputs[1]]
        self.write_compile_opt()
        expected = self.cpu_infer(model_file)
        targets = self.cfg['target']
        model_content = self.read_model_file(model_file)
        import_options = self.get_import_options()

        compiler = None
        dump_hist = self.cfg['dump_hist']
        for k_target, v_target in targets.items():
            tmp_dir = os.path.join(self.case_dir, 'tmp')
            if v_target['eval'] or v_target['infer']:
                compile_options = self.get_compile_options(k_target, model_file, tmp_dir)
                compile_options.target_options = self.get_target_options(
                    k_target, v_target.get("target_options", None))
                compiler = nncase.Compiler(compile_options)
                self.import_model(compiler, model_content, import_options)

            for stage in ['eval', 'infer']:
                if v_target[stage]:
                    for k_mode, v_mode in v_target['mode'].items():
                        if v_mode['enabled']:
                            os.makedirs(tmp_dir, exist_ok=True)
                            if stage == 'eval':
                                self.local_inputs = [self.inputs[0], self.inputs[1]]
                                evaluator = compiler.create_evaluator(3)
                                actual = func(evaluator, "eval")
                            else:
                                self.local_inputs = [self.inputs[0], self.inputs[2]]
                                compiler.compile()
                                kmodel_path = os.path.join(tmp_dir, self.cfg['kmodel_name'])
                                with open(kmodel_path, 'wb') as f:
                                    compiler.gencode(f)
                                sim = nncase.Simulator()
                                with open(kmodel_path, 'rb') as f:
                                    sim.load_model(f)
                                
                                actual = func(sim, "infer")

                            # debug_actual_structure(actual)
                            # print("----------")

                            # debug_actual_structure(expected)
                            print(actual)
                            target_dir = os.path.join(self.case_dir, stage, k_target)
                            os.makedirs(target_dir, exist_ok=True)
                            mode_dir = os.path.join(target_dir, k_mode)
                            shutil.move(tmp_dir, mode_dir)

                            judge, result = self.compare_results(
                                np.array(expected), np.array(actual), stage, k_target, judge_type, k_mode, v_mode['threshold'], dump_hist, mode_dir)

                            
                            if not judge:
                                if test_utils.in_ci():
                                    self.clear(self.case_dir)
                                # assert (judge), f"Fault result in {stage} + {result}"

        if test_utils.in_ci():
            self.clear(self.case_dir)

    
    def run(self, model_file):
        if self.cfg['huggingface_options']['pipeline']:
            self.huggingface_run(self.pipeline_run, model_file, "LLM")
        else:
            self.huggingface_run(self.prefill_run, model_file, "cosine")
        

    def cpu_infer(self, model_file: List[str]):
        self.local_inputs = [self.inputs[0]]
        outputs = []
        for idx, input in enumerate(self.local_inputs):
            # TODO: add attention_mask in inputs
            if self.cfg['huggingface_options']['pipeline']:
                result = self.model.generate(
                    input_ids=torch.from_numpy(np.expand_dims(input['data'][0], 0)),
                    generation_config=self.generation_config)
                res = result[:, input['data'][0].shape[-1]:]
                # print(self.tokenizer.batch_decode(res))
                print(res)
                outputs.append(res.numpy())
                return outputs
            else:
                result = self.model.forward(
                    torch.from_numpy(np.expand_dims(input['data'][0], 0)),
                    return_dict=True,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=(True if self.cfg['huggingface_options']['output_hidden_states']
                                        else False) if self.cfg['huggingface_options']['output_logits'] else True
                )

                count = 0
                if (self.cfg['huggingface_options']['output_logits']):
                    if not test_utils.in_ci():
                        logits = result.logits.detach().to(torch.float32).numpy()[0]
                        dump_data_to_file(self.case_dir, f'cpu_result_{count}', logits)
                        outputs.append(logits)
                        count += 1
                else:
                    if not test_utils.in_ci():
                        hidden_states = recursive_stack(result.hidden_states).detach().to(
                            torch.float32).numpy()[-1][0]
                        dump_data_to_file(self.case_dir, f'cpu_result_{count}', hidden_states)
                        outputs.append(hidden_states)
                        count += 1

                if (self.cfg['huggingface_options']['output_hidden_states']):
                    if not test_utils.in_ci():
                        hidden_states = recursive_stack(result.hidden_states).detach().numpy()
                        hidden_states = np.squeeze(hidden_states, 1)
                        dump_data_to_file(self.case_dir, f'cpu_result_{count}', hidden_states)
                        outputs.append(hidden_states)
                        count += 1

        return outputs

    def parse_model(self, model_path):
        if self.cfg['huggingface_options']['pipeline']:
            if self.cfg['huggingface_options']['output_logits'] == False:
                raise RuntimeError("output_logits must be `True` in pipeline mode")
        
        config = AutoConfig.from_pretrained(model_path + "/config.json")

        if self.cfg['huggingface_options']['num_layers'] != -1:
            self.num_layers = self.cfg['huggingface_options']['num_layers']
            config.num_hidden_layers = self.num_layers
        else:
            self.num_layers = config.num_hidden_layers

        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if hasattr(
            config, "head_dim") else config.hidden_size // config.num_attention_heads

        paged_attention_config = self.cfg['paged_attention_config']

        self.block_size = paged_attention_config['block_size']
        self.num_blocks = paged_attention_config['num_blocks']
        self.max_sessions = paged_attention_config['max_sessions']
        self.max_model_len = (self.block_size * self.num_blocks) // self.max_sessions
        self.kv_type = np.dtype(to_np_type(paged_attention_config['kv_type']))
        self.cache_layout = [getattr(nncase.PagedKVCacheDimKind, item)
                             for item in paged_attention_config['cache_layout']]
        # [ nncase.PagedKVCacheDimKind.it for it in paged_attention_config['cache_layout'] ]
        self.vectorized_axes = [getattr(nncase.PagedKVCacheDimKind, item)
                                for item in paged_attention_config['vectorized_axes']]
        self.lanes = paged_attention_config['lanes']
        self.sharding_axes = [getattr(nncase.PagedKVCacheDimKind, item)
                              for item in paged_attention_config['sharding_axes']]
        self.axis_policies = paged_attention_config['axis_policies']
        self.hierarchy = paged_attention_config['hierarchy']

        self.kv_cache_config = nncase.PagedAttentionConfig(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.kv_type,
            self.block_size,
            self.cache_layout,
            self.vectorized_axes,
            self.lanes,
            self.sharding_axes,
            self.axis_policies
        )

        self.cfg['huggingface_options']['config'] = self.kv_cache_config

        # if hasattr(config, "quantization_config"):
        #     dequantize_weights(model_path)
        #     delattr(config, "quantization_config")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype="auto", device_map="auto", trust_remote_code=True).eval()
        # restore_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = self.model.generation_config
        # self.generation_config.return_dict_in_generate = True # if False, generate only output tokens
        self.generation_config.max_new_tokens = self.cfg['huggingface_options']['max_tokens']
        self.generation_config.do_sample = False
        self.generation_config.temperature = 0.0  # for Stable result
        if (self.cfg['huggingface_options']['output_logits']):
            pass
        else:
            self.generation_config.output_hidden_states = True
        if (self.cfg['huggingface_options']['output_hidden_states']):
            self.generation_config.output_hidden_states = True

        input_dict = {}
        for input_ in self.model.dummy_inputs:
            input_dict["name"] = input_
            input_dict["dtype"] = self.model.dummy_inputs[input_].dtype.__repr__().split('.')[1]
            # TODO: fix dynamic shape
            input_dict['shape'] = [1, "sequence_length"]
            input_dict['model_shape'] = [1, "sequence_length"]
        self.inputs.append(input_dict)
        self.calibs.append(copy.deepcopy(input_dict))

        input_scheduler_eval = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler_eval = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)

        self.inputs.append(dict(name='kv_cache_eval', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=input_scheduler_eval))
        self.calibs.append(dict(name='kv_cache_eval', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=calibs_scheduler_eval))

        input_scheduler = nncase.PagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler = nncase.PagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)

        self.inputs.append(dict(name='kv_cache', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=input_scheduler))
        self.calibs.append(dict(name='kv_cache', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=calibs_scheduler))

    def import_model(self, compiler, model_content, import_options):
        compiler.import_huggingface(model_content, import_options)
