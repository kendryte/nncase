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
    else:
        return None


def dump_data_to_file(dir_path, file_path, data):
    dump_bin_file(os.path.join(dir_path, f'{file_path}.bin'), data)
    dump_txt_file(os.path.join(dir_path, f'{file_path}.txt'), data)
    dump_npy_file(os.path.join(dir_path, f'{file_path}.npy'), data)
    convert_npy_to_json(os.path.join(dir_path, f'{file_path}.npy'), dir_path)


class HuggingfaceTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "huggingface"
        self.num_layers = -1

    def from_huggingface(self, model_path):
        pass

    def run(self, model_dir):
        super().run(model_dir)

    def cpu_infer(self, model_file: List[str]):
        outputs = []
        for idx, input in enumerate(self.inputs):
            if idx != 0:
                continue

            # TODO: add attention_mask in inputs
            result = self.model.forward(
                torch.from_numpy(np.expand_dims(input['data'][0], 0)),
                return_dict=True,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=(True if self.cfg['huggingface_options']['output_hidden_states']
                                      else False) if self.cfg['huggingface_options']['output_logits'] else True
            )

            ''' will be used in future[pipeline run]
            # logits = self.model.generate(
            #     torch.from_numpy(input['data'][0]),
            #     generation_config=self.generation_config,
            # )
            # generated_ids = generated_ids[0][input['data'][0].shape[-1]:-1]
            # output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            '''
            count = 0
            if (self.cfg['huggingface_options']['output_logits']):
                if not test_utils.in_ci():
                    logits = result.logits.detach().numpy()[0]
                    dump_data_to_file(self.case_dir, f'cpu_result_{count}', logits)
                    outputs.append(logits)
                    count += 1
            else:
                if not test_utils.in_ci():
                    hidden_states = recursive_stack(result.hidden_states).detach().numpy()[-1][0]
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

        if hasattr(config, "quantization_config"):
            dequantize_weights(model_path)
            delattr(config, "quantization_config")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype="auto", device_map="cpu", trust_remote_code=True).to(torch.float32).eval()
        restore_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = self.model.generation_config
        # self.generation_config.return_dict_in_generate = True # if False, generate only output tokens
        self.generation_config.max_new_tokens = 64
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

        input_scheduler = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)

        self.inputs.append(dict(name='kv_cache', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=input_scheduler))
        self.calibs.append(dict(name='kv_cache', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=calibs_scheduler))

    def import_model(self, compiler, model_content, import_options):
        compiler.import_huggingface(model_content, import_options)
