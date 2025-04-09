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
                f"Please set your huggingface cache dir in environment variable\033[31m 10.10.1.11 'export HF_HOME=/data/huggingface_cache' \033[0m")

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
    org_safetensors = model_dir + "/model_org.safetensors"
    f32_safetensors = model_dir + "/model.safetensors"
    if not os.path.exists(org_safetensors):
        os.rename(f32_safetensors, org_safetensors)
    state_dict = load_file(org_safetensors)
    
    for key in list(state_dict.keys()):
        if key.endswith('weight_scale'):
            scale_tensor = state_dict[key]
            weight_key = key.replace('.weight_scale', '.weight')
            if weight_key in state_dict:
                weight_tensor = state_dict[weight_key]
                if scale_tensor.numel() == 1:
                    scale = scale_tensor.item()
                    weight_fp32 = weight_tensor.to(torch.float32)
                    scaled_weight = weight_fp32 * scale
                    state_dict[weight_key] = scaled_weight.to(torch.float16)
                else:
                    print(f"Warning: {key} is not a single-element tensor, skipping.")
            else:
                print(f"Warning: Corresponding weight {weight_key} not found, skipping.")

    save_file(state_dict, f32_safetensors)

def restore_weights(model_dir):
    org_safetensors = model_dir + "/model_org.safetensors"
    f32_safetensors = model_dir + "/model.safetensors"
    if os.path.exists(org_safetensors):
        os.rename(org_safetensors, f32_safetensors)

class HuggingfaceTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "huggingface"

    def from_huggingface(self, model_path):
        pass

    def run(self, model_dir):
        super().run(model_dir)

    def cpu_infer(self, model_file: List[str]):
        outputs = []
        for idx, input in enumerate(self.inputs):
            '''
            {
                'input_ids': tensor([[151644, 8948, ... 198, 151644, 77091, 198]]),
                'attention_mask': tensor([[1, 1, 1, 1, ..., 1, 1]])
            }
            '''
            # messages = [
            #     {"role": "system", "content": "You are a assistant!"},
            #     {"role": "user", "content": input}
            # ]
            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(self.case_dir, "input",
                                           f'input_{idx}.bin'), input['data'][idx])
                dump_txt_file(os.path.join(self.case_dir, "input",
                                           f'input_{idx}.txt'), input['data'][idx])

            # TODO: add attention_mask in inputs
            result = self.model.forward(
                torch.from_numpy(input['data'][0]),
                return_dict=True,
                use_cache=self.cfg['huggingface_options']['use_cache'],
                output_attentions=self.cfg['huggingface_options']['output_attentions'],
                output_hidden_states=self.cfg['huggingface_options']['output_hidden_states'],
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
            if not test_utils.in_ci():
                logits = result.logits.detach().numpy()
                # data = np.argmax(logits, 2).flatten()
                # print(data)
                # print(self.tokenizer.decode(data, skip_special_tokens=False))
                dump_bin_file(os.path.join(self.case_dir, f'cpu_result_{count}.bin'), logits)
                dump_txt_file(os.path.join(self.case_dir, f'cpu_result_{count}.txt'), logits)
                dump_npy_file(os.path.join(self.case_dir, f'cpu_result_{count}.npy'), logits)
                outputs.append(logits)
                count += 1
            if (self.cfg['huggingface_options']['use_cache']):
                if not test_utils.in_ci():
                    from transformers import DynamicCache
                    if (isinstance(result.past_key_values, DynamicCache)):
                        k = recursive_stack(result.past_key_values.key_cache)
                        v = recursive_stack(result.past_key_values.value_cache)
                        past_kv = torch.stack([k, v], 1).detach().numpy()
                    else:
                        past_kv = recursive_stack(result.past_key_values).detach().numpy()
                    dump_bin_file(os.path.join(self.case_dir, f'cpu_result_{count}.bin'), past_kv)
                    dump_txt_file(os.path.join(self.case_dir, f'cpu_result_{count}.txt'), past_kv)
                    dump_npy_file(os.path.join(self.case_dir, f'cpu_result_{count}.npy'), past_kv)
                    outputs.append(past_kv)
                    count += 1
            if (self.cfg['huggingface_options']['output_attentions']):
                if not test_utils.in_ci():
                    attentions = recursive_stack(result.attentions).detach().numpy()
                    dump_bin_file(os.path.join(
                        self.case_dir, f'cpu_result_{count}.bin'), attentions)
                    dump_txt_file(os.path.join(
                        self.case_dir, f'cpu_result_{count}.txt'), attentions)
                    dump_npy_file(os.path.join(
                        self.case_dir, f'cpu_result_{count}.npy'), attentions)
                    outputs.append(attentions)
                    count += 1
            if (self.cfg['huggingface_options']['output_hidden_states']):
                if not test_utils.in_ci():
                    hidden_states = recursive_stack(result.hidden_states).detach().numpy()
                    dump_bin_file(os.path.join(
                        self.case_dir, f'cpu_result_{count}.bin'), hidden_states)
                    dump_txt_file(os.path.join(
                        self.case_dir, f'cpu_result_{count}.txt'), hidden_states)
                    dump_npy_file(os.path.join(
                        self.case_dir, f'cpu_result_{count}.npy'), hidden_states)
                    outputs.append(hidden_states)
                    count += 1

        return outputs

    def parse_model(self, model_path):
        config = AutoConfig.from_pretrained(model_path + "/config.json")
        if hasattr(config, "quantization_config"):
            dequantize_weights(model_path)
            delattr(config, "quantization_config") 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype="auto", device_map="auto", trust_remote_code=True).to(torch.float32).eval()
        restore_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = self.model.generation_config
        # self.generation_config.return_dict_in_generate = True # if False, generate only output tokens
        self.generation_config.max_new_tokens = 64
        self.generation_config.do_sample = False
        self.generation_config.temperature = 0.0  # for Stable result
        if (self.cfg['huggingface_options']['output_attentions']):
            self.generation_config.output_attentions = True
        if (self.cfg['huggingface_options']['output_hidden_states']):
            self.generation_config.output_hidden_states = True
        if (self.cfg['huggingface_options']['use_cache']):
            self.generation_config.use_cache = True

        input_dict = {}
        for input_ in self.model.dummy_inputs:
            input_dict["name"] = input_
            input_dict["dtype"] = self.model.dummy_inputs[input_].dtype.__repr__().split('.')[1]
            # TODO: fix dynamic shape
            input_dict['shape'] = [1, "sequence_length"]
            input_dict['model_shape'] = [1, "sequence_length"]
        self.inputs.append(input_dict)
        self.calibs.append(copy.deepcopy(input_dict))

    def import_model(self, compiler, model_content, import_options):
        compiler.import_huggingface(model_content, import_options)
