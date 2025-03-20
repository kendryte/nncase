from posixpath import join
from typing import Sequence
import shutil
import os
import numpy as np
from numpy.core.defchararray import array
from numpy.lib.function_base import select
from test_runner import *
import io
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HuggingfaceTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "huggingface"

    def from_huggingface(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_config = self.model.generation_config
        # generation_config.max_new_tokens = 1
        self.generation_config.do_sample = True

    def run(self, model_dir):
        super().run(model_dir)

    def cpu_infer(self, model_file: List[str]):
        outputs = []
        for idx, input in enumerate(self.inputs):
            messages = [
                {"role": "system", "content": "You are a assistant!"},
                {"role": "user", "content": input}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            '''
            {
                'input_ids': tensor([[151644, 8948, ... 198, 151644, 77091, 198]]),
                'attention_mask': tensor([[1, 1, 1, 1, ..., 1, 1]])
            }
            '''
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128
            )
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(
                    self.case_dir, f'cpu_result_{idx}.bin'), generated_ids.logits)
                dump_txt_file(os.path.join(
                    self.case_dir, f'cpu_result_{idx}.txt'), generated_ids.logits)
            generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):-1]

            output = self.tokenizer.decode(generated_ids)
            outputs.append(output)
        return outputs

    def parse_model(self, model_path):
        pass

    def import_model(self, compiler, model_content, import_options):
        compiler.import_huggingface(model_content, import_options)
