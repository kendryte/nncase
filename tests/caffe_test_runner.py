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

        for input in self.inputs:
            caffe_model.blobs[input['name']].data[...] = self.transform_input(
                self.data_pre_process(input['data']), "float32", "CPU")[0]

        outputs = caffe_model.forward()

        results = []
        for i, output in enumerate(self.outputs):
            result = outputs[output['name']]
            results.append(result)
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(self.case_dir, f'cpu_result_{i}.bin'), result)
                dump_txt_file(os.path.join(self.case_dir, f'cpu_result_{i}.txt'), result)

        return results

    def import_model(self, compiler, model_content, import_options):
        compiler.import_caffe(model_content[1], model_content[0])
