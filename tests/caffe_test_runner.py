import caffe
from test_runner import *
import os
import shutil
import numpy as np
# from typing import Dict, List, Tuple, Union


class CaffeTestRunner(TestRunner):
    def __init__(self, case_name, targets=None):
        super().__init__(case_name, targets)

    def run(self, model_file_list):
        super().run(model_file_list)

    def parse_model_input_output(self, model_path: Union[List[str], str]):
        caffe_model = caffe.Net(model_path[0], model_path[1], caffe.TEST)
        for i, name in enumerate(caffe_model._layer_names):
            if (caffe_model.layers[i].type == "Input"):
                input_dict = {}
                input_dict['name'] = name
                input_dict['dtype'] = np.float32
                input_dict['shape'] = list(caffe_model.blobs[name].data.shape)
                self.inputs.append(input_dict)
                self.calibs.append(input_dict.copy())

    def cpu_infer(self, case_dir: str, model_file_list):
        caffe_model = caffe.Net(model_file_list[0], model_file_list[1], caffe.TEST)

        data = [(k, v.data.shape) for k, v in caffe_model.blobs.items()]
        for i, name in enumerate(caffe_model._layer_names):
            if (caffe_model.layers[i].type == "Input"):
                caffe_model.blobs[name].data[...] = self.inputs[i]['data']

        outputs = caffe_model.forward()

        for i in range(0, len(outputs)):
            result = outputs[data[-1 - i][0]]

            self.output_paths.append((
                os.path.join(case_dir, f'cpu_result_{i}.bin'),
                os.path.join(case_dir, f'cpu_result_{i}.txt')))
            result.tofile(self.output_paths[-1][0])
            self.totxtfile(self.output_paths[-1][1], result)

    def import_model(self, compiler, model_content, import_options):
        compiler.import_caffe(model_content[1], model_content[0])
