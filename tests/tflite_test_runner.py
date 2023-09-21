import tensorflow as tf
from test_runner import *
import os
import shutil
from test_utils import *


class TfliteTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "tflite"
        self.interp = None

    def from_tensorflow(self, module):
        # export model
        tf.saved_model.save(module, self.case_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(self.case_dir)

        # convert model
        tflite_model = converter.convert()
        model_file = os.path.join(self.case_dir, 'test.tflite')
        with open(model_file, 'wb') as f:
            f.write(tflite_model)

        return model_file

    def from_keras(self, keras_model):
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        model_file = os.path.join(self.case_dir, 'test.tflite')
        with open(model_file, 'wb') as f:
            f.write(tflite_model)

        return model_file

    def run(self, model_file):
        if model_file.startswith('examples'):
            model_file = os.path.join(os.path.dirname(__file__), '..', model_file)
        if self.case_dir != os.path.dirname(model_file):
            shutil.copy(model_file, self.case_dir)
            model_file = os.path.join(
                self.case_dir, os.path.basename(model_file))
        super().run(model_file)

    def parse_model(self, model_path: str):
        self.interp = tf.lite.Interpreter(model_path=model_path)

        def translate_shape(shape):
            return [i if i > 0 else self.shape_vars["-1"] for i in shape]

        for item in self.interp.get_input_details():
            input_dict = {}
            input_dict['index'] = item['index']
            input_dict['name'] = item['name']
            input_dict['dtype'] = item['dtype']

            if len(self.shape_vars) == 0:
                # fixed shape
                shape = item['shape']
            else:
                # dynamic shape
                shape = item['shape_signature']

            if len(shape) <= 4:
                input_dict['model_shape'] = translate_shape(shape)
            else:
                if -1 in shape:
                    raise "tflite test_runner not supported dynamic shape which rank > 4"
                input_dict['model_shape'] = shape

            if -1 in shape:
                self.dynamic = True
                self.interp.resize_tensor_input(item['index'], input_dict['model_shape'])
            self.inputs.append(input_dict)
            self.calibs.append(copy.deepcopy(input_dict))

        for item in self.interp.get_output_details():
            output_dict = {}
            output_dict['index'] = item['index']
            output_dict['name'] = item['name']
            output_dict['dtype'] = item['dtype']
            output_dict['model_shape'] = item['shape']
            self.outputs.append(output_dict)

    def cpu_infer(self, model_file: bytes):
        self.interp.allocate_tensors()
        for idx, value in enumerate(self.inputs):
            new_value = self.transform_input(
                self.data_pre_process(value['data']), "float32", "CPU")[0]
            self.interp.set_tensor(value["index"], new_value)
            if self.cfg['compile_opt']['preprocess'] and not test_utils.in_ci():
                dump_bin_file(os.path.join(self.case_dir, f'frame_input_{idx}.bin'), new_value)
                dump_txt_file(os.path.join(self.case_dir, f'frame_input_{idx}.txt'), new_value)

        self.interp.invoke()

        i = 0
        results = []
        for output in self.outputs:
            data = self.interp.get_tensor(output['index'])
            results.append(data)
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(self.case_dir, f'cpu_result_{i}.bin'), data)
                dump_txt_file(os.path.join(self.case_dir, f'cpu_result_{i}.txt'), data)
            i += 1

        return results

    def import_model(self, compiler, model_content, import_options):
        compiler.import_tflite(model_content, import_options)
