import tensorflow as tf
from test_runner import *
import os
import shutil


class TfliteTestRunner(TestRunner):
    def __init__(self, case_name, targets=None, overwirte_configs: dict = None):
        super().__init__(case_name, targets, overwirte_configs)

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
        if self.case_dir != os.path.dirname(model_file):
            shutil.copy(model_file, self.case_dir)
            model_file = os.path.join(
                self.case_dir, os.path.basename(model_file))

        super().run(model_file)

    def parse_model_input_output(self, model_path: str):
        interp = tf.lite.Interpreter(model_path=model_path)

        for item in interp.get_input_details():
            input_dict = {}
            input_dict['index'] = item['index']
            input_dict['name'] = item['name']
            input_dict['dtype'] = item['dtype']
            input_dict['shape'] = item['shape']
            self.inputs.append(input_dict)
            self.calibs.append(input_dict.copy())

        for item in interp.get_output_details():
            output_dict = {}
            output_dict['index'] = item['index']
            output_dict['name'] = item['name']
            output_dict['dtype'] = item['dtype']
            output_dict['shape'] = item['shape']
            self.outputs.append(output_dict)

    def cpu_infer(self, case_dir: str, model_file: bytes):
        interp = tf.lite.Interpreter(model_path=model_file)
        interp.allocate_tensors()
        for input in self.inputs:
            interp.set_tensor(input["index"], input['data'])

        interp.invoke()

        i = 0
        for output in self.outputs:
            data = interp.get_tensor(output['index'])
            self.output_paths.append((
                os.path.join(case_dir, f'cpu_result_{i}.bin'),
                os.path.join(case_dir, f'cpu_result_{i}.txt')))
            data.tofile(self.output_paths[-1][0])
            self.totxtfile(self.output_paths[-1][1], data)
            i += 1
