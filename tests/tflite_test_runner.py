import tensorflow as tf
from test_runner import *
import os
import shutil


class TfliteTestRunner(TestRunner):
    def __init__(self, case_name, targets=None, overwrite_configs: dict = None):
        super().__init__(case_name, targets, overwrite_configs)
        self.model_type = "tflite"

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
        elif model_file.startswith('tflite-models'):
            model_file = os.path.join(os.getenv('TFLITE_MODELS_DIR'),
                                      model_file[len('tflite-models/'):])
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
            input_dict['model_shape'] = item['shape']
            self.inputs.append(input_dict)
            self.calibs.append(copy.deepcopy(input_dict))
            self.dump_range_data.append(copy.deepcopy(input_dict))

        for item in interp.get_output_details():
            output_dict = {}
            output_dict['index'] = item['index']
            output_dict['name'] = item['name']
            output_dict['dtype'] = item['dtype']
            output_dict['model_shape'] = item['shape']
            self.outputs.append(output_dict)

    def cpu_infer(self, case_dir: str, model_file: bytes, type: str, mode: str):
        interp = tf.lite.Interpreter(model_path=model_file)
        interp.allocate_tensors()
        if mode is "dataset":
            topk = []
            for input in self.inputs:
                for in_data in input['data']:
                    interp.set_tensor(input["index"], self.data_pre_process(in_data[0]))
                    interp.invoke()
                    for output in self.outputs:
                        data = interp.get_tensor(output['index'])
                        topk.append((in_data[1], get_topK('cpu', 1, data)[0]))
                if os.path.exists(os.path.join(case_dir, "cpu_dataset.txt")):
                    os.remove(os.path.join(case_dir, "cpu_dataset.txt"))
                self.output_paths.append((
                    os.path.join(case_dir, 'cpu_result.bin'),
                    os.path.join(case_dir, 'cpu_result.txt')))
                with open(self.output_paths[-1][1], 'a') as f:
                    for i in range(len(topk)):
                        f.write(topk[i][0].split('/')[-1] + " " + str(topk[i][1]) + '\n')
        else:
            for input in self.inputs:
                interp.set_tensor(input["index"], self.data_pre_process(input['data']))

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

    def import_model(self, compiler, model_content, import_options):
        compiler.import_tflite(model_content, import_options)
