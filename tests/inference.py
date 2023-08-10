from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils
import preprocess_utils
import socket
import json
from test_utils import *


class Inference:
    def run_inference(self, compiler, target, ptq_enabled, infer_dir):
        in_ci = test_utils.in_ci()
        kpu_targets = test_utils.kpu_targets()
        nuc_ip = test_utils.nuc_ip()
        nuc_port = test_utils.nuc_port()
        test_executable = test_utils.test_executable(target)
        running_on_evb = in_ci and target in kpu_targets and nuc_ip is not None and nuc_port is not None and test_executable is not None and len(
            self.inputs) > 0 and len(self.outputs) > 0

        if ptq_enabled:
            self.set_quant_opt(compiler)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        os.makedirs(infer_dir, exist_ok=True)
        if not in_ci:
            with open(os.path.join(infer_dir, 'test.kmodel'), 'wb') as f:
                f.write(kmodel)

        compile_opt = self.cfg['compile_opt']
        if running_on_evb:
            outputs = self.run_evb(target, kmodel, compile_opt)
        else:
            sim = nncase.Simulator()
            sim.load_model(kmodel)
            self.set_infer_input(sim, compile_opt)
            sim.run()
            outputs = self.dump_infer_output(sim, compile_opt, infer_dir)
        return outputs

    def set_infer_input(self, sim, compile_opt):
        for idx, value in enumerate(self.inputs):
            data = self.transform_input(
                value['data'], compile_opt['input_type'], "infer")[0]
            dtype = compile_opt['input_type']
            if compile_opt['preprocess'] and dtype != 'float32':
                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(self.case_dir, f'input_{idx}_{dtype}.bin'), data)
                    dump_txt_file(os.path.join(self.case_dir, f'input_{idx}_{dtype}.txt'), data)

            sim.set_input_tensor(idx, nncase.RuntimeTensor.from_numpy(data))

    def dump_infer_output(self, sim, compile_opt, infer_dir):
        outputs = []
        for i in range(sim.outputs_size):
            output = sim.get_output_tensor(i).to_numpy()
            if compile_opt['preprocess']:
                if(compile_opt['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                    output = np.transpose(output, [0, 3, 1, 2])
                elif (compile_opt['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                    output = np.transpose(output, [0, 2, 3, 1])
                elif compile_opt['output_layout'] not in ["NCHW", "NHWC"] and compile_opt['output_layout'] != "":
                    tmp_perm = [int(idx) for idx in compile_opt['output_layout'].split(",")]
                    output = np.transpose(
                        output, preprocess_utils.get_source_transpose_index(tmp_perm))
            outputs.append(output)
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(infer_dir, f'nncase_result_{i}.bin'), output)
                dump_txt_file(os.path.join(infer_dir, f'nncase_result_{i}.txt'), output)
        return outputs

    def run_evb(self, target, kmodel, compile_opt):
        ip = test_utils.nuc_ip()
        port = test_utils.nuc_port()
        test_executable = test_utils.test_executable(target)

        # connect server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, int(port)))

        # send target
        dummy = client_socket.recv(1024)
        target_dict = {}
        target_dict['target'] = target
        client_socket.sendall(json.dumps(target_dict).encode())

        # send header
        dummy = client_socket.recv(1024)
        header_dict = {}
        header_dict['case'] = os.path.basename(self.case_dir)
        header_dict['app'] = 1
        header_dict['kmodel'] = 1
        header_dict['inputs'] = len(self.inputs)
        header_dict['outputs'] = len(self.outputs)
        client_socket.sendall(json.dumps(header_dict).encode())

        # send app
        dummy = client_socket.recv(1024)
        file_dict = {}
        file_dict['file_name'] = os.path.basename(test_executable)
        file_dict['file_size'] = os.path.getsize(test_executable)
        client_socket.sendall(json.dumps(file_dict).encode())
        dummy = client_socket.recv(1024)
        with open(test_executable, 'rb') as f:
            client_socket.sendall(f.read())

        # send kmodel
        dummy = client_socket.recv(1024)
        file_dict['file_name'] = 'test.kmodel'
        file_dict['file_size'] = len(kmodel)
        client_socket.sendall(json.dumps(file_dict).encode())
        dummy = client_socket.recv(1024)
        client_socket.sendall(kmodel)

        # send inputs
        for idx, value in enumerate(self.inputs):
            data = self.transform_input(
                value['data'], compile_opt['input_type'], "infer")[0]
            file_dict['file_name'] = f'input_{idx}.bin'
            file_dict['file_size'] = data.size * data.itemsize
            dummy = client_socket.recv(1024)
            client_socket.sendall(json.dumps(file_dict).encode())
            dummy = client_socket.recv(1024)
            client_socket.sendall(data.tobytes())

        # get infer result
        outputs = []
        cmd_result = client_socket.recv(1024).decode()
        if cmd_result.find('finish') != -1:
            client_socket.sendall(f"pls send outputs".encode())

            # recv outputs
            for i in range(len(self.outputs)):
                header = client_socket.recv(1024)
                file_size = int(header.decode())
                client_socket.sendall(f"pls send nncase_result_{i}.bin".encode())

                recv_size = 0
                buffer = bytearray(file_size)
                while recv_size < file_size:
                    slice = client_socket.recv(4096)
                    buffer[recv_size:] = slice
                    recv_size += len(slice)

                output = np.frombuffer(buffer, dtype=self.outputs[i]['dtype'])
                outputs.append(output)
                client_socket.sendall(f"recv nncase_result_{i}.bin succeed".encode())

            client_socket.close()
        else:
            client_socket.close()
            raise Exception(f'{cmd_result}')

        return outputs
