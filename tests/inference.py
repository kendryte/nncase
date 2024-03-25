# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils
import preprocess_utils
import socket
import struct
import json
from test_utils import *
import time
from html import escape


def data_shape_list_string(data):
    return '\n'.join(map(lambda d: ' '.join(map(lambda x: str(x), d['model_shape'])), data))


class Inference:
    def run_inference(self, compiler, target, ptq_enabled, infer_dir):
        in_ci = test_utils.in_ci()
        kpu_targets = test_utils.kpu_targets()
        nuc_ip = test_utils.nuc_ip()
        nuc_port = test_utils.nuc_port()
        test_executable = test_utils.test_executable(target)
        running_on_evb = target in kpu_targets and nuc_ip is not None and nuc_port is not None and test_executable is not None and len(
            self.inputs) > 0 and len(self.outputs) > 0

        if self.cfg['infer_report_opt']['enabled']:
            self.infer_report_dict['priority'] = self.cfg['infer_report_opt']['priority']
            self.infer_report_dict['kind'] = self.cfg['infer_report_opt']['kind']
            self.infer_report_dict['model'] = self.cfg['infer_report_opt']['model_name']
            self.infer_report_dict['shape'] = ',<br/>'.join(
                map(lambda d: '[' + ','.join(map(lambda x: str(x), d['model_shape'])) + ']', self.inputs))
        if ptq_enabled:
            self.set_quant_opt(compiler)

            if self.cfg['infer_report_opt']['enabled']:
                case = os.path.basename(self.case_dir)
                self.infer_report_dict['if_quant_type'] = self.cfg['ptq_opt']['quant_type']
                self.infer_report_dict['w_quant_type'] = self.cfg['ptq_opt']['w_quant_type']

        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        os.makedirs(infer_dir, exist_ok=True)
        if self.dynamic:
            self.dump_kmodel_desc(os.path.join(infer_dir, self.cfg['desc_name']))
        if not in_ci:
            with open(os.path.join(infer_dir, self.cfg['kmodel_name']), 'wb') as f:
                f.write(kmodel)

        compile_opt = self.cfg['compile_opt']
        if running_on_evb:
            outputs = self.run_evb(target, kmodel, compile_opt, infer_dir)
        else:
            sim = nncase.Simulator()
            sim.load_model(kmodel)
            self.set_infer_input(sim, compile_opt)

            if self.cfg['infer_report_opt']['enabled']:
                t1 = time.perf_counter()

            sim.run()

            if self.cfg['infer_report_opt']['enabled']:
                t = (time.perf_counter() - t1) * 1000
                self.infer_report_dict['actual_fps'] = str(round(1000 / t, 3))

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

    def dump_kmodel_desc(self, file):
        input_shapes = data_shape_list_string(self.inputs)
        output_shapes = data_shape_list_string(self.outputs)
        s = f"{len(self.inputs)} {len(self.outputs)}\n{input_shapes}\n{output_shapes}\n"
        with open(file, "w+") as f:
            f.write(s)

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

    def send_msg(self, sock, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        sock.sendall(msg)

    def recv_msg(self, sock):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        return self.recvall(sock, msglen)

    def recvall(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def run_evb(self, target, kmodel, compile_opt, infer_dir):
        ip = test_utils.nuc_ip()
        port = test_utils.nuc_port()
        test_executable = test_utils.test_executable(target)

        # connect server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, int(port)))

        # send target
        dummy = self.recv_msg(client_socket)
        target_dict = {}
        target_dict['target'] = target
        self.send_msg(client_socket, json.dumps(target_dict).encode())

        # send header
        dummy = self.recv_msg(client_socket)
        header_dict = {}
        header_dict['case'] = os.path.basename(self.case_dir)
        header_dict['app'] = 1
        header_dict['kmodel'] = 1
        header_dict['inputs'] = len(self.inputs)
        header_dict['description'] = 1 if self.dynamic else 0
        header_dict['outputs'] = len(self.outputs)
        header_dict['cfg_cmds'] = self.config_cmds()
        self.send_msg(client_socket, json.dumps(header_dict).encode())

        # send app
        dummy = self.recv_msg(client_socket)
        file_dict = {}
        file_dict['file_name'] = os.path.basename(test_executable)
        file_dict['file_size'] = os.path.getsize(test_executable)
        self.send_msg(client_socket, json.dumps(file_dict).encode())
        dummy = self.recv_msg(client_socket)
        with open(test_executable, 'rb') as f:
            client_socket.sendall(f.read())

        # send kmodel
        dummy = self.recv_msg(client_socket)
        file_dict['file_name'] = self.cfg['kmodel_name']
        file_dict['file_size'] = len(kmodel)
        self.send_msg(client_socket, json.dumps(file_dict).encode())
        dummy = self.recv_msg(client_socket)
        client_socket.sendall(kmodel)

        # send inputs
        for idx, value in enumerate(self.inputs):
            dummy = self.recv_msg(client_socket)
            data = self.transform_input(
                value['data'], compile_opt['input_type'], "infer")[0]
            file_dict['file_name'] = f'input_{idx}.bin'
            file_dict['file_size'] = data.size * data.itemsize
            self.send_msg(client_socket, json.dumps(file_dict).encode())
            dummy = self.recv_msg(client_socket)
            client_socket.sendall(data.tobytes())

        # send kmodel.desc
        if self.dynamic:
            dummy = self.recv_msg(client_socket)
            desc_file = os.path.join(infer_dir, self.cfg['desc_name'])
            file_dict['file_name'] = os.path.basename(desc_file)
            file_dict['file_size'] = os.path.getsize(desc_file)
            self.send_msg(client_socket, json.dumps(file_dict).encode())
            dummy = self.recv_msg(client_socket)
            with open(desc_file, 'rb') as f:
                client_socket.sendall(f.read())

        # get infer result
        outputs = []
        header_dict = {}
        ret = self.recv_msg(client_socket)
        header_dict = json.loads(ret.decode())
        ret = header_dict['msg']
        if header_dict['type'].find('finish') != -1:
            if self.cfg['infer_report_opt']['enabled']:
                self.stat_target(infer_dir, ret)

            self.send_msg(client_socket, f"pls send outputs".encode())

            # recv outputs
            for i in range(len(self.outputs)):
                header = self.recv_msg(client_socket)
                file_dict = json.loads(header.decode())
                file_size = file_dict['file_size']
                self.send_msg(client_socket, f"pls send file".encode())

                buffer = bytearray(file_size)
                buffer = self.recvall(client_socket, file_size)

                output = np.frombuffer(buffer, dtype=self.outputs[i]['dtype'])
                outputs.append(output)

                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(infer_dir, f'nncase_result_{i}.bin'), output)
                    dump_txt_file(os.path.join(infer_dir, f'nncase_result_{i}.txt'), output)

            client_socket.close()
        else:
            client_socket.close()

            if self.cfg['infer_report_opt']['enabled']:
                self.infer_report_dict['result'] = 'Fail'
                self.infer_report_dict['remark'] = escape(ret)
                prefix, suffix = os.path.splitext(self.infer_report_file)
                json_file = f'{prefix}_{os.path.basename(self.case_dir)}{suffix}'
                dump_dict_to_json(self.infer_report_dict, json_file)

            raise Exception(ret)

        return outputs
