from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils
import preprocess_utils
import socket
import json
from test_utils import *
import time
import subprocess
from update_trace_info import *
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
        s = f"{len(self.inputs)} {len(self.outputs)}\n{input_shapes}\n{output_shapes}"
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

    def run_evb(self, target, kmodel, compile_opt, infer_dir):
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
        header_dict['description'] = 1 if self.dynamic else 0
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
        file_dict['file_name'] = self.cfg['kmodel_name']
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

        # send kmodel.desc
        if self.dynamic:
            dummy = client_socket.recv(1024)
            desc_file = os.path.join(infer_dir, self.cfg['desc_name'])
            file_dict['file_name'] = os.path.basename(desc_file)
            file_dict['file_size'] = os.path.getsize(desc_file)
            client_socket.sendall(json.dumps(file_dict).encode())
            dummy = client_socket.recv(1024)
            with open(desc_file, 'rb') as f:
                client_socket.sendall(f.read())

        # get infer result
        outputs = []
        header_dict = {}
        ret = client_socket.recv(1024)
        header_dict = json.loads(ret.decode())
        length = header_dict['len']

        # recv result
        count = length // 1024
        left = length % 1024

        client_socket.sendall(f"pls send detail".encode())
        recv_data = b''
        for i in range(count):
            data = client_socket.recv(1024, socket.MSG_WAITALL)
            recv_data += data

        if left:
            recv_data += client_socket.recv(left, socket.MSG_WAITALL)

        detail = recv_data.decode()

        if header_dict['type'].find('finish') != -1:
            if self.cfg['infer_report_opt']['enabled']:
                if not self.dynamic:
                    # update trace info
                    model_name = self.cfg['infer_report_opt']['model_name']
                    infer_result = f'0:{model_name} :\n' + detail
                    trace_file = search_file(infer_dir, 'trace_info.py')
                    assert(trace_file != '')
                    update_trace_info(infer_result, trace_file)

                    # roofline fps/mac usage
                    estimate_file = search_file(infer_dir, 'estimate_fps.py')
                    assert(estimate_file != '')

                    mac_file = search_file(infer_dir, 'mac.csv')
                    assert(mac_file != '')

                    cmd_status, cmd_result = subprocess.getstatusoutput(
                        f'python3 {estimate_file} {mac_file}')
                    assert(cmd_status == 0)
                    data = cmd_result.split(',')
                    assert(len(data) >= 3)
                    self.infer_report_dict['roofline_fps'] = data[1].split(':')[-1].strip()
                    self.infer_report_dict['roofline_mac_usage'] = data[2].split(':')[-1].strip()

                # actual fps
                fps_pattern = re.compile(
                    r"^\|total\s+\|(\d+|\d+.\d+)\s+\|(\d+|\d+.\d+)\s+\|(\d+|\d+.\d+)\s+\|")
                buf = io.StringIO(detail)
                while True:
                    line = buf.readline()
                    if not line:
                        break
                    match = fps_pattern.match(line)
                    if match is not None:
                        self.infer_report_dict['actual_fps'] = str(
                            round(1000 / float(match.group(2)), 3))
                        break

                if not self.dynamic:
                    # actual mac usage
                    draw_trace_file = search_file(infer_dir, 'draw_trace.py')
                    assert(draw_trace_file != '')
                    cmd_status, cmd_result = subprocess.getstatusoutput(
                        f'python3 {draw_trace_file} {mac_file}')
                    assert(cmd_status == 0)
                    data = cmd_result.split(',')
                    assert(len(data) >= 1)
                    self.infer_report_dict['actual_mac_usage'] = data[0].split(':')[-1].strip()

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
                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(infer_dir, f'nncase_result_{i}.bin'), output)
                    dump_txt_file(os.path.join(infer_dir, f'nncase_result_{i}.txt'), output)
                client_socket.sendall(f"recv nncase_result_{i}.bin succeed".encode())

            client_socket.close()
        else:
            client_socket.close()

            if self.cfg['infer_report_opt']['enabled']:
                self.infer_report_dict['result'] = 'Fail'
                self.infer_report_dict['remark'] = escape(detail)
                prefix, suffix = os.path.splitext(self.infer_report_file)
                json_file = f'{prefix}_{os.path.basename(self.case_dir)}{suffix}'
                dump_dict_to_json(self.infer_report_dict, json_file)

            raise Exception(detail)

        return outputs
