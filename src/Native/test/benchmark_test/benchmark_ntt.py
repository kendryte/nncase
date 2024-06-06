import argparse
import os
from pathlib import Path
import subprocess
import socket
import struct
import json
import time
from html import escape


def kpu_targets():
    return os.getenv('KPU_TARGETS', "").split(',')


def nuc_ip():
    return os.getenv('NUC_PROXY_IP')


def nuc_port():
    return os.getenv('NUC_PROXY_PORT')


def report_file(default: str):
    return os.getenv('BENCHMARK_NTT_REPORT_FILE', default)


def generate_markdown(benchmark_list: list, md_file: str):
    # generate dict after sorting
    benchmark_list = sorted(benchmark_list, key=lambda d: (d['kind'], d['op']))
    dict = {}
    for e in benchmark_list:
        kind = e['kind']
        if kind not in dict:
            dict[kind] = []
        dict[kind].append(e)

    # generate html table
    md = '<table>\n'

    # table head
    md += '\t<tr>\n'
    for key in benchmark_list[0]:
        md += f'\t\t<th>{key}</th>\n'
    md += '\t</tr>\n'

    # table row
    for value in dict.values():
        length = len(value)
        for i in range(length):
            md += '\t<tr>\n'
            if i == 0:
                for k, v in value[i].items():
                    if k == 'kind':
                        md += f'\t\t<td rowspan=\'{length}\'>{v}</td>\n'
                    else:
                        md += f'\t\t<td>{v}</td>\n'
            else:
                for k, v in value[i].items():
                    if k != 'kind':
                        md += f'\t\t<td>{v}</td>\n'
            md += '\t</tr>\n'

    md += '</table>\n'

    with open(md_file, 'w') as f:
        f.write(md)


class BenchmarkNTT():
    def __init__(self, arch: str, target: str, bin_path: str):
        self.arch = arch
        self.target = target
        self.bin_path = bin_path
        self.bin_prefix = 'benchmark_ntt_'
        self.bin_list = self.traverse_dir()
        self.benchmark_list = []

    def traverse_dir(self):
        file_list = []
        for bin in Path(os.path.dirname(self.bin_path)).glob(f'{self.bin_prefix}*'):
            file_list.append(bin)
        return file_list

    def parse_result(self, result: str):
        lines = result.split('\n')
        for line in lines:
            items = line.split(' ')
            dict = {}
            dict['kind'], dict['op'] = items[0].split(self.bin_prefix)[1].split('_', 1)
            dict[f'{self.arch}_roofline'] = self.roofline_dict[dict['kind']][dict['op']]
            dict[f'{self.arch}_actual'] = items[-2]
            self.benchmark_list.append(dict)

    def run():
        pass


class BenchmarkNTT_x86_64(BenchmarkNTT):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTT.__init__(self, 'x86_64', target, bin_path)
        self.roofline_dict = {'binary': {'add': 'N/A',
                                         'sub': 'N/A',
                                         'mul': 'N/A',
                                         'div': 'N/A',
                                         'max': 'N/A',
                                         'min': 'N/A',
                                         'floor_mod': 'N/A',
                                         'mod': 'N/A',
                                         'pow': 'N/A',
                                         },
                              'unary': {'abs': 'N/A',
                                        'acos': 'N/A',
                                        'acosh': 'N/A',
                                        'asin': 'N/A',
                                        'asinh': 'N/A',
                                        'ceil': 'N/A',
                                        'cos': 'N/A',
                                        'cosh': 'N/A',
                                        'exp': 'N/A',
                                        'floor': 'N/A',
                                        'log': 'N/A',
                                        'neg': 'N/A',
                                        'round': 'N/A',
                                        'rsqrt': 'N/A',
                                        'sign': 'N/A',
                                        'sin': 'N/A',
                                        'sinh': 'N/A',
                                        'sqrt': 'N/A',
                                        'square': 'N/A',
                                        'swish': 'N/A',
                                        'tanh': 'N/A',
                                        },
                              }

    def run(self):
        for bin in self.bin_list:
            cmd_status, cmd_result = subprocess.getstatusoutput(f'{bin}')
            assert(cmd_status == 0)
            self.parse_result(cmd_result)


class BenchmarkNTT_riscv64(BenchmarkNTT):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTT.__init__(self, 'riscv64', target, bin_path)
        self.roofline_dict = {'binary': {'add': '10.5',
                                         'sub': '10.5',
                                         'mul': '10.5',
                                         'div': '40.3',
                                         'max': '10.5',
                                         'min': '10.5',
                                         'floor_mod': '43',
                                         'mod': '54',
                                         'pow': 'N/A'
                                         },
                              'unary': {'abs': '8',
                                        'acos': 'N/A',
                                        'acosh': 'N/A',
                                        'asin': 'N/A',
                                        'asinh': 'N/A',
                                        'ceil': '18',
                                        'cos': 'N/A',
                                        'cosh': 'N/A',
                                        'exp': 'N/A',
                                        'floor': '18',
                                        'log': 'N/A',
                                        'neg': '8',
                                        'round': '11',
                                        'rsqrt': '8',
                                        'sign': '18',
                                        'sin': 'N/A',
                                        'sinh': 'N/A',
                                        'sqrt': '40',
                                        'square': '8',
                                        'swish': 'N/A',
                                        'tanh': 'N/A',
                                        },
                              }

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

    def run_evb(self, bin):
        # connect server
        ip = nuc_ip()
        port = nuc_port()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, int(port)))

        # send target
        dummy = self.recv_msg(client_socket)
        target_dict = {}
        target_dict['target'] = self.target
        self.send_msg(client_socket, json.dumps(target_dict).encode())

        # send header
        dummy = self.recv_msg(client_socket)
        header_dict = {}
        header_dict['case'] = os.path.basename(bin)
        header_dict['app'] = 1
        header_dict['kmodel'] = 0
        header_dict['inputs'] = 0
        header_dict['description'] = 0
        header_dict['outputs'] = 0
        header_dict['cfg_cmds'] = []
        self.send_msg(client_socket, json.dumps(header_dict).encode())

        # send app
        dummy = self.recv_msg(client_socket)
        file_dict = {}
        file_dict['file_name'] = os.path.basename(bin)
        file_dict['file_size'] = os.path.getsize(bin)
        self.send_msg(client_socket, json.dumps(file_dict).encode())
        dummy = self.recv_msg(client_socket)
        with open(bin, 'rb') as f:
            client_socket.sendall(f.read())

        # get result
        header_dict = {}
        ret = self.recv_msg(client_socket)
        header_dict = json.loads(ret.decode())
        msg = header_dict['msg']
        if header_dict['type'].find('finish') != -1:
            self.send_msg(client_socket, f"recved msg".encode())
            client_socket.close()
            return 0, msg[0]
        else:
            client_socket.close()
            raise Exception(msg)
            return -1, msg

    def run(self):
        if self.target not in kpu_targets():
            return

        for bin in self.bin_list:
            cmd_status, cmd_result = self.run_evb(bin)
            assert(cmd_status == 0)
            lines = cmd_result.split('\r\n')
            new_lines = lines[1:-1]
            new_cmd_result = '\n'.join(new_lines)
            self.parse_result(new_cmd_result)


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser(prog="benchmark_ntt")
    parser.add_argument("--x86_64_target", help='x86_64 target to run on',
                        type=str, default='local')
    parser.add_argument("--x86_64_path", help='bin path for x86_64',
                        type=str, default='x86_64_build/bin')
    parser.add_argument("--riscv64_target", help='riscv64 target to run on',
                        type=str, default='k230')
    parser.add_argument("--riscv64_path", help='bin path for riscv64',
                        type=str, default='riscv64_build/bin')
    args = parser.parse_args()

    # x86_64
    ntt_x86_64 = BenchmarkNTT_x86_64(args.x86_64_target, args.x86_64_path)
    ntt_x86_64.run()

    # riscv64
    ntt_riscv64 = BenchmarkNTT_riscv64(args.riscv64_target, args.riscv64_path)
    ntt_riscv64.run()

    # merge benchmark list
    benchmark_list = []
    for i in range(len(ntt_x86_64.benchmark_list)):
        item = {**ntt_x86_64.benchmark_list[i], **ntt_riscv64.benchmark_list[i]}
        benchmark_list.append(item)

    # generate md
    md_file = report_file('benchmark_ntt.md')
    generate_markdown(benchmark_list, md_file)
