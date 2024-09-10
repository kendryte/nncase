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
            if self.bin_prefix.lower() in line.lower():
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
        self.roofline_dict = {'binary': {'add': '2.4',
                                         'sub': '2.4',
                                         'mul': '2.4',
                                         'div': '3.0',
                                         'max': '2.4',
                                         'min': '2.4',
                                         'floor_mod': '3.23',
                                         'mod': '3.0',
                                         'pow': '42.17',
                                         },
                              'clamp': {'NoPack': '1.5'},
                              'unary': {'abs': '2.0',
                                        'acos': '8.2',
                                        'acosh': '25.34',
                                        'asin': '8.2',
                                        'asinh': '25.34',
                                        'ceil': '1.5',
                                        'cos': '8.7',
                                        'cosh': '12.5',
                                        'exp': '9.27',
                                        'floor': '1.5',
                                        'log': '14.72',
                                        'neg': '1.5',
                                        'round': '1.5',
                                        'rsqrt': '7.5',
                                        'sign': '2.18',
                                        'sin': '8.7',
                                        'sinh': '12.5',
                                        'sqrt': '3.5',
                                        'square': '1.4',
                                        'swish': '12.5',
                                        'tanh': '6.5',
                                        },
                              'matmul': {'no_pack': '49152',
                                         'pack_K': '9728',
                                         'pack_M': '3584',
                                         'pack_N': '2016',
                                         'pack_M_N': '3040',
                                         'pack_M_K': '9728',
                                         'pack_K_N': '2016',
                                         'pack_M_K_N': '2016',
                                         },
                              'reduce': {'Add_reduceM_NoPack': 2047,
                                         'Add_reduceM_PackM': 526,
                                         'Add_reduceMN_NoPack': 2016,
                                         'Add_reduceMN_PackM': 526,
                                         'Add_reduceMN_PackN': 526,
                                         'Add_reduceN_NoPack': 2047,
                                         'Add_reduceN_PackN': 526,
                                         'Max_reduceM_NoPack': 2047,
                                         'Max_reduceM_PackM': 263,
                                         'Max_reduceMN_NoPack': 2016,
                                         'Max_reduceMN_PackM': 263,
                                         'Max_reduceMN_PackN': 263,
                                         'Max_reduceN_NoPack': 2047,
                                         'Max_reduceN_PackN': 263,
                                         'Mean_reduceM_NoPack': 2056,
                                         'Mean_reduceM_PackM': 544,
                                         'Mean_reduceMN_NoPack': 2020.5,
                                         'Mean_reduceMN_PackM': 535,
                                         'Mean_reduceMN_PackN': 535,
                                         'Mean_reduceN_NoPack': 2056,
                                         'Mean_reduceN_PackN': 544,
                                         'Min_reduceM_NoPack': 2047,
                                         'Min_reduceM_PackM': 263,
                                         'Min_reduceMN_NoPack': 2016,
                                         'Min_reduceMN_PackM': 263,
                                         'Min_reduceMN_PackN': 263,
                                         'Min_reduceN_NoPack': 2047,
                                         'Min_reduceN_PackN': 263,
                                         },
                              'softmax': {'no_pack_dim0': 'N/A',
                                          'no_pack_dim1': 'N/A',
                                          'pack0_dim0': 'N/A',
                                          'pack0_dim1': 'N/A',
                                          'pack1_dim0': 'N/A',
                                          'pack1_dim1': 'N/A',
                                          },
                              }

    def run(self):
        for bin in self.bin_list:
            cmd_status, cmd_result = subprocess.getstatusoutput(f'{bin}')
            assert (cmd_status == 0)
            self.parse_result(cmd_result)


class BenchmarkNTT_riscv64(BenchmarkNTT):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTT.__init__(self, 'riscv64', target, bin_path)
        self.roofline_dict = {'binary': {'add': '10.3',
                                         'sub': '10.3',
                                         'mul': '10.3',
                                         'div': '42.3',
                                         'max': '10.3',
                                         'min': '10.3',
                                         'floor_mod': '43',
                                         'mod': '54',
                                         'pow': '139'
                                         },
                              'clamp': {'NoPack': '12.3'},
                              'unary': {'abs': '8',
                                        'acos': '84',
                                        'acosh': '123',
                                        'asin': '83',
                                        'asinh': '127',
                                        'ceil': '18',
                                        'cos': '62',
                                        'cosh': '112',
                                        'exp': '66',
                                        'floor': '18',
                                        'log': '72',
                                        'neg': '8',
                                        'round': '11',
                                        'rsqrt': '58',
                                        'sign': '18',
                                        'sin': '54',
                                        'sinh': '110',
                                        'sqrt': '40',
                                        'square': '8',
                                        'swish': '112',
                                        'tanh': '54',
                                        },
                              'matmul': {'no_pack': '92058',
                                         'pack_K': '155648',
                                         'pack_M': '75674',
                                         'pack_N': '92058',
                                         'pack_M_N': '47322',
                                         'pack_M_K': '95744',
                                         'pack_K_N': '24639',
                                         'pack_M_K_N': '24639',
                                         },
                              'reduce': {'Add_reduceN_NoPack': '5138',
                                         'Add_reduceN_PackN': '5138',
                                         'Add_reduceM_NoPack': '10239',
                                         'Add_reduceM_PackM': '5138',
                                         'Add_reduceMN_NoPack': '4940',
                                         'Add_reduceMN_PackN': '4940',
                                         'Add_reduceMN_PackM': '5084',
                                         'Max_reduceN_NoPack': '6154',
                                         'Max_reduceN_PackN': '6154',
                                         'Max_reduceM_NoPack': '12286',
                                         'Max_reduceM_PackM': '6154',
                                         'Max_reduceMN_NoPack': '5897',
                                         'Max_reduceMN_PackN': '5897',
                                         'Max_reduceMN_PackM': '6089',
                                         'Min_reduceN_NoPack': '6154',
                                         'Min_reduceN_PackN': '6154',
                                         'Min_reduceM_NoPack': '12286',
                                         'Min_reduceM_PackM': '6154',
                                         'Min_reduceMN_NoPack': '5897',
                                         'Min_reduceMN_PackN': '5897',
                                         'Min_reduceMN_PackM': '6089',
                                         'Mean_reduceN_NoPack': '5148',
                                         'Mean_reduceN_PackN': '5142',
                                         'Mean_reduceM_NoPack': '10244',
                                         'Mean_reduceM_PackM': '5148',
                                         'Mean_reduceMN_NoPack': '4945',
                                         'Mean_reduceMN_PackN': '5905',
                                         'Mean_reduceMN_PackM': '5089',
                                         },
                              'softmax': {'no_pack_dim0': 'N/A',
                                          'no_pack_dim1': 'N/A',
                                          'pack0_dim0': 'N/A',
                                          'pack0_dim1': 'N/A',
                                          'pack1_dim0': 'N/A',
                                          'pack1_dim1': 'N/A',
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
            assert (cmd_status == 0)
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
