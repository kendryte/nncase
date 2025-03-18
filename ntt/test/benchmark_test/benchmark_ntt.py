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


def ntt_report_file(default: str):
    return os.getenv('BENCHMARK_NTT_REPORT_FILE', default)


def ntt_matmul_x86_64_report_file(default: str):
    return os.getenv('BENCHMARK_NTT_MATMUL_X86_64_REPORT_FILE', default)


def ntt_matmul_riscv64_report_file(default: str):
    return os.getenv('BENCHMARK_NTT_MATMUL_RISCV64_REPORT_FILE', default)


def generate_benchmark_ntt_md(benchmark_list: list, key: str, md_file: str):
    # generate dict after sorting
    dict = {}
    for e in benchmark_list:
        k = e[key]
        if k not in dict:
            dict[k] = []
        dict[k].append(e)

    # generate html table
    md = '<table>\n'

    # table head
    md += '\t<tr>\n'
    for k in benchmark_list[0]:
        md += f'\t\t<th>{k}</th>\n'
    md += '\t</tr>\n'

    # table row
    for value in dict.values():
        length = len(value)
        for i in range(length):
            md += '\t<tr>\n'
            if i == 0:
                for k, v in value[i].items():
                    if k == key:
                        md += f'\t\t<td rowspan=\'{length}\'>{v}</td>\n'
                    else:
                        md += f'\t\t<td>{v}</td>\n'
            else:
                for k, v in value[i].items():
                    if k != key:
                        md += f'\t\t<td>{v}</td>\n'
            md += '\t</tr>\n'

    md += '</table>\n'

    with open(md_file, 'w') as f:
        f.write(md)


class Benchmark():
    def __init__(self, arch: str, target: str, bin_path: str, bin_prefix: str):
        self.arch = arch
        self.target = target
        self.bin_path = bin_path
        self.bin_prefix = bin_prefix
        self.bin_list = self.traverse_dir()
        self.benchmark_list = []

    def traverse_dir(self):
        file_list = []
        for bin in Path(os.path.dirname(self.bin_path)).glob(f'{self.bin_prefix}*'):
            file_list.append(bin)
        return file_list


class Benchmark_riscv64():
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


class BenchmarkNTT(Benchmark):
    def __init__(self, arch: str, target: str, bin_path: str):
        Benchmark.__init__(self, arch, target, bin_path, 'benchmark_ntt_')

    def parse_result(self, result: str):
        lines = result.split('\n')
        for line in lines:
            if self.bin_prefix.lower() in line.lower():
                items = line.split(' ')
                dict = {}
                dict['kind'], dict['op'] = items[0].split(self.bin_prefix)[1].split('_', 1)
                dict[f'{self.arch}_roofline'] = self.roofline_dict[dict['kind']][dict['op']]
                dict[f'{self.arch}_actual'] = items[-2]
                dict[f'{self.arch}_ratio'] = round(
                    float(dict[f'{self.arch}_roofline']) / (float(dict[f'{self.arch}_actual']) + 0.01), 3)
                self.benchmark_list.append(dict)

    def run():
        pass


class BenchmarkNTT_x86_64(BenchmarkNTT):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTT.__init__(self, 'x86_64', target, bin_path)
        self.roofline_dict = {'binary': {'add': '2.0',
                                         'sub': '2.0',
                                         'mul': '2.0',
                                         'div': '2.0',
                                         'max': '2.0',
                                         'min': '2.0',
                                         'floor_mod': '5.5',
                                         'mod': '2.0',
                                         'pow': '42.17',
                                         },
                              'cast': {'float-int32': '2.0',
                                       'int32-float': '1.5',
                                       'float-uint32': '1.5',
                                       'uint32-float': '1.5',
                                       'float-bool': '6',
                                       'bool-float': '6.0',
                                       'float-f8e4m3': '470',
                                       },
                              'clamp': {'Pack': '2.5'},
                              'unary': {'abs': '1.5',
                                        'acos': '8.2',
                                        'acosh': '25.34',
                                        'asin': '8.2',
                                        'asinh': '25.34',
                                        'ceil': '1.5',
                                        'cos': '8.7',
                                        'cosh': '12.5',
                                        'erf': '26.5',
                                        'exp': '9.27',
                                        'floor': '1.5',
                                        'log': '14.72',
                                        'neg': '1.5',
                                        'round': '1.5',
                                        'rsqrt': '7.5',
                                        'sign': '1.5',
                                        'sin': '8.7',
                                        'sinh': '12.5',
                                        'sqrt': '4.5',
                                        'square': '1.5',
                                        'swish': '12.5',
                                        'tanh': '27.5',
                                        },
                              'matmul': {'no_pack': '49152',
                                         'pack_K': '9728',
                                         'pack_M': '2016',
                                         'pack_N': '2016',
                                         'pack_M_N': '2016',
                                         'pack_M_K': '9728',
                                         'pack_K_N': '2016',
                                         'pack_M_K_N': '2016',
                                         },
                              'reduce': {'Add_reduceM_PackM': 256,
                                         'Add_reduceMN_PackM': 256,
                                         'Add_reduceMN_PackN': 256,
                                         'Add_reduceN_PackN': 256,
                                         'Max_reduceM_PackM': 256,
                                         'Max_reduceMN_PackM': 256,
                                         'Max_reduceMN_PackN': 256,
                                         'Max_reduceN_PackN': 256,
                                         'Mean_reduceM_PackM': 263,
                                         'Mean_reduceMN_PackM': 263,
                                         'Mean_reduceMN_PackN': 263,
                                         'Mean_reduceN_PackN': 263,
                                         'Min_reduceM_PackM': 256,
                                         'Min_reduceMN_PackM': 256,
                                         'Min_reduceMN_PackN': 256,
                                         'Min_reduceN_PackN': 256,
                                         },
                              'slice': {'contiguous_step_1': '1.5',
                                        'contiguous_step_2': '1.5',
                                        'contiguous_step_4': '1.5',
                                        'no_contiguous_step_1': '1.5',
                                        'no_contiguous_step_2': '1.5',
                                        'no_contiguous_step_4': '1.5',
                                        },
                              'gather': {'pack1d_dim0_contiguous': '1.5',
                                         'pack1d_dim0_no_contiguous': '1.5',
                                         'pack1d_dim1_contiguous': '1.5',
                                         'pack1d_dim1_no_contiguous': '1.5',
                                         'pack2d_dim0_contiguous': '1.5',
                                         'pack2d_dim1_contiguous': '1.5',
                                         },
                              'pack': {'N': '2.0',
                                       'C': '2.0',
                                       'H': '2.0',
                                       'W': '1.5',
                                       'NC': '2.5',
                                       'CH': '2.5',
                                       'HW': '2.0',
                                       },
                              'unpack': {'N': '2.0',
                                         'C': '2.0',
                                         'H': '2.0',
                                         'W': '1.5',
                                         'NC': '2.5',
                                         'CH': '2.5',
                                         'HW': '2.0',
                                         },
                              'expand': {'N': '1.5',
                                         'C': '1.5',
                                         'H': '1.5',
                                         'W': '1.5',
                                         'NC': '1.5',
                                         'CH': '1.5',
                                         'HW': '1.5',
                                         'NH': '1.5',
                                         'CW': '1.5',
                                         'NW': '1.5',
                                         },
                              'transpose': {"NCHW": '1.5',
                                            "NCWH": '1.5',
                                            "NHCW": '1.5',
                                            "NHWC": '1.5',
                                            "NWCH": '1.5',
                                            "NWHC": '1.5',
                                            "CNHW": '1.5',
                                            "CNWH": '1.5',
                                            "CHNW": '1.5',
                                            "CHWN": '1.5',
                                            "CWNH": '1.5',
                                            "CWHN": '1.5',
                                            "HNCW": '1.5',
                                            "HNWC": '1.5',
                                            "HCNW": '1.5',
                                            "HCWN": '1.5',
                                            "HWNC": '1.5',
                                            "HWCN": '1.5',
                                            "WNCH": '1.5',
                                            "WNHC": '1.5',
                                            "WCNH": '1.5',
                                            "WCHN": '1.5',
                                            "WHNC": '1.5',
                                            "WHCN": '1.5',
                                            },
                               'compare': {'equal': '2.0',
                                           'not_equal': '2.0',
                                           'greater': '2.0',
                                           'greater_or_equal': '2.0',
                                           'less': '2.0',
                                           'less_or_equal': '2.0',
                                         },
                                'where': {'pack': '2.5',},
                                'scatterND': {'pack': '1.5',},
                                }
                               
                              

    def run(self):
        for bin in self.bin_list:
            cmd_status, cmd_result = subprocess.getstatusoutput(f'{bin}')
            assert (cmd_status == 0)
            self.parse_result(cmd_result)


class BenchmarkNTT_riscv64(BenchmarkNTT, Benchmark_riscv64):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTT.__init__(self, 'riscv64', target, bin_path)
        self.roofline_dict = {'binary': {'add': '7.3',
                                         'sub': '7.3',
                                         'mul': '7.3',
                                         'div': '30.3',
                                         'max': '7.3',
                                         'min': '7.3',
                                         'floor_mod': '40.3',
                                         'mod': '35.3',
                                         'pow': '139'
                                         },
                              'cast': {'float-int32': '5',
                                       'int32-float': '5',
                                       'float-uint32': '5',
                                       'uint32-float': '5',
                                       'float-bool': '20.3',
                                       'bool-float': '19.2',
                                       'float-f8e4m3': '410',
                                       },
                              'clamp': {'Pack': '6.3'},
                              'unary': {'abs': '6.3',
                                        'acos': '84',
                                        'acosh': '123',
                                        'asin': '83',
                                        'asinh': '127',
                                        'ceil': '12.3',
                                        'cos': '62',
                                        'cosh': '112',
                                        'erf': '69',
                                        'exp': '66',
                                        'floor': '12.3',
                                        'log': '72',
                                        'neg': '6.3',
                                        'round': '8.3',
                                        'rsqrt': '58',
                                        'sign': '12.3',
                                        'sin': '54',
                                        'sinh': '110',
                                        'sqrt': '40',
                                        'square': '5.3',
                                        'swish': '112',
                                        'tanh': '197',
                                        },
                              'matmul': {'no_pack': '92058',
                                         'pack_K': '57344',
                                         'pack_M': '8192',
                                         'pack_N': '8192',
                                         'pack_M_N': '8192',
                                         'pack_M_K': '57344',
                                         'pack_K_N': '8192',
                                         'pack_M_K_N': '8192',
                                         },
                              'reduce': {'Add_reduceN_PackN': '3132',
                                         'Add_reduceM_PackM': '3132',
                                         'Add_reduceMN_PackN': '3102',
                                         'Add_reduceMN_PackM': '3102',
                                         'Max_reduceN_PackN': '3126',
                                         'Max_reduceM_PackM': '3126',
                                         'Max_reduceMN_PackN': '3099',
                                         'Max_reduceMN_PackM': '3099',
                                         'Min_reduceN_PackN': '3126',
                                         'Min_reduceM_PackM': '3126',
                                         'Min_reduceMN_PackN': '3099',
                                         'Min_reduceMN_PackM': '3099',
                                         'Mean_reduceN_PackN': '3140',
                                         'Mean_reduceM_PackM': '3140',
                                         'Mean_reduceMN_PackN': '3106',
                                         'Mean_reduceMN_PackM': '3106',
                                         },
                              'slice': {'contiguous_step_1': '4.3',
                                        'contiguous_step_2': '4.3',
                                        'contiguous_step_4': '4.3',
                                        'no_contiguous_step_1': '4.3',
                                        'no_contiguous_step_2': '4.3',
                                        'no_contiguous_step_4': '4.3',
                                        },
                              'gather': {'pack1d_dim0_contiguous': '4.3',
                                         'pack1d_dim0_no_contiguous': '4.3',
                                         'pack1d_dim1_contiguous': '4.3',
                                         'pack1d_dim1_no_contiguous': '4.3',
                                         'pack2d_dim0_contiguous': '4.3',
                                         'pack2d_dim1_contiguous': '4.3',
                                         },
                              'pack': {'N': '6',
                                       'C': '6',
                                       'H': '6',
                                       'W': '4.3',
                                       'NC': '6',
                                       'CH': '6',
                                       'HW': '4.3',
                                       },
                              'unpack': {'N': '6',
                                         'C': '6',
                                         'H': '6',
                                         'W': '4.3',
                                         'NC': '6',
                                         'CH': '6',
                                         'HW': '4.3',
                                         },
                              'expand': {'N': '4.3',
                                         'C': '4.3',
                                         'H': '4.3',
                                         'W': '4.3',
                                         'NC': '4.3',
                                         'CH': '4.3',
                                         'HW': '4.3',
                                         'NH': '4.3',
                                         'CW': '4.3',
                                         'NW': '4.3',
                                         },
                              'transpose': {"NCHW": '4.3',
                                            "NCWH": '4.3',
                                            "NHCW": '4.3',
                                            "NHWC": '4.3',
                                            "NWCH": '4.3',
                                            "NWHC": '4.3',
                                            "CNHW": '4.3',
                                            "CNWH": '4.3',
                                            "CHNW": '4.3',
                                            "CHWN": '4.3',
                                            "CWNH": '4.3',
                                            "CWHN": '4.3',
                                            "HNCW": '4.3',
                                            "HNWC": '4.3',
                                            "HCNW": '4.3',
                                            "HCWN": '4.3',
                                            "HWNC": '4.3',
                                            "HWCN": '4.3',
                                            "WNCH": '4.3',
                                            "WNHC": '4.3',
                                            "WCNH": '4.3',
                                            "WCHN": '4.3',
                                            "WHNC": '4.3',
                                            "WHCN": '4.3',
                                            },
                               'compare': {'equal': '7.3',
                                           'not_equal': '7.3',
                                           'greater': '7.3',
                                           'greater_or_equal': '7.3',
                                           'less': '7.3',
                                           'less_or_equal': '7.3',
                                          },
                               'where': {'pack': '4.3',},
                               'scatterND': {'pack': '4.3'},
                              }

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


class BenchmarkNTTMatmul(Benchmark):
    def __init__(self, arch: str, target: str, bin_path: str):
        Benchmark.__init__(self, arch, target, bin_path, 'benchmark_ntt_matmul_primitive_size')

    def parse_result(self, result: str):
        lines = result.strip().split('\n')
        for line in lines:
            dict = {}
            items = line.split(',')
            dict['pack_mode'] = items[0].strip()
            dict['M'] = items[1].split(':')[1].strip()
            dict['K'] = items[2].split(':')[1].strip()
            dict['N'] = items[3].split(':')[1].strip()
            dict[f'{self.arch}_gflops'] = items[4].split(':')[1].strip()
            self.benchmark_list.append(dict)

    def run():
        pass


class BenchmarkNTTMatmul_x86_64(BenchmarkNTTMatmul):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTTMatmul.__init__(self, 'x86_64', target, bin_path)

    def run(self):
        for bin in self.bin_list:
            cmd_status, cmd_result = subprocess.getstatusoutput('taskset -c 0 ' + f'{bin}')
            assert (cmd_status == 0)
            self.parse_result(cmd_result)


class BenchmarkNTTMatmul_riscv64(BenchmarkNTTMatmul, Benchmark_riscv64):
    def __init__(self, target: str, bin_path: str):
        BenchmarkNTTMatmul.__init__(self, 'riscv64', target, bin_path)

    def run(self):
        if self.target not in kpu_targets():
            return

        for bin in self.bin_list:
            cmd_status, cmd_result = self.run_evb(bin)
            assert (cmd_status == 0)
            lines = cmd_result.split('\r\n')
            new_lines = lines[2:-1]
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

    # 1. benchmark ntt
    # 1.1 x86_64
    ntt_x86_64 = BenchmarkNTT_x86_64(args.x86_64_target, args.x86_64_path)
    ntt_x86_64.run()

    # 1.2 riscv64
    ntt_riscv64 = BenchmarkNTT_riscv64(args.riscv64_target, args.riscv64_path)
    ntt_riscv64.run()

    # 1.3 merge benchmark list
    benchmark_list = []
    for i in range(len(ntt_x86_64.benchmark_list)):
        item = {**ntt_x86_64.benchmark_list[i], **ntt_riscv64.benchmark_list[i]}
        benchmark_list.append(item)

    # 1.4 generate md
    md_file = ntt_report_file('benchmark_ntt.md')
    benchmark_list = sorted(benchmark_list, key=lambda d: (d['kind'], d['op']))
    generate_benchmark_ntt_md(benchmark_list, 'kind', md_file)

    # 2. benchmark ntt matmul
    # 2.1 x86_64
    benchmark_list = []
    ntt_matmul_x86_64 = BenchmarkNTTMatmul_x86_64(args.x86_64_target, args.x86_64_path)
    ntt_matmul_x86_64.run()
    benchmark_list = sorted(ntt_matmul_x86_64.benchmark_list, key=lambda d: (d['pack_mode']))
    md_file = ntt_matmul_x86_64_report_file('benchmark_ntt_matmul_x86_64.md')
    generate_benchmark_ntt_md(benchmark_list, 'pack_mode', md_file)

    # 2.2 riscv64
    benchmark_list = []
    ntt_matmul_riscv64 = BenchmarkNTTMatmul_riscv64(args.riscv64_target, args.riscv64_path)
    ntt_matmul_riscv64.run()
    benchmark_list = sorted(ntt_matmul_riscv64.benchmark_list, key=lambda d: (d['pack_mode']))
    md_file = ntt_matmul_riscv64_report_file('benchmark_ntt_matmul_riscv64.md')
    generate_benchmark_ntt_md(benchmark_list, 'pack_mode', md_file)

