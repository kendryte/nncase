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

import os
import argparse
import stat
import socket
import json
import threading
import queue
import logging
import logging.handlers
import serial
import shutil
import time
import toml


class MySerial:
    def __init__(self, port, baudrate, logger):
        self.s = None
        self.port = port
        self.baudrate = baudrate
        self.logger = logger
        self.timeout = 60

    def open(self):
        self.logger.debug(f'open {self.port} begin')
        self.s = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
        if (self.s.isOpen()):
            self.logger.debug(f'open {self.port} succeed end')
        else:
            self.logger.debug(f'open {self.port} failed end')

    def close(self):
        self.logger.debug(f'close {self.port} begin')
        self.s.close()
        self.logger.debug(f'close {self.port} end')

    def write(self, cmd):
        self.logger.debug(f'write {cmd} begin')
        cmd = cmd + '\r'
        self.s.write(cmd.encode())
        self.s.flush()
        self.logger.debug('write end')

    def read_until(self, expected):
        self.logger.debug('read begin')
        data = self.s.read_until(expected.encode()).decode()
        self.logger.debug('read end: data = {0}'.format(data))
        return data

    def run_cmd(self, cmd, expected=''):
        data = ''
        self.open()
        self.write(cmd)
        if expected != '':
            data = self.read_until(expected)

        self.close()
        return data


class Target:
    def __init__(self, name, cfg, nfs, clear_queue):
        self.name = name
        self.infer_queue = queue.Queue(maxsize=clear_queue.maxsize)
        self.clear_queue = clear_queue
        self.working_dir = cfg['working_dir']
        self.separator = cfg['separator']

        # nfs_dir
        self.nfs_dir = os.path.join(nfs, name)
        if not os.path.exists(self.nfs_dir):
            os.makedirs(self.nfs_dir)

        # logging
        mylogger = logging.getLogger()
        mylogger.setLevel(logging.INFO)
        rf_handler = logging.handlers.RotatingFileHandler(
            f'nuc_proxy_{name}.log', mode='a', maxBytes=32 * 1024 * 1024, backupCount=10)
        rf_handler.setLevel(logging.INFO)
        rf_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        mylogger.addHandler(rf_handler)
        self.logger = mylogger

        # serial
        self.s0 = MySerial(cfg['uart0'], cfg['baudrate0'], self.logger)
        self.s1 = MySerial(cfg['uart1'], cfg['baudrate1'], self.logger)


def recv_file(conn, case_dir, logger):
    conn.sendall(f"pls send file info".encode())
    header = conn.recv(1024)
    file_dict = json.loads(header.decode())
    file_name = file_dict['file_name']
    file_size = file_dict['file_size']
    logger.debug('recv begin: file = {0}, size = {1}'.format(file_name, file_size))
    conn.sendall(f"pls send {file_name}".encode())

    full_file = os.path.join(case_dir, file_name)
    with open(full_file, 'wb') as f:
        recv_size = 0
        while recv_size < file_size:
            slice = conn.recv(4096)
            f.write(slice)
            recv_size += len(slice)

    os.chmod(full_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    logger.debug('recv end')
    return file_name


def recv_worker(conn, target):
    # recv header
    conn.sendall(f"pls send header".encode())
    header = conn.recv(1024)
    header_dict = json.loads(header.decode())
    new_case = header_dict['case'] + str(int(time.time()))
    target.logger.info("test case = {0}".format(new_case))
    case_dir = os.path.join(target.nfs_dir, new_case)
    os.makedirs(case_dir)
    file_num = header_dict['app'] + header_dict['kmodel'] + \
        header_dict['inputs'] + header_dict['description']

    # recv all kinds of files(app + kmodel + inputs)
    cmds = f'cd {target.working_dir}/{target.name}/{new_case};./'
    for i in range(file_num):
        file = recv_file(conn, case_dir, target.logger)
        if i == 0:
            cmds = cmds + file
        else:
            cmds = cmds + ' ' + file

    target.logger.debug("cmds = {0}".format(cmds))
    target.infer_queue.put((cmds, conn, case_dir, header_dict['outputs']))


def infer_worker(target):
    while True:
        cmds, conn, case_dir, output_num = target.infer_queue.get()
        separator = os.path.basename(case_dir) + target.separator
        ret = ''

        # exit from face_detect after rebooting
        # target.s1.run_cmd('q')
        target.s1.run_cmd('')

        for cmd in cmds.split(';'):
            ret = target.s1.run_cmd(cmd, separator)

        # infer result
        dict = {'type': 'finish', 'time': 0.0, 'error': ''}
        if ret.find('terminate') != -1 or ret.find('Exception') != -1:
            target.logger.error('infer exception')
            err = f'infer exception: {ret}'
            dict['type'] = 'exception'
            dict['error'] = err[0:1024]
            conn.sendall(json.dumps(dict).encode())
        elif ret.find(separator) == -1:
            # reboot target when timeout
            target.logger.error('reboot for timeout')
            dict['type'] = 'timeout'
            dict['error'] = 'infer timeout'
            conn.sendall(json.dumps(dict).encode())

            # reboot after login
            target.s0.run_cmd('root')
            target.s0.run_cmd('')
            target.s0.run_cmd('reboot')
            time.sleep(20)
        else:
            dict['time'] = float(ret.split('\n')[-2].split()[1])
            conn.sendall(json.dumps(dict).encode())
            dummy = conn.recv(1024)

            # send outputs
            for i in range(output_num):
                file = os.path.join(case_dir, f'nncase_result_{i}.bin')
                file_size = os.path.getsize(file)
                conn.sendall(str(file_size).encode())
                dummy = conn.recv(1024)

                target.logger.debug('send begin: file = {0}, size = {1}'.format(file, file_size))
                with open(file, 'rb') as f:
                    conn.sendall(f.read())
                target.logger.debug('send end')
                dummy = conn.recv(1024)
            target.logger.debug('infer finish')
        conn.close()
        target.clear_queue.put(case_dir)


def clear_worker(q):
    while True:
        case_dir = q.get()
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)


def main():
    # default config
    config = '''
    ip = '10.99.105.216'
    port = 10000
    nfs = '/data/nfs'
    [k230]
    working_dir = '/sharefs'
    separator = '>'
    uart0 = '/dev/ttyUSB0'
    baudrate0 = 115200
    uart1 = '/dev/ttyUSB1'
    baudrate1 = 115200
    '''

    # args
    parser = argparse.ArgumentParser(prog="nuc_proxy")
    parser.add_argument("--config", help='config string or file', type=str, default=config)
    args = parser.parse_args()
    size = 256
    cfg = {}
    dict = {}

    # load config
    if os.path.isfile(args.config):
        cfg = toml.load(args.config)
    else:
        cfg = toml.loads(args.config)

    # clear thread
    clear_queue = queue.Queue(maxsize=size)
    clear_thread = threading.Thread(target=clear_worker, args=(clear_queue,))
    clear_thread.start()

    # start server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((cfg['ip'], cfg['port']))
    server_socket.listen(size)
    while True:
        conn, addr = server_socket.accept()

        # recv target name
        conn.sendall(f"pls send your target".encode())
        info = conn.recv(1024)
        target_dict = json.loads(info.decode())
        target_name = target_dict['target']

        # create target instance
        if target_name not in dict:
            target = Target(target_name, cfg[target_name], cfg['nfs'], clear_queue)
            infer_thread = threading.Thread(target=infer_worker, args=(target,))
            infer_thread.start()
            dict[target_name] = target

        # start recv thread
        recv_thread = threading.Thread(target=recv_worker, args=(conn, dict[target_name]))
        recv_thread.start()


if __name__ == '__main__':
    main()
