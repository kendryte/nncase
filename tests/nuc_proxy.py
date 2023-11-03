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
import struct
import time
import toml
from typing import Tuple


class MySerial:
    def __init__(self, port, baudrate, separator, logger):
        self.s = None
        self.port = port
        self.baudrate = baudrate
        self.separator = separator
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

    # There is something wrong for self.s.read_until() api, refer to https://github.com/pyserial/pyserial/issues/181
    def read_until(self, expected: str, size=None) -> Tuple[str, bool]:
        self.logger.debug('read begin: expected = {0}'.format(expected))
        if isinstance(expected, str):
            expected = expected.encode()
        lenterm = len(expected)
        data = bytearray()
        expired = True
        timeout = serial.Timeout(self.s.timeout)
        while True:
            c = self.s.read(1)
            if c:
                data += c
                if data[-lenterm:] == expected:
                    expired = False
                    break
                if size is not None and len(data) >= size:
                    break
            else:
                break
            if timeout.expired():
                break
        self.logger.debug('read end: data = {0}, size = {1}'.format(data, len(data)))
        return bytes(data).decode(), expired

    def run_cmd(self, cmd, expected=''):
        data = ''
        expired = False
        self.open()
        self.write(cmd)

        str = expected if expected != '' else self.separator
        data, expired = self.read_until(str)
        self.close()
        return data, expired


class Target:
    def __init__(self, name, cfg, nfs, clear_queue):
        self.name = name
        self.infer_queue = queue.Queue(maxsize=clear_queue.maxsize)
        self.clear_queue = clear_queue
        self.username = cfg['username']
        self.password = cfg['password']
        self.working_dir = cfg['working_dir']

        # nfs_dir
        self.nfs_dir = os.path.join(nfs, name)
        if not os.path.exists(self.nfs_dir):
            os.makedirs(self.nfs_dir)

        # logging
        mylogger = logging.getLogger()
        mylogger.setLevel(logging.DEBUG)
        rf_handler = logging.handlers.RotatingFileHandler(
            f'nuc_proxy_{name}.log', mode='a', maxBytes=32 * 1024 * 1024, backupCount=10)
        rf_handler.setLevel(logging.DEBUG)
        rf_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        mylogger.addHandler(rf_handler)
        self.logger = mylogger

        # serial
        self.s0 = MySerial(cfg['uart0'], cfg['baudrate0'], cfg['separator0'], self.logger)
        self.s1 = MySerial(cfg['uart1'], cfg['baudrate1'], cfg['separator1'], self.logger)

    def reboot(self):
        self.s0.run_cmd('reboot')
        time.sleep(30)


def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_file(conn, case_dir, logger):
    send_msg(conn, f"pls send file info".encode())
    header = recv_msg(conn)
    file_dict = json.loads(header.decode())
    file_name = file_dict['file_name']
    file_size = file_dict['file_size']
    logger.debug('recv begin: file = {0}, size = {1}'.format(file_name, file_size))
    send_msg(conn, f"pls send {file_name}".encode())

    full_file = os.path.join(case_dir, file_name)
    with open(full_file, 'wb') as f:
        data = recvall(conn, file_size)
        f.write(data)

    os.chmod(full_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    logger.debug('recv end')
    return file_name


def recv_worker(conn, target):
    # recv header
    send_msg(conn, f"pls send header".encode())
    header = recv_msg(conn)
    dict = json.loads(header.decode())
    # print('recv_worker: dict = {}'.format(dict))
    new_case = dict['case'] + str(int(time.time()))
    target.logger.info("test case = {0}".format(new_case))
    case_dir = os.path.join(target.nfs_dir, new_case)
    os.makedirs(case_dir)
    file_num = dict['app'] + dict['kmodel'] + dict['inputs'] + dict['description']

    # recv all kinds of files(app + kmodel + inputs)
    cmds = f'cd {target.working_dir}/{target.name}/{new_case};./'
    for i in range(file_num):
        file = recv_file(conn, case_dir, target.logger)
        if i == 0:
            cmds = cmds + file
        else:
            cmds = cmds + ' ' + file

    target.logger.debug("cmds = {0}".format(cmds))
    target.infer_queue.put((dict['cfg_cmds'], cmds, conn, case_dir, dict['outputs']))


def infer_worker(target):
    while True:
        cfg_cmds, run_cmds, conn, case_dir, output_num = target.infer_queue.get()
        test_case = os.path.basename(case_dir)
        s1_separator = test_case + target.s1.separator
        ret = ''
        timeout = False

        # try to login serial
        target.s0.run_cmd(target.username)
        target.s0.run_cmd(target.password)
        target.s1.run_cmd('q\r')

        msg = []
        if len(cfg_cmds) == 0:
            for cmd in run_cmds.split(';'):
                ret, timeout = target.s1.run_cmd(cmd, s1_separator)
            msg.append(ret)
        else:
            for cfg_cmd in cfg_cmds:
                target.s0.run_cmd(cfg_cmd)
                for cmd in run_cmds.split(';'):
                    ret, timeout = target.s1.run_cmd(cmd, s1_separator)
                msg.append(ret)

        # infer result
        dict = {'type': 'finish', 'msg': ''}
        ret = msg[0]
        if ret.find('terminate') != -1 or ret.find('Exception') != -1:
            err = 'infer exception'
            target.logger.error(err)
            dict['type'] = 'exception'
            dict['msg'] = err
            send_msg(conn, json.dumps(dict).encode())
            target.reboot()
        elif timeout:
            err = 'infer timeout'
            target.logger.error(err)
            dict['type'] = 'timeout'
            dict['msg'] = err
            send_msg(conn, json.dumps(dict).encode())
            target.reboot()
        else:
            # send header
            dict['type'] = 'finish'
            dict['msg'] = msg
            send_msg(conn, json.dumps(dict).encode())

            # send outputs
            dummy = recv_msg(conn)
            for i in range(output_num):
                file_dict = {}
                file = os.path.join(case_dir, f'nncase_result_{i}.bin')
                file_size = os.path.getsize(file)
                file_dict['file_size'] = file_size
                send_msg(conn, json.dumps(file_dict).encode())
                dummy = recv_msg(conn)

                target.logger.debug('send begin: file = {0}, size = {1}'.format(file, file_size))
                with open(file, 'rb') as f:
                    conn.sendall(f.read())
                target.logger.debug('send end')
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
    ip = '10.100.105.239'
    port = 10000
    nfs = '/data/nfs'
    [k230]
    username = 'root'
    password = ''
    working_dir = '/sharefs'
    uart0 = '/dev/ttyUSB0'
    baudrate0 = 115200
    separator0 = ']#'
    uart1 = '/dev/ttyUSB1'
    baudrate1 = 115200
    separator1 = '>'
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
        send_msg(conn, f"pls send your target".encode())
        info = recv_msg(conn)
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
