import os
import argparse
import stat
import socket
import json
import threading
import queue
import logging
import logging.handlers
import telnetlib
import time
import serial
import toml


class TelnetClient():
    def __init__(self, mylogger):
        self.tn = telnetlib.Telnet()
        self.logger = mylogger
        self.ip = '10.99.105.216'
        self.timeout = 60

    def login(self, ip, username, password):
        try:
            self.tn.open(ip, port=23)
        except:
            self.logger.error('telnet {0} failed'.format(ip))
            return False

        self.ip = ip
        self.tn.read_until(b'login: ', timeout=self.timeout)
        self.tn.write(username.encode() + b'\r\n')

        cmd_result = self.tn.read_very_eager().decode()
        if 'Login incorrect' not in cmd_result:
            self.logger.info('{0} login succeed'.format(ip))
            return True
        else:
            self.logger.error('{0} login failed'.format(ip))
            return False

    def logout(self):
        self.tn.close()
        self.logger.info('{0} logout succeed'.format(self.ip))

    def execute(self, cmd, flag):
        self.logger.debug('execute: cmd = {0}, flag = {1}'.format(cmd, flag))
        self.tn.write(cmd.encode() + b'\r\n')
        cmd_result = self.tn.read_until(flag.encode(), timeout=self.timeout).decode()
        if flag not in cmd_result:
            # time out
            self.tn.write(telnetlib.IP)
            cmd_result = f'timeout for {self.timeout} seconds'
            self.logger.error('execute {0} failed: {1}'.format(cmd, cmd_result))
            return cmd_result, False
        else:
            self.tn.write('echo $?'.encode() + b'\r\n')
            cmd_status = self.tn.read_until(flag.encode(), self.timeout).decode()
            if cmd_status.find('\r\n0\r\n') == -1:
                self.logger.error('execute {0} failed: {1}'.format(cmd, cmd_result))
                return cmd_result, False
            else:
                return cmd_result, True


def recv_file(conn, target_root, mylogger):
    conn.sendall(f"pls send file info".encode())
    header = conn.recv(1024)
    file_dict = json.loads(header.decode())
    file_name = file_dict['file_name']
    file_size = file_dict['file_size']
    mylogger.debug('recv: file = {0}, size = {1}'.format(file_name, file_size))
    conn.sendall(f"pls send {file_name}".encode())

    full_file = os.path.join(target_root, file_name)
    with open(full_file, 'wb') as f:
        recv_size = 0
        while recv_size < file_size:
            slice = conn.recv(4096)
            f.write(slice)
            recv_size += len(slice)

    # conn.sendall(f"recv {file_name} succeed".encode())
    os.chmod(full_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return file_name


class MySerial:
    def __init__(self, target, port, baudrate):
        self.s = None
        self.target = target
        self.port = port
        self.baudrate = baudrate
        self.timeout = 60
        # self.log_file = '{0}_uart.log'.format(target)

    def open(self):
        self.s = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
        # if (self.s.isOpen()):
        #     print('open {0} succeed'.format(self.port))
        # else:
        #     print('open {0} failed'.format(self.port))

    def close(self):
        self.s.close()

    def write(self, cmd):
        self.s.write(cmd.encode())

    # def readall(self):
    #     f = open(self.log_file, 'a')
    #     while True:
    #         # line = self.s.readline().decode()
    #         #print('readline from serial {0}: {1}'.format(self.port, line))
    #         print('{0}'.format(line), end='')
    #         f.write(line)
    #         if '}\n' in line:
    #             break
    #     f.close()

    def read_until(self, expected):
        data = self.s.read_until(expected).decode()
        # print('read: {0}'.format(data), end='')
        return data


def run_cmds(s, cmds):
    s.open()

    for cmd in cmds.split('&&'):
        s.write(cmd + '\r')

    data = s.read_until(b'}')
    s.close()

    return data


def Consumer(target, q, ip, username, password, working_dir, uart, baudrate):
    # logging
    mylogger = logging.getLogger()
    mylogger.setLevel(logging.DEBUG)
    rf_handler = logging.handlers.RotatingFileHandler(
        f'nuc_proxy_{target}.log', mode='a', maxBytes=32 * 1024 * 1024, backupCount=10)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    mylogger.addHandler(rf_handler)
    telnet_client = TelnetClient(mylogger)

    # serial
    s = MySerial(target, uart, baudrate)

    while True:
        conn = q.get()

        # recv header
        conn.sendall(f"pls send cmd".encode())
        header = conn.recv(1024)
        dict = json.loads(header.decode())
        cmd = dict['cmd']
        mylogger.info("cmd = {0}".format(dict['cmd']))

        # run cmds on device
        cmds = 'cd {0}/{1} && {2}'.format(working_dir, target, cmd)
        ret_str = run_cmds(s, cmds)

        # infer result
        if ret_str.find('terminate') != -1 or ret_str.find('Exception') != -1:
            conn.sendall(f'infer exception: {ret_str}'.encode())
        elif ret_str.find('}') == -1:
            # reboot target when timeout
            conn.sendall(f'infer timeout'.encode())
            mylogger.error('reboot {0}({1}) for timeout'.format(target, ip))
            telnet_client.login(ip, username, password)
            flag = f'[{username}@canaan ~ ]'
            telnet_client.execute('reboot', flag)
            telnet_client.logout()
            time.sleep(30)
        else:
            conn.sendall(f'infer succeed'.encode())
        conn.close()


def main():
    # default config
    config = '''
    [k230]
    ip = '192.168.1.230'
    username = 'root'
    password = ''
    working_dir = '/sharefs'
    uart = '/dev/ttyUSB1'
    baudrate = 2500000
    '''

    # args
    parser = argparse.ArgumentParser(prog="nuc_proxy")
    parser.add_argument("--port", help='listening port', type=int, default=10001)
    parser.add_argument("--config", help='config str or file', type=str, default=config)
    args = parser.parse_args()
    size = 256

    # load config
    cfg = {}
    if os.path.isfile(args.config):
        cfg = toml.load(args.config)
    else:
        cfg = toml.loads(args.config)

    # create queue and thread
    for k in cfg:
        q = queue.Queue(maxsize=size)
        t_consumer = threading.Thread(target=Consumer, args=(
            k, q, cfg[k]["ip"], cfg[k]['username'], cfg[k]['password'], cfg[k]['working_dir'], cfg[k]['uart'], cfg[k]['baudrate']))
        t_consumer.start()
        cfg[k]['queue'] = q

    # server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', args.port))
    server_socket.listen(size)
    while True:
        conn, addr = server_socket.accept()

        # recv target
        conn.sendall(f"pls send your target".encode())
        info = conn.recv(1024)
        dict = json.loads(info.decode())
        cfg[dict['target']]['queue'].put(conn)


if __name__ == '__main__':
    main()
