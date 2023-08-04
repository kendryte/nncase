import os
import argparse
import stat
import socket
import json
import threading
import queue
import logging
import logging.handlers
import time
import serial
import toml

class MySerial:
    def __init__(self, logger, port, baudrate):
        self.s = None
        self.logger = logger
        self.port = port
        self.baudrate = baudrate
        self.timeout = 20

    def open(self):
        self.logger.debug('open serial begin')
        self.s = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
        self.logger.debug('open serial end')
        # if (self.s.isOpen()):
        #     print('open {0} succeed'.format(self.port))
        # else:
        #     print('open {0} failed'.format(self.port))

    def close(self):
        self.logger.debug('close serial begin')
        self.s.close()
        self.logger.debug('close serial end')

    def write(self, cmd):
        self.logger.debug('write begin')
        self.s.write(cmd.encode())
        self.logger.debug('write end')

    def read_until(self, expected):
        self.logger.debug('read begin')
        data = self.s.read_until(expected).decode()
        self.logger.debug('read end: data = {0}'.format(data))
        return data

    def run_cmd(self, cmd):
        self.open()
        self.write(cmd + '\r')
        self.close()

    def run_cmds(self, cmds):
        self.open()

        for cmd in cmds.split('&&'):
            self.write(cmd + '\r')

        data = self.read_until(b'}')
        self.close()
        return data

def Consumer(target, q, working_dir, uart0, baudrate0, uart1, baudrate1):
    # logging
    mylogger = logging.getLogger()
    mylogger.setLevel(logging.INFO)
    rf_handler = logging.handlers.RotatingFileHandler(
        f'nuc_proxy_{target}.log', mode='a', maxBytes=32 * 1024 * 1024, backupCount=10)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    mylogger.addHandler(rf_handler)

    # serial
    s0 = MySerial(mylogger, uart0, baudrate0)
    s1 = MySerial(mylogger, uart1, baudrate1)

    while True:
        conn = q.get()

        # recv header
        conn.sendall(f"pls send cmd".encode())
        header = conn.recv(1024)
        dict = json.loads(header.decode())
        cmd = dict['cmd']
        mylogger.debug("cmd = {0}".format(dict['cmd']))

        # run cmds on device
        cmds = 'cd {0}/{1} && {2}'.format(working_dir, target, cmd)
        ret_str = s1.run_cmds(cmds)

        # infer result
        if ret_str.find('terminate') != -1 or ret_str.find('Exception') != -1:
            err=f'infer exception: {ret_str}'
            mylogger.error('infer exception')
            conn.sendall(err[0:1024].encode())
        elif ret_str.find('}') == -1:
            # reboot target when timeout
            conn.sendall(f'infer timeout'.encode())
            mylogger.error('reboot {0} for timeout'.format(target))
            s0.run_cmd('reboot')
            time.sleep(20)
        else:
            conn.sendall(f'infer succeed'.encode())
            mylogger.debug('infer succeed')
        conn.close()

def main():
    # default config
    config = '''
    [k230]
    working_dir = '/sharefs'
    uart0 = '/dev/ttyUSB0'
    baudrate0 = 115200
    uart1 = '/dev/ttyUSB1'
    baudrate1 = 115200
    '''

    # args
    parser = argparse.ArgumentParser(prog="nuc_proxy")
    parser.add_argument("--ip", help='ip to connect', type=str, default='10.99.105.216')
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
            k, q, cfg[k]['working_dir'], cfg[k]['uart0'], cfg[k]['baudrate0'], cfg[k]['uart1'], cfg[k]['baudrate1']))
        t_consumer.start()
        cfg[k]['queue'] = q

    # server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.ip, args.port))
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
