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

    conn.sendall(f"recv {file_name} succeed".encode())
    os.chmod(full_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return file_name

def Consumer(kpu_target, kpu_ip, kpu_username, kpu_password, nfsroot, q, mylogger):
    # create target root
    target_root = os.path.join(nfsroot, kpu_target)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    telnet_client = TelnetClient(mylogger)
    while True:
        cmd = './'
        conn = q.get()

        # recv header
        header = conn.recv(1024)
        header_dict = json.loads(header.decode())
        mylogger.info("test case = {0}".format(header_dict['case']))
        file_num = header_dict['app'] + header_dict['kmodel'] + header_dict['inputs']
        conn.sendall(f"pls send {file_num} files".encode())

        # recv all kinds of files(app + kmodel + inputs)
        for i in range(file_num):
            file = recv_file(conn, target_root, mylogger)
            if i == 0:
                cmd = cmd + file
            else:
                cmd = cmd + ' ' + file

        # telnet target devcie to infer
        telnet_client.login(kpu_ip, kpu_username, kpu_password)
        flag = f'/mnt/{kpu_target} ]$'
        cmd_result, cmd_status = telnet_client.execute(f'cd /mnt/{kpu_target} && {cmd}', flag)
        if cmd_status:
            conn.sendall(f'infer succeed'.encode())
            dummy = conn.recv(1024)

            # send outputs
            for i in range(header_dict['outputs']):
                file = os.path.join(target_root, f'nncase_result_{i}.bin')
                file_size = os.path.getsize(file)
                conn.sendall(str(file_size).encode())
                dummy = conn.recv(1024)

                with open(file, 'rb') as f:
                    conn.sendall(f.read())
                dummy = conn.recv(1024)
                mylogger.debug('send: file = {0}, size = {1}'.format(file, file_size))
        else:
            conn.sendall(f'infer failed on {kpu_target} board: {cmd_result}'.encode())
        conn.close()

        if 'timeout' not in cmd_result:
            telnet_client.logout()
        else:
            # reboot kpu_target when timeout
            telnet_client.logout()
            mylogger.error('reboot {0}({1}) for timeout'.format(kpu_target, kpu_ip))
            telnet_client.login(kpu_ip, kpu_username, kpu_password)
            flag = f'[{kpu_username}@canaan ~ ]$'
            telnet_client.execute('reboot', flag)
            telnet_client.logout()
            time.sleep(60)

def main():
    # args
    parser = argparse.ArgumentParser(prog="ci_proxy")
    parser.add_argument("--kpu_target", help='kpu device target', type=str, default='k510')
    parser.add_argument("--kpu_ip", help='kpu deivce ip address', type=str, default='10.99.105.216')
    parser.add_argument("--kpu_username", help='kpu device usernmae', type=str, default='root')
    parser.add_argument("--kpu_password", help='kpu device password', type=str, default='')
    parser.add_argument("--nfsroot", help='nfsroot on pc', type=str, default='/data/nfs')
    parser.add_argument("--port", help='listenning port of ci_proxy', type=int, default=51000)
    args = parser.parse_args()

    # logging
    mylogger = logging.getLogger()
    mylogger.setLevel(logging.DEBUG)
    rf_handler = logging.handlers.RotatingFileHandler(f'ci_proxy_{args.kpu_target}.log', mode='a', maxBytes=32 * 1024 * 1024, backupCount=10)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    mylogger.addHandler(rf_handler)

    # producer
    size = 256
    q = queue.Queue(maxsize=size)

    # comsumer
    t_consumer = threading.Thread(target=Consumer, args=(args.kpu_target, args.kpu_ip, args.kpu_username, args.kpu_password, args.nfsroot, q, mylogger))
    t_consumer.start()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', args.port))
    server_socket.listen(size)
    while True:
        conn, addr = server_socket.accept()
        q.put(conn)

if __name__ == '__main__':
    main()