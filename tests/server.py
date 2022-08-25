import os
import stat
import socket
import base64
import json
import threading
import queue
import logging
import telnetlib
import time
import argparse

class TelnetClient():
    def __init__(self,):
        self.tn = telnetlib.Telnet()

    def login(self, ip, username, password):
        try:
            self.tn.open(ip, port=23)
        except:
            logging.info('telnet %s failed' % ip)
            return False

        self.tn.read_until(b'login: ', timeout=10)
        self.tn.write(username.encode('ascii') + b'\n')

        command_result = self.tn.read_very_eager().decode('ascii')
        if 'Login incorrect' not in command_result:
            logging.info('%s login succeed' % ip)
            return True
        else:
            logging.error('%s login failed' % ip)
            return False

    def execute(self,command):
        self.tn.write(command.encode('ascii')+b'\n')
        self.tn.read_until(b'[root@canaan')

    def logout(self):
        self.tn.write(b"exit\n")

def Consumer(kpu_target, kpu_ip, kpu_username, kpu_password, nfsroot, q):
    while True:
        cmd = './'
        msg_dict = q.get()

        target_root = os.path.join(nfsroot, kpu_target)
        if not os.path.exists(target_root):
            os.makedirs(target_root)

        # app
        file = 'nncase_test_ci'
        full_file = os.path.join(target_root, file)
        with open(full_file, 'wb') as f:
            f.write(base64.b64decode(msg_dict['app'].encode('utf-8')))
        os.chmod(full_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        cmd = cmd + file

        # kmodel
        file = 'test.kmodel'
        with open(os.path.join(target_root, file), 'wb') as f:
            f.write(base64.b64decode(msg_dict['kmodel'].encode('utf-8')))
        cmd = cmd + ' ' + file

        # inputs
        inputs = msg_dict['inputs']
        for i in range(len(inputs)):
            file = f'input_0_{i}.bin'
            with open(os.path.join(target_root, file), 'wb') as f:
                f.write(base64.b64decode(msg_dict['inputs'][i].encode('utf-8')))
            cmd = cmd + ' ' + file

        # telnet to target devcie to nncase infer
        telnet_client = TelnetClient()
        telnet_client.login(kpu_ip, kpu_username, kpu_password)
        telnet_client.execute(f'cd /mnt/nfsroot/{kpu_target}')
        telnet_client.execute('sync')
        telnet_client.execute(cmd)
        telnet_client.execute('sync')
        telnet_client.logout()

        # outputs
        reply_dict = {}
        outputs = []
        for i in range(msg_dict['output_num']):
            with open(os.path.join(target_root, f'nncase_result_{i}.bin'), 'rb') as f:
                outputs.append(base64.b64encode(f.read()).decode('utf-8'))

        reply_dict['outputs'] = outputs
        payload = json.dumps(reply_dict).encode('utf-8')

        header = str(len(payload)).encode('utf-8')
        msg_dict['conn'].sendall(header)
        reply = msg_dict['conn'].recv(1024)

        msg_dict['conn'].sendall(payload)
        msg_dict['conn'].close()


def Producer(q, conn):
    in_data = bytes()
    header = conn.recv(1024)
    payload_size = int(header.decode('utf-8'))
    conn.sendall("server start recv payload".encode())

    received_size = 0
    while received_size < payload_size:
        data = conn.recv(4096)
        in_data += data
        received_size += len(data)

    msg_dict = json.loads(in_data.decode('utf-8'))
    msg_dict['conn'] = conn
    q.put(msg_dict)

def main():
    parser = argparse.ArgumentParser(prog="kendryte_CI_server_app")
    parser.add_argument("--kpu_target", help='target, such as k510', type=str, default='k510')
    parser.add_argument("--kpu_ip", help='kpu deivce ip address', type=str, default='192.168.1.5')
    parser.add_argument("--kpu_username", help='kpu device usernmae', type=str, default='root')
    parser.add_argument("--kpu_password", help='kpu device password', type=str, default='')
    parser.add_argument("--nfsroot", help='nfsroot on pc', type=str, default='/home/share/nfsroot')
    parser.add_argument("--port", help='nfsroot on pc', type=int, default=51000)
    args = parser.parse_args()
    size = 32
    q = queue.Queue(maxsize=size)

    # start comsumer thread
    t_consumer = threading.Thread(target=Consumer, args=(args.kpu_target, args.kpu_ip, args.kpu_username, args.kpu_password, args.nfsroot, q))
    t_consumer.start()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', args.port))
    server_socket.listen(size)
    while True:
        conn, addr = server_socket.accept()
        print('connnected by {0}'.format(addr))
        t_producer = threading.Thread(target=Producer, args=(q, conn))
        t_producer.start()

if __name__ == '__main__':
    main()