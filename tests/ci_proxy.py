import os
import argparse
import stat
import socket
import json
import threading
import queue
import logging
import telnetlib
import time

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
        self.tn.write(username.encode() + b'\n')

        command_result = self.tn.read_very_eager().decode()
        if 'Login incorrect' not in command_result:
            logging.info('%s login succeed' % ip)
            return True
        else:
            logging.error('%s login failed' % ip)
            return False

    def execute(self, command, flag):
        self.tn.write(command.encode() + b'\n')
        cmd_result = self.tn.read_until(flag.encode()).decode()

        self.tn.write('echo $?'.encode() + b'\n')
        cmd_status = self.tn.read_until(flag.encode()).decode()
        if cmd_status.find('\r\n0\r\n') == -1:
            return cmd_result, False
        else:
            return cmd_result, True

    def logout(self):
        self.tn.write(b"exit\n")

def get_file(conn, target_root):
    header = conn.recv(1024)
    file_dict = json.loads(header.decode())
    file_name = file_dict['file_name']
    file_size = file_dict['file_size']
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

def Consumer(kpu_target, kpu_ip, kpu_username, kpu_password, nfsroot, q):
    while True:
        cmd = './'
        conn = q.get()

        # create target root
        target_root = os.path.join(nfsroot, kpu_target)
        if not os.path.exists(target_root):
            os.makedirs(target_root)

        # recv file_num
        header = conn.recv(1024)
        file_num_dict = json.loads(header.decode())
        file_num = file_num_dict['app'] + file_num_dict['kmodel'] + file_num_dict['inputs']
        conn.sendall(f"pls send {file_num} files".encode())

        # recv all kinds of files(app + kmodel + inputs)
        for i in range(file_num):
            file = get_file(conn, target_root)
            if i == 0:
                cmd = cmd + file
            else:
                cmd = cmd + ' ' + file

        # telnet target devcie to infer
        telnet_client = TelnetClient()
        telnet_client.login(kpu_ip, kpu_username, kpu_password)
        flag = f'/mnt/{kpu_target} ]$'
        telnet_client.execute(f'cd /mnt/{kpu_target}', flag)
        telnet_client.execute('sync', flag)
        cmd_result, cmd_status = telnet_client.execute(cmd, flag)
        telnet_client.execute('sync', flag)

        if cmd_status:
            conn.sendall(f'infer succeed'.encode())
            dummy = conn.recv(1024)

            # send outputs
            for i in range(file_num_dict['outputs']):
                file = os.path.join(target_root, f'nncase_result_{i}.bin')
                file_size = os.path.getsize(file)
                conn.sendall(str(file_size).encode())
                dummy = conn.recv(1024)

                with open(file, 'rb') as f:
                    conn.sendall(f.read())
                dummy = conn.recv(1024)
        else:
            conn.sendall(f'infer failed on {kpu_target} board: {cmd_result}'.encode())

        telnet_client.execute('rm *', flag)
        telnet_client.execute('sync', flag)
        telnet_client.logout()
        conn.close()

def main():
    parser = argparse.ArgumentParser(prog="kendryte_ci_proxy")
    parser.add_argument("--kpu_target", help='kpu device target', type=str, default='k510')
    parser.add_argument("--kpu_ip", help='kpu deivce ip address', type=str, default='10.99.105.216')
    parser.add_argument("--kpu_username", help='kpu device usernmae', type=str, default='root')
    parser.add_argument("--kpu_password", help='kpu device password', type=str, default='')
    parser.add_argument("--nfsroot", help='nfsroot on pc', type=str, default='/data/nfs')
    parser.add_argument("--port", help='listenning port of ci_proxy', type=int, default=51000)
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
        # print('connnected by {0}'.format(addr))
        q.put(conn)

if __name__ == '__main__':
    main()