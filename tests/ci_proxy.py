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

    os.chmod(full_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return file_name


def Consumer(target, q, nfs_root, ip, port):
    # create target root
    target_root = os.path.join(nfs_root, target)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    # logging
    mylogger = logging.getLogger()
    mylogger.setLevel(logging.DEBUG)
    rf_handler = logging.handlers.RotatingFileHandler(
        f'ci_proxy_{target}.log', mode='a', maxBytes=32 * 1024 * 1024, backupCount=10)
    # rf_handler.setLevel(logging.INFO)
    mylogger.setLevel(logging.DEBUG)
    rf_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    mylogger.addHandler(rf_handler)

    # telnet_client = TelnetClient(mylogger)
    while True:
        cmd = './'
        conn = q.get()

        # recv header
        conn.sendall(f"pls send header".encode())
        header = conn.recv(1024)
        header_dict = json.loads(header.decode())
        mylogger.info("test case = {0}".format(header_dict['case']))
        file_num = header_dict['app'] + header_dict['kmodel'] + header_dict['inputs']

        # recv all kinds of files(app + kmodel + inputs)
        for i in range(file_num):
            file = recv_file(conn, target_root, mylogger)
            if i == 0:
                cmd = cmd + file
            else:
                cmd = cmd + ' ' + file

        # print('cmd = {0}'.format(cmd))

        # connect nuc_proxy server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, int(port)))

        # send target
        dummy = client_socket.recv(1024)
        target_dict = {}
        target_dict['target'] = target
        client_socket.sendall(json.dumps(target_dict).encode())

        # send header
        dummy = client_socket.recv(1024)
        dict = {}
        dict['cmd'] = cmd
        client_socket.sendall(json.dumps(dict).encode())

        infer_result = client_socket.recv(1024).decode()
        client_socket.close()
        # print('infer_result = {0}'.format(infer_result))
        if infer_result.find('succeed') == -1:
            conn.sendall(f'infer failed on {target} board: {infer_result}'.encode())
        else:
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

        conn.close()


def main():
    # args
    parser = argparse.ArgumentParser(prog="ci_proxy")
    parser.add_argument("--ci_proxy_port", help='listening port of ci_proxy',
                        type=int, default=10000)
    parser.add_argument("--nfs_root", help='nfs root on pc', type=str, default='/data/nfs')
    parser.add_argument("--nuc_proxy_ip", help='ip of nuc_proxy', type=str, default='localhost')
    parser.add_argument("--nuc_proxy_port", help='listening port of nuc_proxy',
                        type=int, default=10001)

    args = parser.parse_args()

    dict = {}
    size = 256

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', args.ci_proxy_port))
    server_socket.listen(size)
    while True:
        conn, addr = server_socket.accept()

        # recv target
        conn.sendall(f"pls send your target".encode())
        info = conn.recv(1024)
        target_dict = json.loads(info.decode())
        target = target_dict['target']

        if target not in dict:
            q = queue.Queue(maxsize=size)
            t_consumer = threading.Thread(target=Consumer, args=(
                target, q, args.nfs_root, args.nuc_proxy_ip, args.nuc_proxy_port))
            t_consumer.start()
            dict[target] = q

        dict[target].put(conn)


if __name__ == '__main__':
    main()
