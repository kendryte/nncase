import nncase
import numpy as np
from compare_util import *
import copy
import socket
import json

def get_topK(info, k, result):
    tmp = copy.deepcopy(result)
    predictions = np.squeeze(tmp)
    topK = predictions.argsort()[-k:]
    return topK


def sim_run(kmodel, data, paths, target, model_type, model_shape):
    sim = nncase.Simulator()
    sim.load_model(kmodel)
    if(model_type != "tflite" and model_shape[-1] != 3):
        new_data = np.transpose(data[0], [0, 3, 1, 2]).astype(np.float32)
    else:
        new_data = data[0].astype(np.float32)
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(new_data))
    sim.run()
    result = sim.get_output_tensor(0).to_numpy()
    tmp = []
    tmp.append((data[1], get_topK(target, 1, result)))
    with open(paths[-1][1], 'a') as f:
        for i in range(len(tmp)):
            f.write(tmp[i][0].split("/")[-1] + " " + str(tmp[i][1][0]) + '\n')
    return tmp

def on_board_run(kmodel, data, paths, target, port, case, nncase_test_ci, input_num, output_num, model_type, model_shape):
    # connect server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', int(port)))

    # send header
    header_dict = {}
    header_dict['case'] = case
    header_dict['app'] = 1
    header_dict['kmodel']= 1
    header_dict['inputs'] = 1
    header_dict['outputs'] = 1
    client_socket.sendall(json.dumps(header_dict).encode())
    dummy = client_socket.recv(1024)

    # send app
    file_dict = {}
    file_dict['file_name'] = os.path.basename(nncase_test_ci)
    file_dict['file_size'] = os.path.getsize(nncase_test_ci)
    client_socket.sendall(json.dumps(file_dict).encode())
    dummy = client_socket.recv(1024)
    with open(nncase_test_ci, 'rb') as f:
        client_socket.sendall(f.read())
    dummy = client_socket.recv(1024)

    # send kmodel
    file_dict['file_name'] = 'test.kmodel'
    file_dict['file_size'] = len(kmodel)
    client_socket.sendall(json.dumps(file_dict).encode())
    dummy = client_socket.recv(1024)
    client_socket.sendall(kmodel)
    dummy = client_socket.recv(1024)

    # send inputs
    for i in range(input_num):
        if(model_type != "tflite" and model_shape[-1] != 3):
            new_data = np.transpose(data[0], [0, 3, 1, 2]).astype(np.float32)
        else:
            new_data = data[0].astype(np.float32)

        data_in_bytes = new_data.tobytes()
        file_dict['file_name'] = f'input_0_{i}.bin'
        file_dict['file_size'] = len(data_in_bytes)
        client_socket.sendall(json.dumps(file_dict).encode())
        dummy = client_socket.recv(1024)
        client_socket.sendall(data_in_bytes)
        dummy = client_socket.recv(1024)

    # infer result
    cmd_result = client_socket.recv(1024).decode()
    if cmd_result.find('succeed') != -1:
        client_socket.sendall(f"pls send outputs".encode())

        # recv outputs
        for i in range(output_num):
            header = client_socket.recv(1024)
            file_size = int(header.decode())
            client_socket.sendall(f"pls send nncase_result_{i}.bin".encode())

            recv_size = 0
            buffer = bytearray(file_size)
            while recv_size < file_size:
                slice = client_socket.recv(4096)
                buffer[recv_size:] = slice
                recv_size += len(slice)

            # result
            result = np.frombuffer(buffer, dtype=np.float32)
            tmp = []
            tmp.append((data[1], get_topK(target, 1, result)))
            with open(paths[-1][1], 'a') as f:
                for i in range(len(tmp)):
                    f.write(tmp[i][0].split("/")[-1] + " " + str(tmp[i][1][0]) + '\n')

            client_socket.sendall(f"recv nncase_result_{i}.bin succeed".encode())

        client_socket.close()
        return tmp
    else:
        client_socket.close()
        raise Exception(f'{cmd_result}')