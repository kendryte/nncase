import nncase
import numpy as np
from compare_util import *
import copy


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
