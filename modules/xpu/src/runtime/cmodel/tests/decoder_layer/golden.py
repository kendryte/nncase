import numpy as np
from torch import nn
import torch
from scipy.special import softmax
import os

def rmsnorm(x:np.ndarray, gamma:np.ndarray, beta:np.ndarray, eps=1e-05):
    x = torch.from_numpy(x)
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    output = output * torch.from_numpy(gamma) + torch.from_numpy(beta)
    return output.detach().numpy()

def layernorm(x:np.ndarray, gamma:np.ndarray, beta:np.ndarray):
    return nn.functional.layer_norm(torch.from_numpy(x), [x.shape[-1]], torch.from_numpy(gamma), torch.from_numpy(beta)).detach().numpy()

def swish(x:np.ndarray):
    return x / (1 + np.exp(-x))

def gather(data:np.ndarray, axis:int, indices:np.ndarray):
    return torch.gather(torch.from_numpy(data), axis, torch.from_numpy(indices)).detach().numpy()

const_dir = "/compiler/huochenghai/GNNE/rebuild-ir/nncase/modules/xpu/src/runtime/cmodel/tests/demo3/golden"

hidden_in = np.reshape(np.fromfile(os.path.join(const_dir,"input_0_0.bin"), dtype=np.float32), (1, 384, 8192))
attn_mask = np.reshape(np.fromfile(os.path.join(const_dir,"input_1_0.bin"), dtype=np.float32), (1, 1, 384, 384))
position_ids = np.reshape(np.fromfile(os.path.join(const_dir,"input_2_0.bin"), dtype=np.int64), (1, 384))

input_ln_gamma = np.reshape(np.fromfile(os.path.join(const_dir,"_input_layernorm_Mul_1_output_0_scale.bin"), dtype=np.float32), (8192))
input_ln_beta = np.reshape(np.fromfile(os.path.join(const_dir,"_input_layernorm_Mul_1_output_0_bias.bin"), dtype=np.float32), (8192))

gather1_data = np.reshape(np.fromfile(os.path.join(const_dir, "_self_attn_Gather_7_output_0_data.bin"), dtype=np.float32), (384,128))
gather2_data = np.reshape(np.fromfile(os.path.join(const_dir, "_self_attn_Gather_8_output_0_data.bin"), dtype=np.float32), (384,128))

WQ = np.reshape(np.fromfile(os.path.join(const_dir,"_self_attn_Transpose_output_0_weights.bin"), dtype=np.float32), (64, 8192, 128))
WK = np.reshape(np.fromfile(os.path.join(const_dir,"_self_attn_Transpose_1_output_0_weights.bin"), dtype=np.float32), (64, 8192, 128))
WV = np.reshape(np.fromfile(os.path.join(const_dir,"_self_attn_Transpose_2_output_0_weights.bin"), dtype=np.float32), (64, 8192, 128))
WO = np.reshape(np.fromfile(os.path.join(const_dir,"_self_attn_o_proj_MatMul_output_0_weights.bin"), dtype=np.float32), (8192, 8192))

post_ln_gamma = np.reshape(np.fromfile(os.path.join(const_dir,"_post_attention_layernorm_Mul_1_output_0_scale.bin"), dtype=np.float32), (8192))
post_ln_beta = np.reshape(np.fromfile(os.path.join(const_dir,"_post_attention_layernorm_Mul_1_output_0_bias.bin"), dtype=np.float32), (8192))

mlp_gate_weights = np.reshape(np.fromfile(os.path.join(const_dir,"_mlp_gate_proj_MatMul_output_0_weights.bin"), dtype=np.float32), (8192, 22016))
mlp_up_weights = np.reshape(np.fromfile(os.path.join(const_dir,"_mlp_up_proj_MatMul_output_0_weights.bin"), dtype=np.float32), (8192, 22016))
mlp_down_weights = np.reshape(np.fromfile(os.path.join(const_dir,"_mlp_down_proj_MatMul_output_0_weights.bin"), dtype=np.float32), (22016, 8192))

for index in range(1):
    # np.random.rand(8192).astype(np.float32), np.random.rand(8192).astype(np.float32)
    v0 = rmsnorm(hidden_in, input_ln_gamma, input_ln_beta) # f32[1,384,8192]
    # v0.tofile("v0.bin")
    v1 = np.reshape(v0, [1,1,384, 8192]) # f32[1,1,384,8192]
    v2 = (v1 @ WQ) # f32[1,64,384,128]
    # v2.tofile("v2.bin")
    v3 = gather1_data[position_ids] # f32[1,384,128]
    # v3.tofile("v3.bin")
    v4 = np.reshape(v3, (1,1,384,128))
    v5 = v2 * v4 # f32[1,64,384,128]
    # v5.tofile("v5.bin")
    v6 = v2[:,:,:,64:] # Slice(v3, const(i64[1] : {64L}), const(i64[1] : {9223372036854775807}), 3, 1) # f32[1,64,384,64]
    v7 = -(v6) # f32[1,64,384,64]
    v8 = v2[:,:,:,0:64] # Slice(v3, 0, 64, 3, 1) # f32[1,64,384,64]
    v9 = (v7, v8) # (f32[1,64,384,64], f32[1,64,384,64]

    v10 = np.concatenate(v9, -1) # f32[1,64,384,128]
    # v10.tofile("v10.bin")
    v11 = gather2_data[position_ids] # f32[1,384,128]
    v12 = np.reshape(v11, (1,1,384,128))
    v13 = v10 * v12 # f32[1,64,384,128]
    v14 = v5 + v13 # f32[1,64,384,128]
    v14.tofile("v14.bin")
    v15 = np.reshape(v0, [1,1,384, 8192]) # f32[1,1,384,8192]
    v16 = (v15 @ WK) # f32[1,64,384,128]
    v17 = v16 * v4 # f32[1,64,384,128]
    v18 = v16[:,:,:,64:] # Slice(v3, const(i64[1] : {64L}), const(i64[1] : {9223372036854775807}), 3, 1) # f32[1,64,384,64]
    v19 = -(v18) # f32[1,64,384,64]
    v20 = v16[:,:,:,0:64] # Slice(v3, 0, 64, 3, 1) # f32[1,64,384,64]
    v21 = (v19, v20) # (f32[1,64,384,64], f32[1,64,384,64])

    v22 = np.concatenate(v21, -1) # f32[1,64,384,128]
    # v22.tofile("v22.bin")
    v23 = v22 * v12 # f32[1,64,384,128]
    v24 = v17 + v23 # f32[1,64,384,128]
    v25 = np.transpose(v24, [0,1,3,2]) # f32[1,64,128,384]
    v26 = v14 @ v25 # f32[1,64,384,384]
    # v26.tofile("v26.bin")
    v27 = (v26/ 11.31370) # f32[1,64,384,384]
    v28 = v27 + attn_mask # f32[1,64,384,384]
    # v28.tofile("v28.bin")
    v29 = softmax(v28, -1) # f32[1,64,384,384]
    v30 = np.reshape(v0, [1,1,384, 8192]) # f32[1,1,384,8192]
    v31 = (v30 @ WV) # f32[1,64,384,128]
    v32 = v29 @ v31 # f32[1,64,384,128]
    v33 = np.transpose(v32, [0,2,1,3]) # f32[1,384,64,128]
    v34 = np.reshape(v33, [1,384,8192]) # f32[1,384,8192]
    v35 = v34 @ WO # f32[1,384,8192]
    v36 = (hidden_in + v35) # f32[1,384,8192]
    v37 = rmsnorm(v36, post_ln_gamma, post_ln_beta) # f32[1,384,8192]
    v38 = v37 @ mlp_gate_weights # f32[1,384,22016]
    v39 = swish(v38) # f32[1,384,22016]
    v40 = v37 @ mlp_up_weights # f32[1,384,22016]
    v41 = v39 * v40 # f32[1,384,22016]
    v42 = v41 @ mlp_down_weights # f32[1,384,8192]
    v43 = v36 + v42 # f32[1,384,8192]

    # v43.tofile("v43.bin")
    hidden_in = v43
hidden_in.tofile("hidden_in.bin")