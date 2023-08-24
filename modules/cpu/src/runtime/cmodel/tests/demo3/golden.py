import numpy as np
from torch import nn
import torch
from scipy.special import softmax
def layernorm(x:np.ndarray):
    return nn.LayerNorm([x.shape[-1]], elementwise_affine=False)(torch.from_numpy(x)).detach().numpy()

def swish(x:np.ndarray):
    return x / (1 + np.exp(-x))


hidden_in = np.random.rand(1, 384, 8192).astype(np.float32)
attn_mask = np.random.rand(1, 1, 384, 384).astype(np.float32)
position_ids = np.random.randint(0, 384, [1, 384])

# np.random.rand(8192).astype(np.float32), np.random.rand(8192).astype(np.float32)
v0 = layernorm(hidden_in) # f32[1,384,8192]
v1 = (v0 @ np.random.rand(8192,8192).astype(np.float32)) # f32[1,384,8192]
v2 = np.reshape(v1, [1,384,64,128]) # f32[1,384,64,128]
v3 = np.transpose(v2, [0,2,1,3]) # f32[1,64,384,128]
v4 = np.random.rand(384,128).astype(np.float32)[position_ids] #Gather(np.random.rand(384,128).astype(np.float32), 0, position_ids) # f32[1,384,128]
v5 = np.reshape(v4, [1,1,384,128] ) # f32[1,1,384,128]
v6 = (v3 * v5) # f32[1,64,384,128]
v7 = v3[:,:,:,64:] # Slice(v3, const(i64[1] : {64L}), const(i64[1] : {9223372036854775807}), 3, 1) # f32[1,64,384,64]
v8 = -(v7) # f32[1,64,384,64]
v9 = v3[:,:,:,0:64] # Slice(v3, 0, 64, 3, 1) # f32[1,64,384,64]
v10 = (v8, v9) # (f32[1,64,384,64], f32[1,64,384,64])

v11 = np.concatenate(v10, -1) # f32[1,64,384,128]
v12 = np.random.rand(384,128).astype(np.float32)[position_ids] #Gather(np.random.rand(384,128).astype(np.float32),0, position_ids) # f32[1,384,128]
v13 = np.reshape(v12, [1,1,384,128]) # f32[1,1,384,128]
v14 = (v11 * v13) # f32[1,64,384,128]
v15 = (v6 + v14) # f32[1,64,384,128]
v16 = (v0 @ np.random.rand(8192,8192).astype(np.float32)) # f32[1,384,8192]
v17 = np.reshape(v16, [1,384,64,128]) # f32[1,384,64,128]
v18 = np.transpose(v17, [0,2,1,3]) # f32[1,64,384,128]
v19 = (v18 * v5) # f32[1,64,384,128]
v20 = v18[:,:,:,64:] # Slice(v18, 64, 9223372036854775807, 3, 1) # f32[1,64,384,64]
v21 = -(v20) # f32[1,64,384,64]
v22 = v18[:,:,:,0:64]# Slice(v18, 0, 64, 3, 1) # f32[1,64,384,64]
v23 = (v21, v22) # (f32[1,64,384,64], f32[1,64,384,64])

v24 = np.concatenate(v23, -1) # f32[1,64,384,128]
v25 = (v24 * v13) # f32[1,64,384,128]
v26 = (v19 + v25) # f32[1,64,384,128]
v27 = np.transpose(v26, [0,1,3,2]) # f32[1,64,128,384]
v28 = (v15 @ v27) # f32[1,64,384,384]
v29 = (v28/ 11.31370) # f32[1,64,384,384]
v30 = (v29 + attn_mask) # f32[1,64,384,384]
v31 = softmax(v30, -1) # f32[1,64,384,384]
v32 = (v0 @ np.random.rand(8192,8192).astype(np.float32)) # f32[1,384,8192]
v33 = np.reshape(v32, [1,384,64,128]) # f32[1,384,64,128]
v34 = np.transpose(v33, [0,2,1,3]) # f32[1,64,384,128]
v35 = (v31 @ v34) # f32[1,64,384,128]
v36 = np.transpose(v35, [0,2,1,3]) # f32[1,384,64,128]
v37 = np.reshape(v36, [1,384,8192]) # f32[1,384,8192]
v38 = (v37 @ np.random.rand(8192,8192).astype(np.float32)) # f32[1,384,8192]
v39 = (hidden_in + v38) # f32[1,384,8192]
# np.random.rand(8192).astype(np.float32), np.random.rand(8192).astype(np.float32)
v40 = layernorm(v39) # f32[1,384,8192]
v41 = (v40 @ np.random.rand(8192,22016).astype(np.float32)) # f32[1,384,22016]
v42 = swish(v41) # f32[1,384,22016]
v43 = (v40 @ np.random.rand(8192,22016).astype(np.float32)) # f32[1,384,22016]
v44 = (v42 * v43) # f32[1,384,22016]
v45 = (v44 @ np.random.rand(22016,8192).astype(np.float32)) # f32[1,384,8192]
v46 = (v39 + v45) # f32[1,384,8192]
v46.tofile("")