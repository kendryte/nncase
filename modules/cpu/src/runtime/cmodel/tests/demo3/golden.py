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
v0_gamma [8192] [8192] 
v0_beta [8192] [8192] 
v37_gamma [8192] [8192] 
v37_beta [8192] [8192] 
v2_w [64, 8192, 128] [8@b, 2048@t, 128] 
v3_data [384, 128] [384, 128] 
v11_data [384, 128] [384, 128] 
v16_w [64, 8192, 128] [8@b, 2048@t, 128] 
v31_w [64, 8192, 128] [8@b, 2048@t, 128] 
v35_w [8192, 8192] [1024@b, 2048@t] 
v38_w [8192, 22016] [1024@b, 5504@t] 
v40_w [8192, 22016] [1024@b, 5504@t] 
v42_w [22016, 8192] [2752@b, 2048@t] 
v0 = nn.functional.layer_norm(var_5, [8192], self.v0_gamma, self.v0_beta) # v0 [1, 384, 8192] [1, 48@b, 8192] 
v1 = flow.unsqueeze(v0, 1)  # v1 [1, 1, 384, 8192] [1, 1, 48@b, 8192] 
v2 = flow.matmul(v1, self.v2_w)  # v2 [1, 64, 384, 128] [1, 8@b, 384, 128] 
v3 = self.v3_data[var_6]  # Gather(const(f32[384,128]), const(i64 : 0), )# v3 [1, 384, 128] [1, 384, 128] 
v4 = flow.reshape(v3, [1, 1, 384, 128])  # v4 [1, 1, 384, 128] [1, 1, 384, 128] 
v5 = flow.mul(v2, v4)  # v5 [1, 64, 384, 128] [1, 8@b, 384, 128] 
v6 = v2[:, :, :, 64:]  # v6 [1, 64, 384, 64] [1, 8@b, 384, 64] 
v7 = -v6  # v7 [1, 64, 384, 64] [1, 8@b, 384, 16@t] 
v8 = v2[:, :, :, :64]  # v8 [1, 64, 384, 64] [1, 8@b, 384, 64] 
v10 = flow.concat(v9, -1)  # v10 [1, 64, 384, 128] [1, 8@b, 96@t, 128] 
v11 = self.v11_data[var_6] # v11 [1, 384, 128] [1, 384, 128] 
v12 = flow.reshape(v11, [1, 1, 384, 128])  # v12 [1, 1, 384, 128] [1, 1, 384, 128] 
v13 = flow.mul(v10, v12)  # v13 [1, 64, 384, 128] [1, 8@b, 96@t, 128] 
v14 = flow.add(v5, v13)  # v14 [1, 64, 384, 128] [1, 8@b, 96@t, 128] 
v15 = flow.unsqueeze(v0, 1)  # v15 [1, 1, 384, 8192] [1, 1, 48@b, 8192] 
v16 = flow.matmul(v15, self.v16_w)  # v16 [1, 64, 384, 128] [1, 8@b, 384, 128] 
v17 = flow.mul(v16, v4)  # v17 [1, 64, 384, 128] [1, 8@b, 384, 128] 
v18 = v16[:, :, :, 64:] # v18 [1, 64, 384, 64] [1, 8@b, 384, 64] 
v19 = flow.neg(v18)  # v19 [1, 64, 384, 64] [1, 8@b, 384, 16@t] 
v20 = v16[:, :, :, 0:64] # v20 [1, 64, 384, 64] [1, 8@b, 384, 64] 
v22 = flow.concat(v21, -1)  # v22 [1, 64, 384, 128] [1, 8@b, 96@t, 128] 
v23 = flow.mul(v22, v12)  # v23 [1, 64, 384, 128] [1, 8@b, 96@t, 128] 
v24 = flow.add(v17, v23)  # v24 [1, 64, 384, 128] [1, 8@b, 96@t, 128] 
v25 = flow.transpose(v24, [0, 1, 3, 2])  # v25 [1, 64, 128, 384] [1, 8@b, 128, 96@t] 
v26 = flow.matmul(v14, v25)  # v26 [1, 64, 384, 384] [1, 8@b, 384, 384] 
v27 = flow.div(v26, 11.313708)  # v27 [1, 64, 384, 384] [1, 8@b, 384, 384] 
v28 = flow.add(v27, var_7)  # v28 [1, 64, 384, 384] [1, 8@b, 384, 384] 
v29 = flow.softmax(v28, -1)  # v29 [1, 64, 384, 384] [1, 8@b, 96@t, 384] 
v30 = flow.unsqueeze(v0, 1)  # v30 [1, 1, 384, 8192] [1, 1, 48@b, 8192] 
v31 = flow.matmul(v30, self.v31_w)  # v31 [1, 64, 384, 128] [1, 8@b, 384, 128] 
v32 = flow.matmul(v29, v31)  # v32 [1, 64, 384, 128] [1, 8@b, 384, 128] 
v33 = flow.transpose(v32, [0, 2, 1, 3])  # v33 [1, 384, 64, 128] [1, 384, 8@b, 128] 
v34 = flow.reshape(v33, [1, 384, 8192]) # v34 [1, 384, 8192] [1, 384, 1024@b] 
v35 = flow.matmul(v34, self.v35_w) # v35 [1, 384, 8192] [1, 384, 8192] 
v36 = flow.add(var_5, v35) # v36 [1, 384, 8192] [1, 12@b@t, 8192] 
v37 = nn.functional.layer_norm(v36, [8192], self.v37_gamma, self.v37_beta) # v37 [1, 384, 8192] [1, 12@b@t, 8192] 
v38 = flow.matmul(v37, self.v38_w) # v38 [1, 384, 22016] [1, 384, 5504@t] 
v39 = nn.functional.silu(v38) # v39 [1, 384, 22016] [1, 48@b, 5504@t] 
v40 = flow.matmul(v37, self.v40_w) # v40 [1, 384, 22016] [1, 384, 5504@t] 
v41 = flow.mul(v39, v40) # v41 [1, 384, 22016] [1, 384, 5504@t] 
v42 = flow.matmul(v41, self.v42_w) # v42 [1, 384, 8192] [1, 384, 8192] 
v43 = flow.add(v36, v42) # v43 [1, 384, 8192] [1, 12@b@t, 8192] 
