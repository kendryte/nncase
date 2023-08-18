import numpy as np
from torch import nn
import torch
from scipy.special import softmax


X = np.random.rand(384, 8192).astype(np.float32)
WQ = np.random.rand(64, 8192, 128).astype(np.float32) * 5 - 1
WK = np.random.rand(64, 8192, 128).astype(np.float32) * 5 - 1
WV = np.random.rand(64, 8192, 128).astype(np.float32) * 5 - 1
WM = np.random.rand(8192, 8192).astype(np.float32) * 5 - 1

X.tofile('X.bin')
WQ.tofile('WQ.bin')
WK.tofile('WK.bin')
WV.tofile('WV.bin')
WM.tofile('WM.bin')


Q = X @ WQ  # [64,384,128]
K = X @ WK  # [64,384,128]
V = X @ WV  # [64,384,128]

QK = Q @ np.transpose(K, [0, 2, 1])  # [64,384,384]

S = softmax(QK)  # [64,384,384]

Y = np.transpose(S @ V, [1, 0, 2]).reshape([384, -1])  # [64,384,128] ->  [64,8192]

M = Y @ WM  # [384,8192]

Norm = nn.LayerNorm([8192], elementwise_affine=False)(torch.from_numpy(X + M))
Norm.detach().numpy().tofile('Norm.bin')


# a = np.arange( 64 *384* 128).reshape([64, 384, 128])

# b = np.transpose(a,[1,2,0])
# fb = b.flatten()
# fa = a.flatten()

# newa = []
# s0 = 384*128
# s1 = 128
# s2 = 1
# for d1 in range(384):
#   for d2 in range(128):
#     for d0 in range(64):
#       newa.append(fa[d0 * s0 + d1*s1 + d2*s2])

# print(np.allclose(fb,np.array(newa) ))