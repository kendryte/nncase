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

QK.tofile("QKH.bin")

S = softmax(QK,-1)  # [64,384,384]

S.tofile("Softmax.bin")

Y = np.transpose(S @ V, [1, 0, 2]).reshape([384, -1])  # [64,384,128] ->  [64,8192]

YM = Y @ WM  # [384,8192]

YM.tofile("YM.bin")

Norm = nn.LayerNorm([8192], elementwise_affine=False)(torch.from_numpy(X + YM))
Norm.detach().numpy().tofile('Norm.bin')
