import numpy as np
from torch import nn
import torch
    
X = np.random.rand(384, 8192).astype(np.float32)
WK = np.random.rand(64, 8192, 128).astype(np.float32) * 5 - 1

K = X @ WK

Sum =np.sum(np.sum(WK, axis=0, keepdims=False), axis=0,keepdims=False)

X.tofile("X.bin")
WK.tofile("WK.bin")
K.tofile("K.bin")
Sum.tofile("Sum.bin")

# Activate module
Norm = nn.LayerNorm([8192], elementwise_affine=False)(torch.from_numpy( X))
Norm.detach().numpy().tofile('Norm.bin')

RSums =[]
for t in range(32):
      RSums.append(X[:,t*256:(t+1)*256].sum(1,keepdims=False))
RSum =  np.sum(np.array(RSums),axis=0,keepdims=False)
      
RSum.tofile('RSum.bin')

RSumSqrs =[]
for t in range(32):
      RSumSqrs.append(np.square(X[:,t*256:(t+1)*256]).sum(1,keepdims=False))
RSumSqr =  np.sum(np.array(RSumSqrs),axis=0,keepdims=False)
      
RSumSqr.tofile('RSumSqr.bin')