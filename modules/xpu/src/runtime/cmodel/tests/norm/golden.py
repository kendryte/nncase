import numpy as np
from torch import nn
import torch
from scipy.special import softmax
import os


def rmsnorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps=1e-05):
    x = torch.from_numpy(x)
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    output = output * torch.from_numpy(gamma) + torch.from_numpy(beta)
    return output.detach().numpy()


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
    return nn.functional.layer_norm(torch.from_numpy(x), [x.shape[-1]], torch.from_numpy(gamma), torch.from_numpy(beta)).detach().numpy()


def swish(x: np.ndarray):
    return x / (1 + np.exp(-x))


def gather(data: np.ndarray, axis: int, indices: np.ndarray):
    return torch.gather(torch.from_numpy(data), axis, torch.from_numpy(indices)).detach().numpy()


hidden_in = np.random.rand(1, 384, 8192).astype(np.float32)
hidden_in.tofile('hidden_in.bin')
output = rmsnorm(hidden_in, np.ones([8192], np.float32), np.zeros([8192], np.float32))
# f32[1,384,8192]
output.tofile("output.bin")
