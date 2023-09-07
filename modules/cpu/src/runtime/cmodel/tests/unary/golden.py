import numpy as np
from torch import nn
import torch
from scipy.special import softmax
import os

hidden_in = np.random.rand(1, 384, 2048).astype(np.float32)
hidden_in.tofile('hidden_in.bin')

output = np.arcsin(hidden_in)

output.tofile("output.bin")
