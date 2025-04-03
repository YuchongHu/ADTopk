import torch
import random
from abc import ABC, abstractmethod


class Memory(ABC):

    def __init__(self):
        self.beta = 1.0
        self.gamma = 1.0
    @abstractmethod
    def compensate(self, tensor, name):
        pass
    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        pass

class NoneMemory(Memory):

    def __init__(self):         
        self.beta = 1.0         
        self.gamma = 1.0
        
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

class ResidualMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.residuals[name] = residual
