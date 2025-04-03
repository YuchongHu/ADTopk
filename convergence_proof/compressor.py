import torch
import random
from abc import ABC, abstractmethod

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)

def sparsify(tensor, compress_ratio):

    tensor = tensor.flatten().cuda()

    len = tensor.numel()
    k = max(1, int(len * compress_ratio))

    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices

def desparsify(tensors, numel):
    values, indices = tensors
    
    tensor_decompressed = torch.zeros(
        numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio=0.01):
        super().__init__()

        self.compress_ratio = compress_ratio

    def compress3(self, tensor, name, tensor_noef):
        tensors = sparsify(tensor, self.compress_ratio)

        val, idx = tensors
        ctx = tensor.numel(), tensor.size()
        
        # answer: val, idx, tensor_noef, tensor_ef
        tensor_compressed3 = val, idx, tensor_noef.flatten().cuda(), tensor.flatten().cuda()

        return tensor_compressed3, ctx


    def compress(self, tensor, name):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()         
        return tensors, ctx



    def decompress(self, tensors, ctx):

        if ctx == None:             
            tensor, = tensors             
            return tensor         
        else:
            numel, shape = ctx
            tensor_decompressed = desparsify(tensors, numel)
            return tensor_decompressed.view(shape)

    # def decompress_add(self, tensors, ctx):
    #     numel, shape = ctx
    #     values, indices = tensors
    #     if values.numel()==numel:
    #         return values
    #     tensor_decompressed = torch.zeros(
    #         numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
    #     tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
    #     return tensor_decompressed.view(shape)

class NoneCompressor(Compressor):

    def compress(self, tensor, name):
        return [tensor], None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor

    
class AllChannelTopkCompressor(Compressor):

    def __init__(self, compress_ratio=0.01):
        super().__init__()
        self.compress_ratio = compress_ratio

    # tensor稀疏化得到top-k的稀疏值
    def sparsify(self, tensor, compress_ratio):
        
        numel = tensor.numel()
        shape =tensor.shape
        
        # if True:
        if tensor.dim() >1:

            # All-Channel Top-k
            if shape[0]>shape[1]:
                k= max(1, int(shape[0] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=0,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 0, indices_flatten_1)
            else:
                k= max(1, int(shape[1] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)   
                   
            return values_flatten_1, indices_flatten_1

        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values = tensor
        indices = torch.arange(0,numel).cuda(tensor.device)
        return values, indices

    # tensor反稀疏化
    def desparsify(self, tensors, numel, shape):
        values, indices = tensors
        # if True:
        if values.numel()==numel:
            return values
        else:
            # All-Channel Top-k
            tensor_decompressed = torch.zeros(
                shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            if shape[0]>shape[1]:
                tensor_decompressed.scatter_(0, indices, values)
            else:
                tensor_decompressed.scatter_(1, indices, values)
            return tensor_decompressed

    # 抽象方法重载compress
    def compress(self, tensor, name):
        tensors = self.sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):

        if ctx == None:
            tensor, = tensors
            return tensor
        else:
            numel, shape = ctx
            tensor_decompressed =self.desparsify(tensors, numel,shape)
            return tensor_decompressed.view(shape)

    def compress3(self, tensor, name, tensor_noef):
        tensors = self.sparsify(tensor, self.compress_ratio)          
        val, idx = tensors      
        ctx = tensor.numel(), tensor.size()                  
        # answer: val, idx, tensor_noef, tensor_ef         
        tensor_compressed3 = val, idx, tensor_noef.flatten().cuda(), tensor.flatten().cuda()          
        return tensor_compressed3, ctx  