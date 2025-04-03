

import torch
from adtopk_lib import Compressor
import random
import horovod.torch as hvd
import numpy as np

# 
# 20250328
# Layer-wise adaptive sparsification compression scheme
# Online sparsification density adjustment scheme
class LayerWiseAllChannelTopkCompressor(Compressor):
    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.compressed_tensors = {}
        self.attributes = {}
        self.thres_mean_arr=[]

        self.tensor_flatten_np_arr=[]
        self.values_flatten_global_np_arr=[]
        self.values_flatten_channel_np_arr=[]
        self.density_decay = True # 
        self.density_decay_rate = 0.1
        
    def initialize(self, named_parameters):
        tensors_info = []
        sum_numel = 0
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            sum_numel += shape[1]
            tensors_info.append((shape[1],name))
        tensors_info.sort()  # this may be a slow operation (using CPU)
        target_numel = int(sum_numel * self.compress_ratio)
        for idx,val in enumerate(tensors_info):
            tensor_count = len(tensors_info) - idx
            avg_numel = target_numel // tensor_count
            if val[0] * tensor_count > target_numel:
                self.attributes[val[1]] = {'cp_ratio':avg_numel,'compress':True}
                target_numel -= avg_numel
            else:
                self.attributes[val[1]] = {'cp_ratio':val[0],'compress':False}
                target_numel -= val[0]


    def sparsify(self, tensor, epoch, name):
        numel = tensor.numel()
        shape = tensor.shape
        should_compress = name in self.attributes.keys() and self.attributes[name]['compress']
        if should_compress:
            k = max(1,self.attributes[name]['cp_ratio'])
            if self.density_decay:
                self.attributes[name]['cp_ratio'] = max(1,int(self.attributes[name]['cp_ratio'] * (1 - self.density_decay_rate)))
        elif tensor.dim() > 1 or 'fc' in name:
            should_compress = True
            k = max(1,int(shape[1] * self.compress_ratio))
        if should_compress:
            _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1, sorted=False,)
            values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)
            return values_flatten_1, indices_flatten_1
        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values=tensor
        indices=torch.arange(0,numel).cuda(tensor.device)
        return values, indices

    def desparsify(self, tensors, numel, shape):
        values, indices = tensors
        # if True:
        if values.numel()==numel:
            return values
        else:
            tensor_decompressed = torch.zeros(
                shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(1, indices, values)
            return tensor_decompressed

    def compress(self, tensor, name):
        tensors = self.sparsify(tensor, self.epoch, name)
        ctx = tensor.numel(), tensor.size()
        self.compressed_tensors[name] = (tensors,ctx)
        return tensors, ctx

    def show_sparsify(self, tensor):
        print('----------'+str(self.rank)+'----------')
        print(tensor.shape)
        tensor = tensor.flatten()
        print(tensor)

    def decompress(self, tensors, ctx, name=None):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed =self.desparsify(tensors, numel,shape)

        return tensor_decompressed.view(shape)

    def decompress_add(self, tensors, ctx):
        if ctx==None:
            tensor, = tensors
            return tensor

        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        else:
            tensor_decompressed = torch.zeros(
                shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            size = hvd.size()
            sizes = [tensor_decompressed.shape[0]] * size
            indices_list = indices.split(sizes)
            indices = torch.concatenate(indices_list,axis = 1)
            values_list = values.split(sizes)
            values = torch.concatenate(values_list, axis = 1)
            
            tensor_decompressed = tensor_decompressed.scatter_add(1, indices, values)

        return tensor_decompressed.view(shape)
    
    def get_layer_tensor(self,name):
        if name in self.compressed_tensors:
            return self.decompress(*self.compressed_tensors[name])
        else:
            return None
    def is_density_decay(self):
        return self.density_decay
    def turnoff_density_decay(self):
        self.density_decay = False