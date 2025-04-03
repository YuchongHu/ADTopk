import torch
import pprint
from horovod.torch import allgather, allgather_async, synchronize
import horovod.torch as hvd
import numpy as np



class AllgatherEval():
    def __init__(self, compressor, memory, world_size, lr=1.0):
        self.compressor = compressor
        self.memory = memory
        self.world_size = world_size
        self.klist = []
        self.e_klist = []
        self.iteration = 0
        self.lr = 1.0

    def topk(self, tensor, compress_ratio=0.01):
        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        k = max(1, int(numel * compress_ratio))
        # 获取top-k值的索引
        _, indices = torch.topk(tensor.abs(), k, sorted=False,)
        values = torch.gather(tensor, 0, indices)
        tensors = values, indices
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        # 填充稀疏值
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed


    def adtopk(self, tensor, compress_ratio=0.01):
        
        numel = tensor.numel()
        shape =tensor.shape
        if tensor.dim() >1:
            if shape[0]>shape[1]:
                k= max(1, int(shape[0] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=0,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 0, indices_flatten_1)
            else:
                k= max(1, int(shape[1] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)

            values, indices =  values_flatten_1, indices_flatten_1
            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(0, indices_flatten_1, values_flatten_1)
            return tensor_decompressed
        else:
            tensor = tensor.flatten().cuda()
            return tensor


    def calculate_kexi(self, list_tensor_decompressed, list_tensor_noef, list_tensor_ef, name):

        allgathered_1 = self.compressor.aggregate(list_tensor_ef) * self.lr / self.world_size
        allgathered_1 = allgathered_1.flatten()
        allgathered_1 = self.adtopk(allgathered_1, 0.01)

        allgathered_2 = self.compressor.aggregate(list_tensor_decompressed) * self.lr / self.world_size
        allgathered_2 = allgathered_2.flatten()

        allgathered_3 = self.compressor.aggregate(list_tensor_noef) * self.lr / self.world_size

        val1 = (allgathered_1 - allgathered_2).norm()
        val2 = allgathered_3.norm()
        if hvd.rank() == 0:
            val = (val1/val2).item()
            self.klist.append(val)
            # print(val)

    
    def get_kexi(self):
        return self.klist




    def send_step(self, tensor, name):

        if tensor.dim()==1: 
            tensors_compressed,ctx =[tensor], None
            handles = self.async_send(tensors_compressed, name)
            return handles, ctx
        elif 'fc' in name:
            tensors_compressed, ctx = self.compressor.compress(tensor, name)
            handles = self.async_send(tensors_compressed, name)
            return handles, ctx
        else:
            if self.iteration == 1:
                tensor_noef = tensor.flatten()
                tensor = self.memory.compensate(tensor, name)
                
                tensors_compressed3, ctx = self.compressor.compress3(tensor, name, tensor_noef)
                
                val, idx, tensor_noef, tensor_ef = tensors_compressed3
                topked_tensor = val, idx
                self.memory.update(tensor, name, self.compressor, topked_tensor, ctx)
                
                handles = self.async_send(tensors_compressed3, name)
                return handles, ctx
            
            else:
                tensor = self.memory.compensate(tensor, name)
                tensors_compressed, ctx = self.compressor.compress(tensor, name)         
                self.tensor_decompressed=self.memory.update(tensor, name, self.compressor, tensors_compressed, ctx)         
                handles = self.async_send(tensors_compressed, name)         
                return handles, ctx

    def async_send(self, tensors_compressed, name):
        """
        tensors_compressed: val,idx, tensor_noef, tensor_ef
        """

        tensors_size = []
        for t in tensors_compressed:
            size_dim0 = t.size()[0] if len(t.size())>0 else t.numel()
            tensors_size.append(size_dim0)
        
        # 判断是否要求压缩的tensor维度大小相同
        if self.compressor.tensors_size_are_same:
            tensors_size_ag = [tensors_size] * self.world_size  # list of tensor sizes per rank
            tensor_sizes = zip(*tensors_size_ag)  # transpose
        else:
            tensors_size = torch.tensor(tensors_size)  # TODO: set device
            gathered = allgather(tensors_size)  # tensor of tensor sizes per rank
            tensor_sizes = gathered.view([self.world_size, -1]).t().tolist()  # transpose, to list
        
        # 
        handles = []
        for tensor_compressed in tensors_compressed:
            handle = allgather_async(tensor_compressed)
            handles.append(handle)
        return handles, tensor_sizes

    def receive_step(self, handles, ctx,name,tensor):

        tensors_aggregated_avg=self.wait_receive(handles, ctx, name)

        return tensors_aggregated_avg

    def wait_receive(self, result, ctx, name):

        if self.iteration == 1:

            handles, tensor_sizes = result
            tensors_ag = []

            for handle, sizes in zip(handles, tensor_sizes):
                gathered = synchronize(handle)
                tensors_ag.append(gathered.split(sizes))
    
            list_tensor_decompressed = []
            list_tensor_ef = []
            list_tensor_noef = []
            
            # in every iteration, process a rank's tensor 
            for tensor_compressed in zip(*tensors_ag):
                if tensor_compressed[0].dim()>1 and 'fc' not in name:
                    values, indices, tensor_noef, tensor_ef = tensor_compressed

                    list_tensor_noef.append(tensor_noef) 
                    list_tensor_ef.append(tensor_ef)
                    
                    topked_tensor = values, indices
                    tensor_decompressed = self.compressor.decompress(topked_tensor, ctx)
                    list_tensor_decompressed.append(tensor_decompressed)
                else:
                    tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx)
                    list_tensor_decompressed.append(tensor_decompressed) 

            if hvd.rank() == 0 and self.iteration == 1 and list_tensor_decompressed[0].dim()>1 and 'fc' not in name:
                self.calculate_kexi(list_tensor_decompressed, list_tensor_noef, list_tensor_ef, name)
            
            tensors_aggregated = self.compressor.aggregate(list_tensor_decompressed)

            return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated

        else:

            handles, tensor_sizes = result
            tensors_ag = []

            for handle, sizes in zip(handles, tensor_sizes):
                gathered = synchronize(handle)
                tensors_ag.append(gathered.split(sizes))
    
            list_tensor_decompressed = []
            # in every iteration, process a rank's tensor 
            for tensor_compressed in zip(*tensors_ag):
                tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx)
                list_tensor_decompressed.append(tensor_decompressed)
            
            tensors_aggregated = self.compressor.aggregate(list_tensor_decompressed)
            return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated

    def update_iter(self):
        self.iteration += 1
    
    def clear_iter(self):
        self.iteration = 0
    
    def clear_klist(self):
        self.klist = []
    
    def update_eklist(self):
        if hvd.rank() == 0:
            size = len(self.klist)
            kexi = sum(self.klist) / size
            self.e_klist.append(kexi)

    def print_eklist(self):
        print(self.e_klist)



