
import torch
from adtopk_lib import Compressor
from adtopk_lib.heterogeneous.sparsification import LayerWiseAllChannelTopkCompressor


# 20250328
# Partial staleness gradient compensation
def staleness_gradient_compensation(tensor:torch.Tensor,compressor:Compressor,name):
    if isinstance(compressor,LayerWiseAllChannelTopkCompressor):
        local_gradient = compressor.get_layer_tensor(name)
        if(local_gradient == None or torch.norm(local_gradient,p=1) < torch.norm(tensor)):
            return tensor
        else:
            return tensor + local_gradient
    else:

        return tensor

























