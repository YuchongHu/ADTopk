# ADTopk: All-Dimension Top-k Compression for High-Performance Distributed Training Systems
__ADTopk__  is an all-dimension Top-k sparsification scheme, which selects the largest 𝑘 elements from all dimensions of the gradient in each layer, meaning that each dimension must provide some elements, so as to avoid the dimension missing. Further, __ADTopk__ enables each dimension to perform sorting locally within the elements of the dimension, and thus all dimensions can perform multiple local sortings independently and parallelly, instead of a single global sorting for the entire gradient in each layer. 

On top of __ADTopk__, we further propose an interleaving compression scheme and an efficient threshold estimation algorithm so as to enhance the performance of __ADTopk__. We build a sparsification compression data-parallel DNN training framework and implement a compression library containing state-of-the-art sparsification algorithms.


# Introduction

This code repository covers:

### ADTopk

- __ADTopk__ with a quantity-based inter-worker buffering method to control the consistency of the gradient type and number buffered by inter-worker during each gradient merging.
- An interleaving compression scheme that accelerates model convergence and an efficient threshold estimation algorithm that reduces the compression overhead.

### State-of-the-art gradient sparsification compression methods.

- [DGC](https://arxiv.org/pdf/1712.01887.pdf)
- [Gaussiank](https://arxiv.org/pdf/1911.08772.pdf)
- [Redsync](https://www.sciencedirect.com/science/article/pii/S0743731518308657)
- [OkTopk](https://dl.acm.org/doi/abs/10.1145/3503221.3508399)
- [SIDCo](https://proceedings.mlsys.org/paper_files/paper/2021/file/fea47a8aa372e42f3c84327aec9506cf-Paper.pdf)

# Implementation

We use the PyTorch framework and implemented the prototype system of __ADTopk__ based on the [Horovod](https://github.com/horovod/horovod) framework using NCCL as the communication library. Overview of our system is as follows.

<!-- ![Overview](Overview.jpg) -->
<center class ='img'>
<img src="overview_1.jpg" width="700px" />
</center>


We implement __ADTopk__, which mainly consists of four main modules, an interleaving compression module (i.e., __ADTopk__-i), a threshold estimation sparsification module (i.e., __ADTopk__-t), a communication and aggregation module, a residual gradient error feedback module, and a scalable heterogeneous training module (i.e., __ADTopk__-s).
We also implement an experimental proof module to prove the stable convergence of __ADTopk__ distributed SGD on multiple training tasks.

# Installation
### Prerequisites
- CUDA-11.6
- NCCL-2.8.3
- PyTorch-1.3.+
- [OpenMPI-4.0.+](https://www-lb.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.27.+](https://github.com/horovod/horovod)

### Install ADTopk
```
git clone https://github.com/User/ADTopk.git
cd ADTopk
pip install -r requirements.txt
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.21.3
pip install -e .
```

# Quick start
To run CV jobs:
```
cd ./examples/cv_examples/
bash run.sh
```
To run NLP jobs:
```
cd ./examples/nlp_examples/bert/pytorch/scripts
bash run_squad.sh
```

# Papers
An 11-page conference version of this paper appeared in the Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing (HPDC2024), June 2024.
In this version, we add theoretical proof of __ADTopk__ convergence and scale __ADTopk__ to heterogeneous cluster training. In the experimental part, we rerun more experiments for analysis and discussion.


- ADTopk: All-Dimension Top-k Compression for High-Performance Data-Parallel DNN Training

If you are using this repository for your paper, please cite our work
```
@inproceedings{ming2024adtopk,
  title={ADTopk: All-Dimension Top-k Compression for High-Performance Data-Parallel DNN Training},
  author={Zhangqiang, Ming and Yuchong, Hu and Wenxiang, Zhou and Xinjue, Zheng and Chenxuan, Yao and Dan, Feng},
  booktitle={Proceedings of the 33nd International Symposium on High-Performance Parallel and Distributed Computing},
  url={https://doi.org/10.1145/3625549.3658678}
  year={2024}
}
```

# Referred Datasets
- CIFAR-100: [https://www.cs.utoronto.ca/~kriz/cifar.html](https://www.cs.utoronto.ca/~kriz/cifar.html)
- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)
- Wikitex-2/103: [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)
- SQuAD: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)


# License
See [LICENSE](https://github.com/zqming-cs/ADTopk/blob/main/LICENSE).





