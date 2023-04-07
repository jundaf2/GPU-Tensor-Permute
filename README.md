# GPU-Tensor-Permute
This repo permute the sequence data described by `cudnnSeqDataDescriptor_t` in cuDNN v8 for RNN and Transformer applications. 

In cuDNN v8, Sequence Data is 4D tensors that describe its data dimensions with 4 dimensions `CUDNN_SEQDATA_BATCH_DIM`, `CUDNN_SEQDATA_BEAM_DIM`, `CUDNN_SEQDATA_TIME_DIM` and `CUDNN_SEQDATA_VECT_DIM`. For its permutation, only the first three can be permutated while the `CUDNN_SEQDATA_VECT_DIM` is fixed to be the last dimension. In other words, the `CUDNN_SEQDATA_VECT_DIM` is always contiguous in the memory. 

