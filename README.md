# GPU-Tensor-Permute
This repo permute the sequence data described by `cudnnSeqDataDescriptor_t` in cuDNN v8 for RNN and Transformer applications. 

In cuDNN v8, Sequence Data is 4D tensors that describe its data dimensions with 4 dimensions `CUDNN_SEQDATA_BATCH_DIM`, `CUDNN_SEQDATA_BEAM_DIM`, `CUDNN_SEQDATA_TIME_DIM` and `CUDNN_SEQDATA_VECT_DIM`. For its permutation, only the first three can be permutated while the `CUDNN_SEQDATA_VECT_DIM` is fixed to be the last dimension. In other words, the `CUDNN_SEQDATA_VECT_DIM` is always contiguous in the memory. 

Thus the permutation is easily transformed to the reordering of the first three dimensions. As no transposition is involved, the permutation is implemented by a warp-level memory copy when assign each `CUDNN_SEQDATA_VECT_DIM` to a warp. The following figure shows the achieved bandwidth of the permutation compared with cudaMemcpy deviceTodevice.

## Parameter
### dimension
4D array with i-th entry indicating its i-th dimension size. 
### permuation 
* mode 0: the i-th entry indicates the new location of the original i-th dimension.
* mode 1: the i-th entry indicates the original location of the transformed i-th dimension.
## Permuation Bandwidth compared with cudaMemcpy deviceTodevice
![Permuation Bandwidth compared with cudaMemcpy deviceTodevice](./bandwidth%20permute.png)
