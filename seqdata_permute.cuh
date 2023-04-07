#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common/fmt.hpp"
#include "common/utils.hpp"
#include "common/clara.hpp"
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <array>
#include <random>

void checkCudaError(cudaError_t code, const char *expr, const char *file, int line) {
  if (code) {
      fprintf(stderr, "ERROR: CUDA error at %s:%d, code=%d (%s) in '%s'\n\n",
              file, line, (int)code, cudaGetErrorString(code), expr);
      exit(1);
  }
}

#define CHECK_CUDA_ERR(...)                                             \
    do {                                                                \
        checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  \
    } while (0)

/* this kernel permutes
*/
template <typename T>
__global__ void TransformSeqDataAxesKernel(const int dim[4], const int permute[4], const T* src, T* dst){

}


__device__ __inline__ void flat_perm_dim(int* stride, const int perm_dim[4], int perm){
  *stride = 1;
  for(int i=perm+1; i<3; i++)
    *stride *= perm_dim[i];
}

template <>
__global__ void TransformSeqDataAxesKernel<float>(const int dim[4], const int permute[4], const float* src, float* dst){
  int vec_id = blockIdx.x*(blockDim.x/32) + threadIdx.x / 32;
  int src_offset = dim[3] / 4 * vec_id;

  int idx_dim0 = vec_id / (dim[1]*dim[2]);
  int idx_dim1 = vec_id % (dim[1]*dim[2]) / (dim[2]);
  int idx_dim2 = vec_id % (dim[1]*dim[2]) % (dim[2]);

  int perm_dim[4] = {dim[permute[0]],dim[permute[1]],dim[permute[2]],dim[permute[3]]};
  int perm_stride_0, perm_stride_1, perm_stride_2;

  flat_perm_dim(&perm_stride_0, perm_dim, permute[0]);
  flat_perm_dim(&perm_stride_1, perm_dim, permute[1]);
  flat_perm_dim(&perm_stride_2, perm_dim, permute[2]);

  // if( blockIdx.x==0&&threadIdx.x==0)
  //   printf("perm_stride_0=%d, perm_stride_1=%d, perm_stride_2=%d", perm_stride_0, perm_stride_1, perm_stride_2);
  int trg_offset = dim[3] / 4 * (idx_dim0 * perm_stride_0 + idx_dim1 * perm_stride_1 + idx_dim2 * perm_stride_2) ;

  const float4 *input4 = reinterpret_cast<const float4 *>(src);
  float4 *res4 = reinterpret_cast<float4 *>(dst);
  float4 vinput4;
  int vec_size4 = dim[3] / 4;
  for (int i = threadIdx.x % 32; i < vec_size4; i += 32) {
    vinput4 = input4[src_offset + i];
    res4[trg_offset + i] = vinput4;
  }
}

template <>
__global__ void TransformSeqDataAxesKernel<__half>(const int dim[4], const int permute[4], const __half* src, __half* dst){
  int vec_id = blockIdx.x*(blockDim.x/32) + threadIdx.x / 32;
  int src_offset = dim[3] / 8 * vec_id;

  int idx_dim0 = vec_id / (dim[1]*dim[2]);
  int idx_dim1 = vec_id % (dim[1]*dim[2]) / (dim[2]);
  int idx_dim2 = vec_id % (dim[1]*dim[2]) % (dim[2]);

  int perm_dim[4] = {dim[permute[0]],dim[permute[1]],dim[permute[2]],dim[permute[3]]};
  int perm_stride_0, perm_stride_1, perm_stride_2;

  flat_perm_dim(&perm_stride_0, perm_dim, permute[0]);
  flat_perm_dim(&perm_stride_1, perm_dim, permute[1]);
  flat_perm_dim(&perm_stride_2, perm_dim, permute[2]);

  // if( blockIdx.x==0&&threadIdx.x==0)
  //   printf("perm_stride_0=%d, perm_stride_1=%d, perm_stride_2=%d", perm_stride_0, perm_stride_1, perm_stride_2);
  int trg_offset = dim[3] / 8 * (idx_dim0 * perm_stride_0 + idx_dim1 * perm_stride_1 + idx_dim2 * perm_stride_2) ;

  const float4 *input4 = reinterpret_cast<const float4 *>(src);
  float4 *res4 = reinterpret_cast<float4 *>(dst);
  float4 vinput4;
  int vec_size4 = dim[3] / 8;
  for (int i = threadIdx.x % 32; i < vec_size4; i += 32) {
    vinput4 = input4[src_offset + i];
    res4[trg_offset + i] = vinput4;
  }
}


template <typename T>
void LaunchTransformSeqDataAxesKernel(cudaStream_t stream, const int dim[4], const int permute[4], const T* src, T* dst){
  
  unsigned dim012 = dim[0] * dim[1] * dim[2];

  unsigned warps_per_block = 2;
  dim3 blocks((dim012-1) / (warps_per_block) + 1);
  dim3 threads(warps_per_block*32);

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERR(cudaEventCreate(&start));
  CHECK_CUDA_ERR(cudaEventCreate(&stop));

  // malloc device space of dim and permute
  int *dimA, *permuteA;
  CHECK_CUDA_ERR(cudaMalloc((void**)&dimA, 4*sizeof(int)));
  CHECK_CUDA_ERR(cudaMalloc((void**)&permuteA, 4*sizeof(int)));
  CHECK_CUDA_ERR(cudaMemcpy(dimA, dim, 4*sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERR(cudaMemcpy(permuteA, permute, 4*sizeof(int), cudaMemcpyHostToDevice));

  // // Warmup
  // TransformSeqDataAxesKernel<T><<<blocks, threads,0,stream>>>(dimA, permuteA, src, dst);
  // CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

  // Record the start event
  CHECK_CUDA_ERR(cudaEventRecord(start, stream));
  // if (std::is_same<T, float>::value) {
  //   TransformSeqDataAxesKernel<float><<<blocks, threads,0,stream>>>(dimA, permuteA, src, dst);
  // } else if (std::is_same<T, __half>::value) {
  //   TransformSeqDataAxesKernel<__half><<<blocks, threads,0,stream>>>(dimA, permuteA, src, dst);
  // } else {
  //   printf("Not support this type");
  // }
  TransformSeqDataAxesKernel<T><<<blocks, threads,0,stream>>>(dimA, permuteA, src, dst);
  CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
  // Record the stop event
  CHECK_CUDA_ERR(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  CHECK_CUDA_ERR(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  CHECK_CUDA_ERR(cudaEventElapsedTime(&msecTotal, start, stop));

  CHECK_CUDA_ERR(cudaEventDestroy(start));
  CHECK_CUDA_ERR(cudaEventDestroy(stop));

  // free device space of dim and permute
  CHECK_CUDA_ERR(cudaFree(dimA));
  CHECK_CUDA_ERR(cudaFree(permuteA));

  // Compute and print the performance
  int repeat_number = 1;
  int elem_num = dim[0]*dim[1]*dim[2]*dim[3];
  double   msec  = static_cast<double>(msecTotal) / static_cast<double>(repeat_number);
  double   flops = 0.0 * static_cast<double>(elem_num);
  double   gigaFlops         = (flops * 1.0e-9f) / (msec / 1000.0f);
  double   bandWidth         = static_cast<double>(elem_num)*sizeof(T)*2 / (msec * 1000.f * 1000); // *2 means read and write
  std::string flop_str = std::string(ANSI_COLOR_MAGENTA)+std::string("%.2f GFlop/s, %.3f ms, %.2f GB/s(Memory)\n")+ std::string(ANSI_COLOR_RESET);
  printf(flop_str.c_str(), gigaFlops, msec, bandWidth);
}