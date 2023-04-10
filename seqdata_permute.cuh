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
__global__ void TransformSeqDataAxesKernel(const int stride_1, const int stride_2, const int vec_size, const int perm_stride_0, const int perm_stride_1, const int perm_stride_2, const T* src, T* dst){

}

__device__ __host__ __inline__ void flat_perm_dim(int* stride, const int perm_dim[4], int perm){
  *stride = 1;
  for(int i=perm+1; i<3; i++)
    *stride *= perm_dim[i];
}

template <>
__global__ void TransformSeqDataAxesKernel<float>(const int stride_1, const int stride_2, const int vec_size, const int perm_stride_0, const int perm_stride_1, const int perm_stride_2, const float* src, float* dst){
  int vec_id = blockIdx.x*(blockDim.x/32) + threadIdx.x / 32;
  int vec_size4 = (vec_size >> 2);

  int src_offset = vec_size4 * vec_id;

  int idx_dim0 = vec_id / stride_1;
  int idx_dim1 = vec_id % stride_1 / stride_2;
  int idx_dim2 = vec_id % stride_1 % stride_2;


  int trg_offset = vec_size4 * (idx_dim0 * perm_stride_0 + idx_dim1 * perm_stride_1 + idx_dim2 * perm_stride_2) ;

  const float4 *input4 = reinterpret_cast<const float4 *>(src);
  float4 *output4 = reinterpret_cast<float4 *>(dst);
  float4 vinput4;
  for (int i = threadIdx.x % 32; i < vec_size4; i += 32) {
    vinput4 = input4[src_offset + i];
    output4[trg_offset + i] = vinput4;
  }
}

template <>
__global__ void TransformSeqDataAxesKernel<__half>(const int stride_1, const int stride_2, const int vec_size, const int perm_stride_0, const int perm_stride_1, const int perm_stride_2, const __half* src, __half* dst){
  int vec_id = blockIdx.x*(blockDim.x/32) + threadIdx.x / 32;
  int vec_size4 = (vec_size >> 3);

  int src_offset = vec_size4 * vec_id;

  int idx_dim0 = vec_id / stride_1;
  int idx_dim1 = vec_id % stride_1 / stride_2;
  int idx_dim2 = vec_id % stride_1 % stride_2;

  int trg_offset = vec_size4 * (idx_dim0 * perm_stride_0 + idx_dim1 * perm_stride_1 + idx_dim2 * perm_stride_2) ;

  const float4 *input4 = reinterpret_cast<const float4 *>(src);
  float4 *output4 = reinterpret_cast<float4 *>(dst);
  float4 vinput4;
  for (int i = threadIdx.x % 32; i < vec_size4; i += 32) {
    vinput4 = input4[src_offset + i];
    output4[trg_offset + i] = vinput4;
  }
}


// define a function calculates the bandwidth
template <typename T>
void print_launch_info(float msecTotal, int elem_num, int repeat_number, std::string name){
  double   msec  = static_cast<double>(msecTotal) / static_cast<double>(repeat_number);
  double   flops = 0.0 * static_cast<double>(elem_num);
  double   gigaFlops         = (flops * 1.0e-9f) / (msec / 1000.0f);
  double   bandWidth         = static_cast<double>(elem_num)*sizeof(T) / (msec * 1000.f * 1000); // *2 means read and write
  std::string flop_str = std::string(ANSI_COLOR_MAGENTA)+name+" "+std::string("%.2f GFlop/s, %.3f ms, %.2f GB/s(Memory)\n")+ std::string(ANSI_COLOR_RESET);
  printf(flop_str.c_str(), gigaFlops, msec, bandWidth);
}

template <typename T>
void LaunchTransformSeqDataAxesKernel(cudaStream_t stream, unsigned mode, const int dim[4], const int permute[4], const T* src, T* dst){
  
  unsigned dim012 = dim[0] * dim[1] * dim[2];

  int perm_dim[4] = {};
  int perm_stride_0, perm_stride_1, perm_stride_2;

  if(mode==0){
    for(int i=0; i<4; i++)
      perm_dim[permute[i]] = dim[i];
    flat_perm_dim(&perm_stride_0, perm_dim, permute[0]);
    flat_perm_dim(&perm_stride_1, perm_dim, permute[1]);
    flat_perm_dim(&perm_stride_2, perm_dim, permute[2]);  
  }
  else if(mode==1){
    for(int i=0; i<4; i++)
      perm_dim[i] = dim[permute[i]];
    // find the index of 0, 1, 2 in permute array
    int permute_0, permute_1, permute_2;
    for(int i=0; i<4; i++){
      if(permute[i]==0)
        permute_0 = i;
      else if(permute[i]==1)
        permute_1 = i;
      else if(permute[i]==2)
        permute_2 = i;
    }

    flat_perm_dim(&perm_stride_0, perm_dim, permute_0);
    flat_perm_dim(&perm_stride_1, perm_dim, permute_1);
    flat_perm_dim(&perm_stride_2, perm_dim, permute_2);

  }

  const int stride_1 = (dim[1]*dim[2]);
  const int stride_2 = (dim[2]);
  const int vec_size = (dim[3]);

  std::cout << "stride_1: " << stride_1 << " stride_2: " << stride_2 << " vec_size: " << vec_size << " perm_stride_0: " << perm_stride_0*vec_size << " perm_stride_1: " << perm_stride_1*vec_size << " perm_stride_2: " << perm_stride_2*vec_size << std::endl;

  // Allocate CUDA events that we'll use for timing
  float msecTotal = 0.0f;
  cudaEvent_t start, stop;
  CHECK_CUDA_ERR(cudaEventCreate(&start));
  CHECK_CUDA_ERR(cudaEventCreate(&stop));


  {
    // Record the start event
    CHECK_CUDA_ERR(cudaEventRecord(start, stream));
    CHECK_CUDA_ERR(cudaMemcpyAsync(dst, src, dim012*vec_size*sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // Record the stop event
    CHECK_CUDA_ERR(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    CHECK_CUDA_ERR(cudaEventElapsedTime(&msecTotal, start, stop));
    print_launch_info<T>(msecTotal, dim012*vec_size*2, 1, "cudaMemcpyDeviceToDevice");
  }



  {
    unsigned warps_per_block = 8;
    dim3 blocks((dim012-1) / (warps_per_block) + 1);
    dim3 threads(warps_per_block*32);
    // Record the start event
    CHECK_CUDA_ERR(cudaEventRecord(start, stream));
    TransformSeqDataAxesKernel<T><<<blocks, threads,0,stream>>>(stride_1, stride_2, vec_size, perm_stride_0, perm_stride_1, perm_stride_2, src, dst);
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // Record the stop event
    CHECK_CUDA_ERR(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    CHECK_CUDA_ERR(cudaEventElapsedTime(&msecTotal, start, stop));
    
    // Compute and print the performance
    print_launch_info<T>(msecTotal, dim012*vec_size*2, 1, "TransformSeqDataAxesKernel");
  }

  CHECK_CUDA_ERR(cudaEventDestroy(start));
  CHECK_CUDA_ERR(cudaEventDestroy(stop));


  
}