// This example demonstrates parallel floating point vector
// addition with a simple __global__ function.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "deviceQuery.h"

#define BLOCK_SIZE 512

// this kernel computes the vector sum c = a + b
// each thread performs one pair-wise addition
void vector_reduction_seq(const float *a,
                          float *c,
                          const size_t n){
    c[0] = 0;
	for(int i = 0; i < n; i++){
		c[0] += a[i];
	}
}

__device__ void warp_reduce(volatile float* sD, int tid) { //unroll last warp (32 threads)
    sD[tid] += sD[tid + 32];
    sD[tid] += sD[tid + 16];
    sD[tid] += sD[tid + 8];
    sD[tid] += sD[tid + 4];
    sD[tid] += sD[tid + 2];
    sD[tid] += sD[tid + 1];
}

__global__ void vector_reduction(float *a,
                                 float *c,
                                 const size_t n){
  // compute the global element index this thread should process
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) { //binary reduction
      if (tid < s) {
          a[i] += a[i + s];
      }
      __syncthreads();
  }

  if (tid == 0) atomicAdd(c, a[i]);
}

__global__ void vector_reduction_shared(const float* a, float* c, const size_t n) {
    extern __shared__ float sD[];
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    sD[tid] = a[i] + a[i+blockSize]; //add on first load
    __syncthreads();
    for(unsigned int s=blockSize/2; s > 32; s >>= 1) { //binary reduction
        if (tid < s) {
            sD[tid] += sD[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warp_reduce(sD, tid); //unroll last warp for block

    if (tid == 0) atomicAdd(c,sD[0]); //add each block value to final value
}

int main(void){
    QueryDevice();
    std::cout << std::endl;
    for (int o = 0; o < 5; ++o) {
        const int num_elements = 1<<(20 + o);

        std::cout << "Problem size: 2^" << (20 + o) << " elements" << std::endl;

        // compute the size of the arrays in bytes
        const int num_bytes = num_elements * sizeof(float);

        // points to host & device arrays
        float *device_array_a = 0;
        float *device_c = 0;
        float *host_array_a   = 0;
        float *host_c   = 0;

        // malloc the host arrays
        host_array_a = (float*)malloc(num_bytes);
        host_c = (float*)malloc(sizeof(float));

        // cudaMalloc the device arrays
        cudaMalloc((void**)&device_array_a, num_bytes);
        cudaMalloc((void**)&device_c, sizeof(float));

        // if any memory allocation failed, report an error message
        if(host_array_a == 0  || host_c == 0 ||
         device_array_a == 0 || device_c == 0){
        printf("couldn't allocate memory\n");
        return 1;
        }

        // initialize host_array_a & host_array_b
        for(int i = 0; i < num_elements; ++i){
        // make array a a linear ramp
        host_array_a[i] = 1;
        }

        const size_t num_launches = 10;
        double average_seq_time;
        struct timespec start, end;
        if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
          perror( "clock gettime" );
          exit( EXIT_FAILURE );
        }

        for(int i = 0; i < num_launches; i++){
        vector_reduction_seq(host_array_a, host_c, num_elements);
        }

        if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
          perror( "clock gettime" );
          exit( EXIT_FAILURE );
        }
        float serialAns = host_c[0];
        //compute the time in s
        average_seq_time = ( end.tv_sec - start.tv_sec )
        	+ (double)( end.tv_nsec - start.tv_nsec ) / 1e+9;
        //take the average
        average_seq_time /= num_launches;
        std::cout << "Average sequential time: " << average_seq_time << "s" << std::endl;
        // compute c = a + b on the device
        const size_t block_size = BLOCK_SIZE;
        size_t grid_size = num_elements / block_size;

        // deal with a possible partial final block
        if(num_elements % block_size) ++grid_size;
        // time the kernel launches using CUDA events
        cudaEvent_t launch_begin, launch_end;
        cudaEventCreate(&launch_begin);
        cudaEventCreate(&launch_end);

        float average_time_simple = 0.0;
        for(int i = 0; i < num_launches; ++i){
        	  // record a CUDA event immediately before and after the kernel launch
          cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
          host_c[0] = 0;
          cudaMemcpy(device_c, host_c, sizeof(float), cudaMemcpyHostToDevice);
          cudaEventRecord(launch_begin,0);
          // launch the kernel
          vector_reduction<<<grid_size, block_size>>>(device_array_a, device_c, num_elements);
          cudaEventRecord(launch_end,0);
          cudaEventSynchronize(launch_end);
          float time = 0.0;
          // measure the time (ms) spent in the kernel
          cudaEventElapsedTime(&time, launch_begin, launch_end);
          average_time_simple += time;
          cudaMemcpy(host_c, device_c, sizeof(float), cudaMemcpyDeviceToHost);
          if (serialAns != host_c[0]) return 0;
        }
        average_time_simple /= num_launches;
        std::cout << "Average global time: " << average_time_simple << "ms" << std::endl;
        float average_time_shared = 0.0;
        for(int i = 0; i < num_launches; ++i){
          // record a CUDA event immediately before and after the kernel launch
          cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
          host_c[0] = 0;
          cudaMemcpy(device_c, host_c, sizeof(float), cudaMemcpyHostToDevice);
          cudaEventRecord(launch_begin,0);
          // launch the kernel
          vector_reduction_shared<<<grid_size, block_size/2, (block_size/2)*sizeof(float)>>>(device_array_a, device_c, num_elements);
          cudaEventRecord(launch_end,0);
          cudaEventSynchronize(launch_end);
          float time = 0.0;
          // measure the time (ms) spent in the kernel
          cudaEventElapsedTime(&time, launch_begin, launch_end);
          average_time_shared += time;
          cudaMemcpy(host_c, device_c, sizeof(float), cudaMemcpyDeviceToHost);
          if (serialAns != host_c[0]) return 0;
        }
        average_time_shared /= num_launches;
        std::cout << "Average shared time: " << average_time_shared << "ms" << std::endl;
        float num_ops=num_elements;
        float seq_throughput = num_ops / (average_seq_time) / 1000000000.0f;
        float simple_throughput = num_ops / (average_time_simple / 1000.0f) / 1000000000.0f;
        float shared_throughput = num_ops / (average_time_shared / 1000.0f) / 1000000000.0f;
        std::cout << "Throughput of sequential: " << seq_throughput << " GB/s" << std::endl;
        std::cout << "Throughput of global kernel: " << simple_throughput << " GB/s" << std::endl;
        std::cout << "Throughput of shared kernel: " << shared_throughput << " GB/s" << std::endl;
        std::cout << "Global kernel performance speed-up over sequential: " << simple_throughput / seq_throughput << "x" << std::endl;
        std::cout << "Shared kernel performance speed-up over sequential: " << shared_throughput / seq_throughput << "x" << std::endl;
        std::cout << "Shared kernel performance speed-up over global kernel: " << shared_throughput / simple_throughput << "x" << std::endl;
        std::cout << std::endl;
        cudaEventDestroy(launch_begin);
        cudaEventDestroy(launch_end);

        // deallocate memory
        free(host_array_a);
        free(host_c);

        cudaFree(device_array_a);
        cudaFree(device_c);
    }
    return 0;
}
