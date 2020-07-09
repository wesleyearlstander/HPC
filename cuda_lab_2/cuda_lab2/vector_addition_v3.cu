// This example demonstrates parallel floating point vector
// addition with a simple __global__ function.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

// this kernel computes the vector sum c = a + b
// each thread performs one pair-wise addition
void vector_add_seq(const float *a,
                    const float *b,
                    float *c,
                    const size_t n){
	for(int i = 0; i < n; i++){
		c[i] = a[i] + b[i];
	}
}

__global__ void vector_add(const float *a,
                           const float *b,
                           float *c,
                           const size_t n){
  // compute the global element index this thread should process
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  // avoid accessing out of bounds elements
  if(i < n){
    // sum elements
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_shared(const float* a, const float* b, float* c, const size_t n) {
    __shared__ float s[512];
    __shared__ float bs[512];
    __shared__ float as[512];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    bs[tid] = b[i];
    as[tid] = a[i];
    if(i < n){
      // sum elements
      s[tid] = as[tid] + bs[tid];
    }
    c[i] = s[tid];
}

int main(void){
  // create arrays of 1M elements
  const int num_elements = 1<<20;

  // compute the size of the arrays in bytes
  const int num_bytes = num_elements * sizeof(int);

  // points to host & device arrays
  float *device_array_a = 0;
  float *device_array_b = 0;
  float *device_array_c = 0;
  float *host_array_a   = 0;
  float *host_array_b   = 0;
  float *host_array_c   = 0;

  // malloc the host arrays
  host_array_a = (float*)malloc(num_bytes);
  host_array_b = (float*)malloc(num_bytes);
  host_array_c = (float*)malloc(num_bytes);

  // cudaMalloc the device arrays
  cudaMalloc((void**)&device_array_a, num_bytes);
  cudaMalloc((void**)&device_array_b, num_bytes);
  cudaMalloc((void**)&device_array_c, num_bytes);

  // if any memory allocation failed, report an error message
  if(host_array_a == 0 || host_array_b == 0 || host_array_c == 0 ||
     device_array_a == 0 || device_array_b == 0 || device_array_c == 0){
    printf("couldn't allocate memory\n");
    return 1;
  }

  // initialize host_array_a & host_array_b
  for(int i = 0; i < num_elements; ++i){
    // make array a a linear ramp
    host_array_a[i] = i;
    // make array b random
    host_array_b[i] = rand()%num_elements;
  }
  // copy arrays a & b to the device memory space
  cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_array_b, host_array_b, num_bytes, cudaMemcpyHostToDevice);
  const size_t num_launches = 10;
  double average_seq_time;
  struct timespec start, end;
  std::cout << "Timing sequential implementation...";
  if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
  }

  for(int i = 0; i < num_launches; i++){
		vector_add_seq(host_array_a, host_array_b, host_array_c, num_elements);
  }

  if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
  }
  //compute the time in s
  average_seq_time = ( end.tv_sec - start.tv_sec )
		+ (double)( end.tv_nsec - start.tv_nsec ) / 1e+9;
  //take the average
  average_seq_time /= num_launches;
  std::cout << " done." << std::endl;
  std::cout << average_seq_time << "s" << std::endl;
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
  std::cout << "Timing simple implementation...";
  for(int i = 0; i < num_launches; ++i){
	  // record a CUDA event immediately before and after the kernel launch
	  cudaEventRecord(launch_begin,0);
	  // launch the kernel
	  vector_add<<<grid_size, block_size>>>(device_array_a, device_array_b, device_array_c, num_elements);
	  cudaEventRecord(launch_end,0);
	  cudaEventSynchronize(launch_end);
	  float time = 0.0;
	  // measure the time (ms) spent in the kernel
	  cudaEventElapsedTime(&time, launch_begin, launch_end);
	  average_time_simple += time;
  }
  // copy the result back to the host memory space
  cudaMemcpy(host_array_c, device_array_c, num_bytes, cudaMemcpyDeviceToHost);
  average_time_simple /= num_launches;
  std::cout << " done." << std::endl;
  std::cout << average_time_simple << "ms" << std::endl;

  float average_time_shared = 0.0;
  std::cout << "Timing shared implementation...";
  for(int i = 0; i < num_launches; ++i){
	  // record a CUDA event immediately before and after the kernel launch
	  cudaEventRecord(launch_begin,0);
	  // launch the kernel
	  vector_add_shared<<<grid_size, block_size>>>(device_array_a, device_array_b, device_array_c, num_elements);
	  cudaEventRecord(launch_end,0);
	  cudaEventSynchronize(launch_end);
	  float time = 0.0;
	  // measure the time (ms) spent in the kernel
	  cudaEventElapsedTime(&time, launch_begin, launch_end);
	  average_time_shared += time;
  }
  // copy the result back to the host memory space
  cudaMemcpy(host_array_c, device_array_c, num_bytes, cudaMemcpyDeviceToHost);
  average_time_shared /= num_launches;
  std::cout << " done." << std::endl;
  std::cout << average_time_shared << "ms" << std::endl;

  float num_ops=num_elements;
  float seq_throughput = num_ops / (average_seq_time) / 1000000000.0f;
  float simple_throughput = num_ops / (average_time_simple / 1000.0f) / 1000000000.0f;
  float shared_throughput = num_ops / (average_time_shared / 1000.0f) / 1000000000.0f;
  std::cout << "Throughput of sequential: " << seq_throughput << " GB/s" << std::endl;
  std::cout << "Throughput of simple kernel: " << simple_throughput << " GB/s" << std::endl;
  std::cout << "Simple performance improvement: " << simple_throughput / seq_throughput << "x" << std::endl;
  std::cout << "Throughput of shared kernel: " << shared_throughput << " GB/s" << std::endl;
  std::cout << "Shared performance improvement: " << shared_throughput / seq_throughput << "x" << std::endl;
  std::cout << "Shared performance over simple improvement: " << shared_throughput / simple_throughput << "x" << std::endl;

  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate memory
  free(host_array_a);
  free(host_array_b);
  free(host_array_c);

  cudaFree(device_array_a);
  cudaFree(device_array_b);
  cudaFree(device_array_c);
}
