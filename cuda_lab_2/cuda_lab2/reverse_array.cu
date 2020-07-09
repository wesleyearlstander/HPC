// includes, system
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// implement the kernel using global memory
__global__ void reverseArray(int *d_out, int *d_in, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = n - i - 1;
    d_out[i] = d_in[ir];
}
// implement the kernel using shared memory
__global__ void reverseArray_shared(int *d_out, int *d_in, int n){
    /*__shared__ int s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = n-i-1;
    s[t] = d_in[i];
    __syncthreads();
    d_out[ti] = s[t];*/
    __shared__ int s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tr = 256 - t - 1;
    s[tr] = d_in[i];
    d_out[(1023-blockIdx.x)*blockDim.x+tr] = s[tr];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv){
    // pointer for host memory and size
    int *h_a;
    int dimA = 256 * 1024; // 256K elements (1MB total)
    // pointer for device memory
    int *d_b, *d_a;
    // define grid and block size
    int numThreadsPerBlock = 256;
    // Part 1: compute number of blocks needed based on array size and desired block size
    int numBlocks = dimA / numThreadsPerBlock;
    // allocate host and device memory
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = (int *) malloc(memSize);
    cudaMalloc( (void **) &d_a, memSize );
    cudaMalloc( (void **) &d_b, memSize );
    // Initialize input array on host
    for (int i = 0; i < dimA; ++i){
        h_a[i] = i;
    }
    // Copy host array to device array
    cudaMemcpy( d_a, h_a, memSize, cudaMemcpyHostToDevice );
    // launch kernel
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

    cudaEvent_t launch_begin, launch_end;
     cudaEventCreate(&launch_begin);
     cudaEventCreate(&launch_end);
     const size_t num_launches = 4;
    float average_time_simple = 0.0;
    std::cout << "Timing simple implementation...";
    for(int i = 0; i < num_launches; ++i){	  // record a CUDA event immediately before and after the kernel launch	  cudaEventRecord(launch_begin,0);	  // launch the kernel
      cudaEventRecord(launch_begin,0);
      reverseArray<<< dimGrid, dimBlock >>>( d_b, d_a, dimA);
        // block until the device has completed
      cudaDeviceSynchronize();
        // check if kernel execution generated an error
        // Check for any CUDA errors
      checkCUDAError("kernel invocation");
      cudaEventRecord(launch_end,0);
      cudaEventSynchronize(launch_end);
      float time = 0.0;
      // measure the time (ms) spent in the kernel	  cudaEventElapsedTime(&time, launch_begin, launch_end);
      cudaEventElapsedTime(&time, launch_begin, launch_end);
      average_time_simple += time;
    }
      // copy the result back to the host memory space
    cudaMemcpy( h_a, d_b, memSize, cudaMemcpyDeviceToHost );
    average_time_simple /= num_launches;
    std::cout << " done." << std::endl;
    std::cout << average_time_simple << "ms" << std::endl;
    // Check for any CUDA errors
    checkCUDAError("memcpy");
    // verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++){
        assert(h_a[i] == dimA - 1 - i );
    }

    float average_time_shared = 0.0;
    std::cout << "Timing shared implementation...";
    for(int i = 0; i < num_launches; ++i){	  // record a CUDA event immediately before and after the kernel launch	  cudaEventRecord(launch_begin,0);	  // launch the kernel
      cudaEventRecord(launch_begin,0);
      reverseArray_shared<<< dimGrid, dimBlock >>>( d_b, d_a, dimA);
        // block until the device has completed
      cudaDeviceSynchronize();
        // check if kernel execution generated an error
        // Check for any CUDA errors
      checkCUDAError("kernel invocation");
      cudaEventRecord(launch_end,0);
      cudaEventSynchronize(launch_end);
      float time = 0.0;
      // measure the time (ms) spent in the kernel	  cudaEventElapsedTime(&time, launch_begin, launch_end);
      cudaEventElapsedTime(&time, launch_begin, launch_end);
      average_time_shared += time;
    }
      // copy the result back to the host memory space
    cudaMemcpy( h_a, d_b, memSize, cudaMemcpyDeviceToHost );
    average_time_shared /= num_launches;
    std::cout << " done." << std::endl;
    std::cout << average_time_shared << "ms" << std::endl;

    // Check for any CUDA errors
    checkCUDAError("memcpy");
    // verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++){
        assert(h_a[i] == dimA - 1 - i );
    }


    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    // free host memory
    free(h_a);
    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.
    printf("Global memory -- verified.\n");
    return 0;
}
void checkCUDAError(const char *msg){
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
