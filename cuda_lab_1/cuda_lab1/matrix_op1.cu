// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>


#define N (1 << 22)

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void multiplyCPU(const float *h_v1, const float *h_v2, float *h_out) {
    for (int i = 0; i < N; i++) {
        h_out[i] = h_v1[i] * h_v2[i];
    }
}

void expensiveFunctionCPU(const float *h_v1, const float *h_v2, float *h_out) {
    for (int i = 0; i < N; i++) {
        float a = h_v1[i], b = h_v2[i];
        h_out[i] = (a * b) * (sqrt(a + b) + sqrt(a) + sqrt(b - a) + sqrt(b));
    }
}

__global__ void multiplyGPU(const float *g_v1, const float *g_v2, float *g_out) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    g_out[index] = g_v1[index] * g_v2[index];
}

__global__ void expensiveFunctionGPU(const float *g_v1, const float *g_v2, float *g_out) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    float a = g_v1[index], b = g_v2[index];
    g_out[index] = (a * b) * (sqrt(a + b) + sqrt(a) + sqrt(b - a) + sqrt(b));

}

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
    ///////////////////// LOTS OF INITIALIZATION CODE ////////////////////////
    // pointers to host vectors v1 and v1
    float *h_v1, *h_v2;
    // pointers to host output for multiply and expensive
    float *h_multiply_correct, *h_expensive_correct;
    // pointers to store results from gpu functions on the host
    float *h_multiply_out, *h_expensive_out;

    // pointer for device vectors v1 and v2
    float *d_v1, *d_v2;
    // pointers to device output
    float *d_multiply_out, *d_expensive_out;



    // Number of test iterations to use for timing
    int testIterations = 3;

    // allocate memory for pointers
    h_v1 = (float *)malloc(N * sizeof(float));
    h_v2 = (float *)malloc(N * sizeof(float));
    h_multiply_correct  = (float *)malloc(N * sizeof(float));
    h_expensive_correct = (float *)malloc(N * sizeof(float));
    h_multiply_out  = (float *)malloc(N * sizeof(float));
    h_expensive_out = (float *)malloc(N * sizeof(float));

    // allocate memory for device pointers
    cudaMalloc( (void **) &d_v1, N * sizeof(float));
    cudaMalloc( (void **) &d_v2, N * sizeof(float));
    cudaMalloc( (void **) &d_multiply_out, N *sizeof(float));
    cudaMalloc( (void **) &d_expensive_out, N *sizeof(float));

    // Initialize v1 and v2
    for (int i = 0; i < N; i++) {
        h_v1[i] = 1.0 * (i % 10000);
        h_v2[i] = 2 * h_v1[i];
    }
    float multiply_time = 0;
    float expensive_time = 0;
    float time = 0;

    // CPU functions over testIterations
    for (int i = 0; i < testIterations; i++) {
        // zero memory for outputs
        memset(h_multiply_correct,  0, N * sizeof(float));
        memset(h_expensive_correct,  0, N * sizeof(float));
        time = clock();
		// run cpu kernel
        multiplyCPU(h_v1, h_v2, h_multiply_correct);
        multiply_time += (clock()-time)/CLOCKS_PER_SEC*1000;
        // run cpu kernel
        time = clock();
        expensiveFunctionCPU(h_v1, h_v2, h_expensive_correct);
        expensive_time += (clock()-time)/CLOCKS_PER_SEC*1000;
    }
    printf("Multiply serial run time: %fms\n", multiply_time / testIterations);
    printf("Expensive serial run time: %fms\n", expensive_time / testIterations);

    //////////////////////// INSERT CODE IN THIS SECTION /////////////////////
    // GPU functions over testIterations
    multiply_time = 0;
    expensive_time = 0;
    for (int i = 0; i < testIterations; i++) {
        // zero output memory
        memset(h_multiply_out, 0, N * sizeof(float));
        memset(h_expensive_out, 0, N * sizeof(float));
        cudaMemset(d_multiply_out, 0, N * sizeof(float));
        cudaMemset(d_expensive_out, 0, N * sizeof(float));
        // zero input memory
        cudaMemset(d_v1, 0, N * sizeof(float));
        cudaMemset(d_v2, 0, N * sizeof(float));

        // transfer data to GPU
        cudaMemcpy(d_v1, h_v1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, h_v2, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t multiply_begin, multiply_end, expensive_begin, expensive_end;
        cudaEventCreate(&multiply_begin);
        cudaEventCreate(&multiply_end);
        cudaEventCreate(&expensive_begin);
        cudaEventCreate(&expensive_end);

        cudaEventRecord(multiply_begin, 0);
        const size_t block_size = 1024;
        size_t grid_size = N / block_size;
        if(N % block_size) ++grid_size;
        multiplyGPU<<<grid_size, block_size>>>(d_v1, d_v2, d_multiply_out);
        cudaEventRecord(multiply_end, 0);
        cudaEventSynchronize(multiply_end);

	    cudaEventElapsedTime(&time, multiply_begin, multiply_end);
        multiply_time += time;

        // transfer data from GPU
        cudaMemcpy(h_multiply_out, d_multiply_out, N * sizeof(float), cudaMemcpyDeviceToHost);


        // Check for any CUDA errors
        checkCUDAError("multiplyGPU");

        // zero input memory
        cudaMemset(d_v1, 0, N * sizeof(float));
        cudaMemset(d_v2, 0, N * sizeof(float));

        // transfer data to GPU
        cudaMemcpy(d_v1, h_v1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, h_v2, N * sizeof(float), cudaMemcpyHostToDevice);

        const size_t block_size2 = 128;
        size_t grid_size2 = N / block_size2;
        if(N % block_size2) ++grid_size2;

        cudaEventRecord(expensive_begin, 0);
        expensiveFunctionGPU<<<grid_size2, block_size2>>>(d_v1, d_v2, d_expensive_out);
        cudaEventRecord(expensive_end, 0);
        cudaEventSynchronize(expensive_end);

        cudaEventElapsedTime(&time, expensive_begin, expensive_end);
        expensive_time += time;

        // transfer data from GPU
        cudaMemcpy(h_expensive_out, d_expensive_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Check for any CUDA errors
        checkCUDAError("expensiveFunctionGPU");
    }
    printf("Multiply kernel run time: %fms\n", multiply_time/ testIterations);
    printf("Expensive kernel run time: %fms\n", expensive_time/ testIterations);
    /////////////////////////////// VALIDATION ///////////////////////////////

    // check if output from gpu kernels is correct
    for (int i = 0; i < N; i++) {
        if (!(abs(h_multiply_out[i] - h_multiply_correct[i]) <= 0.0001)) {
            printf("Test failed (h_multiply_out[%d]:%f != h_multiply_correct[%d]:%f)!\n",
                   i, h_multiply_out[i], i, h_multiply_correct[i]);
            exit(1);
        }
        if (!(abs(h_expensive_out[i] - h_expensive_correct[i]) <= 0.00001 * abs(h_expensive_correct[i]))) {
            printf("Test failed (h_expensive_out[%d]:%f != h_expensive_correct[%d]:%f)!\n",
                   i, h_expensive_out[i], i, h_expensive_correct[i]);
            exit(1);
        }
    }

    printf("Test passed!\n");

    //////////////////////////////// CLEANUP /////////////////////////////////
    // free host memory
    free(h_v1);
    free(h_v2);
    free(h_multiply_correct);
    free(h_expensive_correct);
    free(h_multiply_out);
    free(h_expensive_out);

    // free device memory
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_multiply_out);
    cudaFree(d_expensive_out);
    return 0;
}
