// This example demonstrates the use of shared per-block arrays
// implement an optimized dense matrix multiplication algorithm.


#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "deviceQuery.h"

#define BLOCK_SIZE 8

void matrix_transpose_seq(float *a, float *b, size_t width){
	for (int i =0; i < width; ++i) {
		for (int j = 0; j < width; ++j) {
			b[i*width+j] = a[j*width+i];
		}
	}
}

__global__ void matrix_transpose_simple(const float *a, float *b, size_t width){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	b[row*width+col] = a[col*width+row];
}

__global__ void matrix_transpose_shared (const float *a, float *b, size_t width) {
	__shared__ float s[BLOCK_SIZE][BLOCK_SIZE];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = bx*BLOCK_SIZE + tx;
	int col = by*BLOCK_SIZE + ty;
	int in = row + col * width;

	int rowOut = by*BLOCK_SIZE + tx;
	int colOut = bx*BLOCK_SIZE + ty;
	int out = rowOut + colOut * width;

	s[ty][tx] = a[in];
	__syncthreads();
	b[out] = s[tx][ty];
}

// compare two matrix to see if they are equal -- for verification
int matrixEqual(  float *matrixA, float *matrixB, int m, int n ){
  int bad = 0;
  for ( int y = 0; y < m && !bad ; y++ )
    for ( int x = 0; x < n && !bad ; x++ ){
      if ( abs(matrixA[y*n+x] - matrixB[y*n+x]) > 1e-4 ){
        bad++;
      }
    }
  return !bad;
}

int main(void){

	QueryDevice();
    std::cout << std::endl;
	for (int o = 0; o < 4; ++o) {
		const size_t n = 1<<(9+o);
		dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
		dim3 num_blocks(n / block_size.x, n / block_size.y);
		std::cout << "Matrix size: 2^" << (9+o) << "x" << "2^" << (9+o) << std::endl;
		std::cout << "Block size: " << block_size.x << "x" << block_size.y << std::endl;
		float *h_a, *h_s, *h_res;

		h_a = (float *)malloc(sizeof(float) * n * n);
		h_s = (float *)malloc(sizeof(float) * n * n);
		h_res = (float*)malloc(sizeof(float) * n * n);

		for(int i = 0; i < n*n; ++i){
		h_a[i] = static_cast<float>(rand()) / RAND_MAX;
		}

		float *d_a = 0, *d_c = 0;
		cudaMalloc((void**)&d_a, sizeof(float) * n * n);
		cudaMalloc((void**)&d_c, sizeof(float) * n * n);

		// copy input to the device
		cudaMemcpy(d_a, h_a, sizeof(float) * n * n, cudaMemcpyHostToDevice);

		// time the kernel launches using CUDA events
		cudaEvent_t launch_begin, launch_end;
		cudaEventCreate(&launch_begin);
		cudaEventCreate(&launch_end);
		//time many sequential run and take the average
		size_t num_launches = 10;
		double average_seq_time;
		struct timespec start, end;
		if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
		perror( "clock gettime" );
		exit( EXIT_FAILURE );
		}
		for(int i = 0; i < num_launches; i++){
		  matrix_transpose_seq(h_a, h_s, n);
		}

		if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
		  perror( "clock gettime" );
		  exit( EXIT_FAILURE );
		}
		//compute the time in s
		average_seq_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;
		//take the average
		average_seq_time /= num_launches;
		std::cout << "Average sequential time: " << average_seq_time << "s" << std::endl;
		//launch a single "warm-up" kernel
		matrix_transpose_simple<<<num_blocks,block_size>>>(d_a, d_c, n);
		cudaMemcpy(h_res, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
		int equal = matrixEqual(h_res, h_s, n, n);
		if(!equal) {
			return 0;
		}
		// time many kernel launches and take the average time
		float average_simple_time = 0;
		for(int i = 0; i < num_launches; ++i){
		// record a CUDA event immediately before and after the kernel launch
		cudaEventRecord(launch_begin,0);
		matrix_transpose_simple<<<num_blocks,block_size>>>(d_a, d_c, n);
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
		// measure the time spent in the kernel
		float time = 0;
		cudaEventElapsedTime(&time, launch_begin, launch_end);
		average_simple_time += time;
		}
		average_simple_time /= num_launches;
		std::cout << "Average global kernel time: " << average_simple_time << "ms" << std::endl;
		matrix_transpose_shared<<<num_blocks,block_size>>>(d_a, d_c, n);
		cudaMemcpy(h_res, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
		equal = matrixEqual(h_res, h_s, n, n);
		if(!equal){
			return 0;
		}

		float average_tiled_time = 0;
		for(int i = 0; i < num_launches; ++i){
			cudaEventRecord(launch_begin,0);
			matrix_transpose_shared<<<num_blocks,block_size>>>(d_a, d_c, n);
			cudaEventRecord(launch_end,0);
			cudaEventSynchronize(launch_end);
			float time = 0;
			cudaEventElapsedTime(&time, launch_begin, launch_end);
			average_tiled_time += time;
		}
		average_tiled_time /= num_launches;
		std::cout << "Average shared kernel time: " << average_tiled_time << "ms" << std::endl;


		float mem_size = sizeof(float) * n * n;
		float num_ops = 2 * mem_size;
		float seq_throughput = num_ops / average_seq_time / 1000000000.0f;
		float simple_throughput = num_ops / (average_simple_time / 1000.0f) / 1000000000.0f;
		float tiled_throughput = num_ops / (average_tiled_time / 1000.0f) / 1000000000.0f;

		std::cout << "Throughput of sequential implementation: " << seq_throughput << "GB/s" << std::endl;
		std::cout << "Throughput of global kernel: " << simple_throughput << "GB/s" << std::endl;
		std::cout << "Throughput of shared kernel: " << tiled_throughput << "GB/s" << std::endl;
		std::cout << "Performance speedup: global over sequential " << simple_throughput / seq_throughput << "x" << std::endl;
		std::cout << "Performance speedup: shared over sequential " << tiled_throughput / seq_throughput << "x" << std::endl;
		std::cout << "Performance speedup: shared over global " << tiled_throughput / simple_throughput << "x" << std::endl;
		std::cout << "" << std::endl;

		// destroy the CUDA events
		cudaEventDestroy(launch_begin);
		cudaEventDestroy(launch_end);

		// deallocate device memory
		cudaFree(d_a);
		cudaFree(d_c);
		free(h_a);
		free(h_s);
		free(h_res);
	}
	return 0;
}
